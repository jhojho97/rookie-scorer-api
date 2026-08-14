# -*- coding: utf-8 -*-
"""
app.py
------
FastAPI backend for the Accounting Rookie Research-Potential web app.

Endpoints
  GET  /health              -> liveness + which target is loaded
  POST /predict             -> score ONE candidate (cv + optional jmp uploads)
  POST /predict/batch       -> score MANY candidates (a .zip of candidate folders)
  GET  /jobs/{job_id}       -> poll a batch job's status/results

The heavy objects (models, SHAP explainers, reference data, API clients) are
built ONCE at startup. Each /predict request runs the live pipeline
(DeepSeek extraction + BYU + OpenAI embedding + XGBoost + local SHAP) and
returns presentation-neutral JSON for the frontend to render.

Run:
  set DEEPSEEK_API_KEY=...   set OPENAI_API_KEY=...
  cd deepseek_pipeline/webapp
  uvicorn app:app --reload --port 8000
  # interactive docs at http://localhost:8000/docs
"""

import os
import io
import sys
import time
import uuid
import zipfile
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import (FastAPI, UploadFile, File, Form, HTTPException,
                     BackgroundTasks, Depends, Header)
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from inference import CandidateScorer, PIPE, ROOT          # noqa: E402
from pdf_utils import extract_upload                        # noqa: E402
sys.path.append(str(ROOT))
from build_scibert_dataset import (                          # noqa: E402
    find_cv_file, find_jmp_file, extract_text, extract_jmp_sections,
)

# --- Config ------------------------------------------------------------------
DATA_CSV = os.environ.get("ROOKIE_DATA_CSV",
                          str(ROOT / "2015-2018_rookie_dataset.csv"))
TARGET   = os.environ.get("ROOKIE_TARGET", "pub_w_top_5pct")
# fetch_2degree hits BYU author pages per coauthor -> slow. Off by default for
# web latency; set ROOKIE_FETCH_2DEGREE=1 to enable.
FETCH_2DEG = os.environ.get("ROOKIE_FETCH_2DEGREE", "0") == "1"

# Max candidates per batch submission. Cost stays bounded (~$0.06-1.20 for 20)
# and, with BATCH_WORKERS parallelism, a full batch runs in a few minutes rather
# than 10-25. Raise via ROOKIE_MAX_BATCH after moving to a bigger instance.
MAX_BATCH = int(os.environ.get("ROOKIE_MAX_BATCH", "20"))

# How many candidates to score concurrently within one batch job. Per candidate
# the wall time is ~95% network wait (3 LLM extraction calls + BYU scrape +
# embedding), so threads overlap almost perfectly. Peak MEMORY stays at roughly
# one candidate because the only memory-heavy step (TreeSHAP) is serialised
# behind a lock inside CandidateScorer.score -- important on Render's 512MB tier.
BATCH_WORKERS = max(1, int(os.environ.get("ROOKIE_BATCH_WORKERS", "4")))

# --- Auth + CORS -------------------------------------------------------------
# Scoring endpoints require a shared token in the  X-API-Key  header. Set
# API_TOKEN as a secret in the deployment. Fail closed: if it's unset, scoring
# is refused (so the endpoint is never accidentally left open).
API_TOKEN = os.environ.get("API_TOKEN", "")

# Restrict CORS to your frontend origin(s). Set ALLOWED_ORIGINS to a
# comma-separated list of URLs (e.g. your Streamlit Space). Falls back to "*"
# only if unset -- always set it in production.
_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
ALLOW_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()] or ["*"]


def require_key(x_api_key: str = Header(default="")):
    """Reject any scoring request without the correct X-API-Key header."""
    if not API_TOKEN:
        raise HTTPException(503, "Server is missing API_TOKEN configuration.")
    if x_api_key != API_TOKEN:
        raise HTTPException(401, "Invalid or missing X-API-Key.")


# --- Upload tickets ----------------------------------------------------------
# The frontend proxies API calls through Vercel so the browser never sees
# API_TOKEN. But Vercel functions cap the REQUEST BODY at 4.5MB, which a batch of
# CV+JMP PDFs blows straight past (FUNCTION_PAYLOAD_TOO_LARGE). So file uploads
# go from the browser DIRECTLY here, authorised by a single-use, short-lived
# ticket that the (authenticated) Vercel route mints with the real API key.
# The long-lived API_TOKEN still never reaches the browser.
TICKET_TTL = int(os.environ.get("ROOKIE_TICKET_TTL", "600"))   # seconds
_TICKETS: dict[str, float] = {}                                # ticket -> expiry
_TICKET_LOCK = threading.Lock()


def _issue_ticket() -> dict:
    now = time.time()
    ticket = uuid.uuid4().hex + uuid.uuid4().hex
    with _TICKET_LOCK:
        # opportunistic sweep so the dict can't grow without bound
        for t, exp in [(t, e) for t, e in _TICKETS.items() if e < now]:
            _TICKETS.pop(t, None)
        _TICKETS[ticket] = now + TICKET_TTL
    return {"ticket": ticket, "expires_in": TICKET_TTL}


def _consume_ticket(ticket: str) -> bool:
    """Validate and burn a ticket. Single-use: a replayed ticket is rejected."""
    with _TICKET_LOCK:
        expiry = _TICKETS.pop(ticket, None)
    return expiry is not None and expiry >= time.time()


def require_key_or_ticket(x_api_key: str = Header(default=""),
                          x_upload_ticket: str = Header(default="")):
    """Auth for the upload endpoints: the shared API key OR a valid ticket."""
    if not API_TOKEN:
        raise HTTPException(503, "Server is missing API_TOKEN configuration.")
    if x_api_key == API_TOKEN:
        return
    if x_upload_ticket and _consume_ticket(x_upload_ticket):
        return
    raise HTTPException(401, "Invalid or missing X-API-Key / X-Upload-Ticket.")


app = FastAPI(title="Rookie Research Potential API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["*"], allow_headers=["*"],
)

SCORER: CandidateScorer | None = None
JOBS: dict[str, dict] = {}          # in-memory batch jobs (swap for Redis in prod)


@app.on_event("startup")
def _startup():
    global SCORER
    SCORER = CandidateScorer(DATA_CSV, target=TARGET, fetch_2degree=FETCH_2DEG)


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    # Allow HEAD too so uptime pingers (which default to HEAD) get 200, not 405.
    return {"status": "ok", "target": TARGET,
            "model_ready": SCORER is not None,
            "fetch_2degree": FETCH_2DEG,
            "max_batch": MAX_BATCH,
            "batch_workers": BATCH_WORKERS}


@app.post("/upload-ticket", dependencies=[Depends(require_key)])
def upload_ticket():
    """Mint a single-use, short-lived ticket that authorises ONE direct upload.
    Called server-side by the frontend proxy (which holds the API key) after it
    has authenticated the user; the browser then uploads straight to this API,
    sidestepping the proxy's 4.5MB body limit."""
    return _issue_ticket()


def _read_jmp(filename, data):
    """Uploaded JMP -> title+abstract+intro text for embedding."""
    full = extract_upload(filename, data)
    return extract_jmp_sections(full) if full else ""


# --- Single candidate --------------------------------------------------------
@app.post("/predict", dependencies=[Depends(require_key_or_ticket)])
async def predict(cv: UploadFile = File(...),
                  jmp: UploadFile = File(None),
                  top_n: int = 5):
    if SCORER is None:
        raise HTTPException(503, "Model not ready.")
    try:
        cv_text = extract_upload(cv.filename, await cv.read())
        if not cv_text.strip():
            raise HTTPException(422, "Could not extract text from the CV file.")
        jmp_text = _read_jmp(jmp.filename, await jmp.read()) if jmp else ""
        result = SCORER.score(cv_text, jmp_text, top_n=top_n)
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Scoring failed: {e}")


# --- Batch: shared scoring loop ----------------------------------------------
def _score_job(job_id: str, candidates: list[dict], top_n: int):
    """Score a prepared list of candidates, up to BATCH_WORKERS at a time. Each
    candidate is a dict with keys name / cv_text / jmp_text.

    Concurrency is safe here because a candidate's wall time is dominated by
    network waits (LLM extraction, BYU lookup, embedding) while the memory-heavy
    TreeSHAP step is serialised inside CandidateScorer.score -- so peak memory
    stays at ~one candidate no matter how many workers run.

    Results are written back IN SUBMISSION ORDER (slot per candidate) so the
    output does not shuffle just because one candidate finished first."""
    job = JOBS[job_id]
    try:
        n = len(candidates)
        job["total"] = n
        slots: list[dict | None] = [None] * n
        lock = threading.Lock()

        def run(i: int) -> None:
            c = candidates[i]
            try:
                res = SCORER.score(c["cv_text"], c["jmp_text"], top_n=top_n)
                res["candidate"] = c["name"]
            except Exception as e:
                res = {"candidate": c["name"], "status": "error", "reason": str(e)}
            with lock:
                slots[i] = res
                job["cost_usd"] = round(
                    job.get("cost_usd", 0.0) + (res.get("cost", {}) or {}).get("usd", 0.0), 6)
                job["done"] += 1
                # Publish completed results so far, in order, for live polling.
                job["results"] = [r for r in slots if r is not None]

        workers = max(1, min(BATCH_WORKERS, n))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, range(n)))

        job["results"] = [r for r in slots if r is not None]
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["reason"] = str(e)


# --- Batch A: a zip of candidate subfolders (each with a CV/JMP) -------------
def _run_batch(job_id: str, zip_bytes: bytes, top_n: int):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                z.extractall(tmp)
            # candidate folders = immediate subdirs that contain a document
            root = Path(tmp)
            folders = [d for d in root.rglob("*") if d.is_dir()
                       and any(f.suffix.lower() in {".pdf", ".doc", ".docx"}
                               for f in d.iterdir() if f.is_file())]
            candidates = []
            for folder in sorted(folders, key=lambda p: p.name.lower()):
                cvf = find_cv_file(folder)
                jmf = find_jmp_file(folder, cv_file=cvf)
                cv_text = extract_text(cvf) if cvf else ""
                jfull = extract_text(jmf) if jmf else ""
                jmp_text = extract_jmp_sections(jfull) if jfull else ""
                candidates.append({"name": folder.name,
                                   "cv_text": cv_text, "jmp_text": jmp_text})
            _score_job(job_id, candidates, top_n)
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["reason"] = str(e)


@app.post("/predict/batch", dependencies=[Depends(require_key_or_ticket)])
async def predict_batch(background: BackgroundTasks,
                        archive: UploadFile = File(...), top_n: int = 5):
    """Batch from a .zip of candidate folders (each folder = one candidate with
    a CV and optional JMP; CV/JMP identified by filename + size heuristics)."""
    if SCORER is None:
        raise HTTPException(503, "Model not ready.")
    data = await archive.read()
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "done": 0, "total": None,
                    "results": [], "cost_usd": 0.0}
    background.add_task(_run_batch, job_id, data, top_n)
    return {"job_id": job_id}


# --- Batch B: loose files as explicit CV+JMP pairs ---------------------------
@app.post("/predict/batch_files", dependencies=[Depends(require_key_or_ticket)])
async def predict_batch_files(background: BackgroundTasks,
                              cv: list[UploadFile] = File(...),
                              jmp: list[UploadFile] = File(default=[]),
                              name: list[str] = Form(default=[]),
                              top_n: int = 5):
    """Batch from loose files, paired by position: candidate i = cv[i] + jmp[i].
    `jmp` is optional; if provided it must have the same length as `cv` (send a
    0-byte file for a candidate that has no JMP). `name` optionally labels each
    candidate (defaults to the CV filename stem). Text is extracted here (uploads
    can't be read off-thread); scoring runs in the background -> returns job_id."""
    if SCORER is None:
        raise HTTPException(503, "Model not ready.")
    if not cv:
        raise HTTPException(422, "Provide at least one cv file.")
    if len(cv) > MAX_BATCH:
        raise HTTPException(422, f"Batch limit is {MAX_BATCH} candidates per "
                                 f"submission; you sent {len(cv)}.")
    if jmp and len(jmp) != len(cv):
        raise HTTPException(422, "If jmp files are provided, there must be exactly "
                                 "one per cv, in the same order (use a 0-byte file "
                                 "for a candidate without a JMP).")
    candidates = []
    for i, cvf in enumerate(cv):
        cv_text = extract_upload(cvf.filename, await cvf.read())
        jmp_text = ""
        if jmp:
            jbytes = await jmp[i].read()
            if jbytes:                      # 0-byte placeholder => no JMP
                jmp_text = _read_jmp(jmp[i].filename, jbytes)
        label = (name[i] if i < len(name) and name[i].strip()
                 else (Path(cvf.filename).stem if cvf.filename else f"candidate_{i+1}"))
        candidates.append({"name": label, "cv_text": cv_text, "jmp_text": jmp_text})
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "done": 0,
                    "total": len(candidates), "results": [], "cost_usd": 0.0}
    background.add_task(_score_job, job_id, candidates, top_n)
    return {"job_id": job_id, "total": len(candidates)}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_key)])
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id.")
    return {"status": job["status"],
            "progress": f"{job['done']}/{job['total']}" if job["total"] else "0/?",
            "results": job["results"],
            "cost_usd": round(job.get("cost_usd", 0.0), 6),
            "reason": job.get("reason")}
