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
import asyncio
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
from starlette.concurrency import run_in_threadpool

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from inference import CandidateScorer, PIPE, ROOT          # noqa: E402
from pdf_utils import extract_upload                        # noqa: E402
from usage_store import make_store                          # noqa: E402
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

# Ceiling on scorings running at once ACROSS ALL jobs and users. BATCH_WORKERS
# only bounds one job, so N simultaneous batches previously put 4N scorings on a
# shared-CPU instance at once. Measured effect: with five batches in flight,
# /health took 63-125s to answer -- the event loop was starved by CPU-bound
# model work holding the GIL. Everything queues here instead, so a busy server
# stays responsive and users wait in line rather than all crawling together.
MAX_CONCURRENT_SCORINGS = max(1, int(
    os.environ.get("ROOKIE_MAX_CONCURRENT_SCORINGS", str(BATCH_WORKERS))))
_SCORE_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_SCORINGS)


def score_one(cv_text: str, jmp_text: str, top_n: int):
    """Every scoring in the process goes through here, so the cap is global."""
    with _SCORE_SLOTS:
        return SCORER.score(cv_text, jmp_text, top_n=top_n)

# --- Auth + CORS -------------------------------------------------------------
# Scoring endpoints require a shared token in the  X-API-Key  header. Set
# API_TOKEN as a secret in the deployment. Fail closed: if it's unset, scoring
# is refused (so the endpoint is never accidentally left open).
API_TOKEN = os.environ.get("API_TOKEN", "")

# Restrict CORS to your frontend origin(s). Set ALLOWED_ORIGINS to a
# comma-separated list of URLs (e.g. your Vercel app). Falls back to "*"
# only if unset -- always set it in production.
#
# Entries are NORMALISED before use, because an Origin header is scheme + host
# + port and nothing else: browsers never send a trailing slash or a path, and
# they lowercase the origin. A value of "https://example.com/" therefore matches
# NOTHING and every preflight fails with "Disallowed CORS origin" -- which is
# invisible for as long as traffic arrives server-side (where CORS does not
# apply) and only breaks when the browser starts calling this API directly.
def _norm_origin(o: str) -> str:
    o = o.strip().rstrip("/")
    if "//" in o:                       # keep scheme://host, drop any path
        scheme, _, rest = o.partition("//")
        o = f"{scheme}//{rest.split('/', 1)[0]}"
    return o.lower()


_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
ALLOW_ORIGINS = [_norm_origin(o) for o in _origins.split(",") if o.strip()] or ["*"]


def require_key(x_api_key: str = Header(default="")):
    """Reject any scoring request without the correct X-API-Key header."""
    if not API_TOKEN:
        raise HTTPException(503, "Server is missing API_TOKEN configuration.")
    if x_api_key != API_TOKEN:
        raise HTTPException(401, "Invalid or missing X-API-Key.")


# --- Per-user spend limits ---------------------------------------------------
# Every scoring spends real money on the DeepSeek + OpenAI keys THIS server
# holds. Registration is open, so without a server-side cap any account can burn
# the operator's credits in a loop -- a client-side counter cannot stop that,
# since the client is the untrusted party. Callers are identified by the
# X-User-Id the authenticated frontend proxy attaches (the browser cannot forge
# it: reaching this API at all requires the API key or a proxy-minted ticket).
#
# Counters live in USAGE (see usage_store): a Redis REST endpoint when one is
# configured, otherwise in-process. The store reports whether it is durable, and
# that flag is passed through to callers rather than assumed -- an in-process
# counter resets on every spin-up of a free instance, which is precisely when a
# user who waited out an idle timeout would get their budget back.
USER_MONTHLY_USD = float(os.environ.get("ROOKIE_USER_MONTHLY_USD", "5"))
USER_HOURLY_SCORES = int(os.environ.get("ROOKIE_USER_HOURLY_SCORES", "60"))

USAGE = make_store()


def user_usage(uid: str) -> dict:
    """This user's spend so far this month and scorings in the current hour."""
    spent = USAGE.month_usd(uid)
    recent = USAGE.hour_count(uid)
    return {
        "uid": uid,
        "month_usd": round(spent, 6),
        "cap_usd": USER_MONTHLY_USD,
        "remaining_usd": round(max(USER_MONTHLY_USD - spent, 0.0), 6),
        "scores_last_hour": recent,
        "hourly_limit": USER_HOURLY_SCORES,
        "enforced": True,
        # True only with an external store. When False the cap still applies
        # within one process lifetime but is NOT a monthly guarantee -- say so
        # rather than letting the number imply one.
        "durable": bool(getattr(USAGE, "durable", False)),
        "store": getattr(USAGE, "name", "memory"),
    }


def enforce_budget(uid: str) -> None:
    """Refuse new work once a user is over budget or scoring too fast.

    This is a pre-check, not a reservation: work already in flight can carry a
    user slightly past the cap before the next request is refused. That is fine
    for a spend guard (the overshoot is one scoring, ~$0.001) and avoids holding
    a lock across a minute of network calls."""
    if not uid or uid == "anonymous":
        return                       # direct API-key callers (ops) are exempt
    u = user_usage(uid)
    if USER_MONTHLY_USD > 0 and u["month_usd"] >= USER_MONTHLY_USD:
        cap = f"{USER_MONTHLY_USD:.2f}" if USER_MONTHLY_USD >= 0.01 else f"{USER_MONTHLY_USD:g}"
        raise HTTPException(429, f"Monthly scoring budget of ${cap} "
                                 f"reached (${u['month_usd']:.4f} used).")
    if USER_HOURLY_SCORES > 0 and u["scores_last_hour"] >= USER_HOURLY_SCORES:
        raise HTTPException(429, f"Rate limit: {USER_HOURLY_SCORES} scorings per hour. "
                                 f"Try again shortly.")


def record_spend(uid: str, usd: float) -> None:
    if not uid:
        return
    USAGE.add(uid, usd)


# --- Upload tickets ----------------------------------------------------------
# The frontend proxies API calls through Vercel so the browser never sees
# API_TOKEN. But Vercel functions cap the REQUEST BODY at 4.5MB, which a batch of
# CV+JMP PDFs blows straight past (FUNCTION_PAYLOAD_TOO_LARGE). So file uploads
# go from the browser DIRECTLY here, authorised by a single-use, short-lived
# ticket that the (authenticated) Vercel route mints with the real API key.
# The long-lived API_TOKEN still never reaches the browser.
TICKET_TTL = int(os.environ.get("ROOKIE_TICKET_TTL", "600"))   # seconds
_TICKETS: dict[str, tuple[float, str]] = {}          # ticket -> (expiry, uid)
_TICKET_LOCK = threading.Lock()


def _issue_ticket(uid: str = "") -> dict:
    """Mint a ticket BOUND to a user, so spend from the resulting upload is
    attributed to them even though the browser sends no API key."""
    now = time.time()
    ticket = uuid.uuid4().hex + uuid.uuid4().hex
    with _TICKET_LOCK:
        # opportunistic sweep so the dict can't grow without bound
        for t in [t for t, (e, _) in _TICKETS.items() if e < now]:
            _TICKETS.pop(t, None)
        _TICKETS[ticket] = (now + TICKET_TTL, uid)
    return {"ticket": ticket, "expires_in": TICKET_TTL}


def _consume_ticket(ticket: str) -> tuple[bool, str]:
    """Validate and burn a ticket. Single-use: a replayed ticket is rejected.
    Returns (ok, uid)."""
    with _TICKET_LOCK:
        entry = _TICKETS.pop(ticket, None)
    if entry is None:
        return False, ""
    expiry, uid = entry
    return expiry >= time.time(), uid


def require_key_or_ticket(x_api_key: str = Header(default=""),
                          x_upload_ticket: str = Header(default=""),
                          x_user_id: str = Header(default="")) -> str:
    """Auth for the upload endpoints: the shared API key OR a valid ticket.
    Returns the caller's user id so their spend can be metered, and refuses
    callers who are over budget."""
    if not API_TOKEN:
        raise HTTPException(503, "Server is missing API_TOKEN configuration.")
    if x_api_key == API_TOKEN:
        uid = x_user_id or "anonymous"
    else:
        ok, ticket_uid = _consume_ticket(x_upload_ticket) if x_upload_ticket else (False, "")
        if not ok:
            raise HTTPException(401, "Invalid or missing X-API-Key / X-Upload-Ticket.")
        # Trust the uid bound at mint time, never a header on this request.
        uid = ticket_uid or "anonymous"
    enforce_budget(uid)
    return uid


print(f"[cors] allowed origins = {ALLOW_ORIGINS}", flush=True)

app = FastAPI(title="Rookie Research Potential API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["*"], allow_headers=["*"],
)

SCORER: CandidateScorer | None = None
JOBS: dict[str, dict] = {}          # in-memory batch jobs (swap for Redis in prod)

# Batch results are confidential: candidate names, parsed CV attributes and
# per-feature SHAP for people who never agreed to be stored. They used to live
# in JOBS until the process restarted, which on a busy instance could be days
# and was retention by accident rather than by policy. They are now deleted a
# fixed time after the job finishes, with a hard cap on how many are held at
# once. The client saves a completed batch locally, so nothing is lost by the
# server forgetting it.
JOB_TTL = int(os.environ.get("ROOKIE_JOB_TTL_SECONDS", "3600"))     # 1 hour
MAX_JOBS = int(os.environ.get("ROOKIE_MAX_JOBS", "200"))
# Backstop so a job wedged in "running" cannot be retained forever.
JOB_MAX_AGE = int(os.environ.get("ROOKIE_JOB_MAX_AGE_SECONDS", str(4 * 3600)))
# How often the background sweeper runs, independent of incoming traffic.
SWEEP_INTERVAL = int(os.environ.get("ROOKIE_SWEEP_INTERVAL_SECONDS", "300"))
_JOBS_LOCK = threading.Lock()


def _touch_job(job: dict) -> None:
    """Restart the retention clock. Called as a job progresses so a long batch
    is never swept mid-run, and again when it finishes."""
    job["expires_at"] = time.time() + JOB_TTL


def _is_expired(job: dict, now: float) -> bool:
    """A RUNNING job never expires on the TTL. The retention clock is only
    meaningful once there is a result to retain, and a single candidate can
    outlive a short TTL on its own — sweeping then would delete work the user
    is still waiting for. JOB_MAX_AGE still bounds a job stuck running."""
    if now >= job.get("created_at", now) + JOB_MAX_AGE:
        return True
    if job.get("status") == "running":
        return False
    return now >= job.get("expires_at", 0)


def sweep_jobs() -> int:
    """Drop expired jobs, then the oldest if we are still over the cap."""
    now = time.time()
    with _JOBS_LOCK:
        for jid in [j for j, v in JOBS.items() if _is_expired(v, now)]:
            JOBS.pop(jid, None)
        if len(JOBS) > MAX_JOBS:
            # Evict finished jobs before running ones: dropping a job still in
            # progress loses work the user paid for and is still waiting on.
            oldest = sorted(JOBS.items(),
                            key=lambda kv: (kv[1].get("status") == "running",
                                            kv[1].get("created_at", 0)))
            for jid, _ in oldest[:len(JOBS) - MAX_JOBS]:
                JOBS.pop(jid, None)
        return len(JOBS)


def new_job(uid: str, total=None) -> dict:
    sweep_jobs()
    return {"status": "running", "done": 0, "total": total, "results": [],
            "cost_usd": 0.0, "uid": uid, "created_at": time.time(),
            "expires_at": time.time() + JOB_TTL}


async def _sweep_loop():
    """Expire results on a timer, so deletion does not depend on traffic.

    Sweeping only inside request handlers means a quiet instance keeps
    confidential results past their window simply because nobody called it --
    retention would be a side effect of load. This runs regardless. It is
    cheap: a pass over at most MAX_JOBS entries under a lock.
    """
    while True:
        await asyncio.sleep(SWEEP_INTERVAL)
        try:
            sweep_jobs()
        except Exception as e:                       # noqa: BLE001
            print(f"[jobs] sweep failed: {e}", flush=True)


@app.on_event("startup")
def _startup():
    global SCORER
    SCORER = CandidateScorer(DATA_CSV, target=TARGET, fetch_2degree=FETCH_2DEG)
    asyncio.get_event_loop().create_task(_sweep_loop())
    print(f"[jobs] retention {JOB_TTL}s, sweeping every {SWEEP_INTERVAL}s", flush=True)


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    # Allow HEAD too so uptime pingers (which default to HEAD) get 200, not 405.
    # Also sweep here: the uptime monitor hits this every few minutes, which
    # backs up the timer if that task ever dies.
    sweep_jobs()
    return {"status": "ok", "target": TARGET,
            "model_ready": SCORER is not None,
            # Which tuned models are live: "optuna" or "grid" (ROOKIE_MODEL).
            "model": getattr(getattr(SCORER, "explainer", None), "model_name", None),
            # Ranking references loaded, e.g. ["C+D+E", "C+D"]; "C+D" is the one
            # a candidate with no readable paper is ranked against.
            "references": ["+".join(k) for k in getattr(getattr(SCORER, "explainer", None), "references", {}) or {}],
            "fetch_2degree": FETCH_2DEG,
            "max_batch": MAX_BATCH,
            "batch_workers": BATCH_WORKERS,
            "max_concurrent_scorings": MAX_CONCURRENT_SCORINGS,
            "user_monthly_usd": USER_MONTHLY_USD,
            "user_hourly_scores": USER_HOURLY_SCORES,
            "usage_store": getattr(USAGE, "name", "memory"),
            "usage_durable": bool(getattr(USAGE, "durable", False)),
            "job_ttl_seconds": JOB_TTL,
            "jobs_held": len(JOBS)}


@app.post("/upload-ticket", dependencies=[Depends(require_key)])
def upload_ticket(x_user_id: str = Header(default="")):
    """Mint a single-use, short-lived ticket that authorises ONE direct upload.
    Called server-side by the frontend proxy (which holds the API key) after it
    has authenticated the user; the browser then uploads straight to this API,
    sidestepping the proxy's 4.5MB body limit.

    Budget is checked HERE as well as on the upload itself, so an over-quota
    user is turned away before they wait through a large file transfer."""
    enforce_budget(x_user_id)
    out = _issue_ticket(x_user_id)
    out["usage"] = user_usage(x_user_id) if x_user_id else None
    return out


@app.get("/usage", dependencies=[Depends(require_key)])
def usage(x_user_id: str = Header(default="")):
    """Server-side truth for a user's metered spend. The client keeps its own
    ledger for instant feedback, but only this one gates anything."""
    if not x_user_id:
        raise HTTPException(422, "X-User-Id is required.")
    return user_usage(x_user_id)


def _read_jmp(filename, data):
    """Uploaded JMP -> title+abstract+intro text for embedding."""
    full = extract_upload(filename, data)
    return extract_jmp_sections(full) if full else ""


# --- Single candidate --------------------------------------------------------
@app.post("/predict")
async def predict(cv: UploadFile = File(...),
                  jmp: UploadFile = File(None),
                  top_n: int = 5,
                  uid: str = Depends(require_key_or_ticket)):
    if SCORER is None:
        raise HTTPException(503, "Model not ready.")
    try:
        cv_text = extract_upload(cv.filename, await cv.read())
        if not cv_text.strip():
            raise HTTPException(422, "Could not extract text from the CV file.")
        jmp_text = _read_jmp(jmp.filename, await jmp.read()) if jmp else ""
        # run_in_threadpool, NOT a direct call: this handler is async, so a
        # synchronous scoring here blocks the event loop for its whole ~35s and
        # stalls every other request in the process -- health checks and other
        # users' job polling included.
        result = await run_in_threadpool(score_one, cv_text, jmp_text, top_n)
        record_spend(uid, (result.get("cost") or {}).get("usd", 0.0))
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Scoring failed: {e}")


# --- Batch: shared scoring loop ----------------------------------------------
def _score_job(job_id: str, candidates: list[dict], top_n: int, uid: str = ""):
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
                res = score_one(c["cv_text"], c["jmp_text"], top_n)
                res["candidate"] = c["name"]
            except Exception as e:
                res = {"candidate": c["name"], "status": "error", "reason": str(e)}
            # Drop the CV and paper text as soon as this candidate is scored,
            # rather than holding every candidate's document in memory until
            # the whole batch finishes.
            c["cv_text"] = c["jmp_text"] = ""

            spent = (res.get("cost", {}) or {}).get("usd", 0.0)
            record_spend(uid, spent)
            with lock:
                slots[i] = res
                job["cost_usd"] = round(job.get("cost_usd", 0.0) + spent, 6)
                job["done"] += 1
                # Publish completed results so far, in order, for live polling.
                job["results"] = [r for r in slots if r is not None]
                _touch_job(job)   # a long batch must not expire while running

        workers = max(1, min(BATCH_WORKERS, n))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, range(n)))

        job["results"] = [r for r in slots if r is not None]
        job["status"] = "done"
        _touch_job(job)           # retention clock starts from completion
    except Exception as e:
        job["status"] = "error"
        job["reason"] = str(e)
        _touch_job(job)


# --- Batch A: a zip of candidate subfolders (each with a CV/JMP) -------------
def _run_batch(job_id: str, zip_bytes: bytes, top_n: int, uid: str = ""):
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
            if len(candidates) > MAX_BATCH:
                raise ValueError(f"Batch limit is {MAX_BATCH} candidates; the "
                                 f"archive contained {len(candidates)}.")
            _score_job(job_id, candidates, top_n, uid)
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["reason"] = str(e)


@app.post("/predict/batch")
async def predict_batch(background: BackgroundTasks,
                        archive: UploadFile = File(...), top_n: int = 5,
                        uid: str = Depends(require_key_or_ticket)):
    """Batch from a .zip of candidate folders (each folder = one candidate with
    a CV and optional JMP; CV/JMP identified by filename + size heuristics)."""
    if SCORER is None:
        raise HTTPException(503, "Model not ready.")
    data = await archive.read()
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = new_job(uid)
    background.add_task(_run_batch, job_id, data, top_n, uid)
    return {"job_id": job_id}


# --- Batch B: loose files as explicit CV+JMP pairs ---------------------------
@app.post("/predict/batch_files")
async def predict_batch_files(background: BackgroundTasks,
                              cv: list[UploadFile] = File(...),
                              jmp: list[UploadFile] = File(default=[]),
                              name: list[str] = Form(default=[]),
                              top_n: int = 5,
                              uid: str = Depends(require_key_or_ticket)):
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
    JOBS[job_id] = new_job(uid, total=len(candidates))
    background.add_task(_score_job, job_id, candidates, top_n, uid)
    return {"job_id": job_id, "total": len(candidates)}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_key)])
def job_status(job_id: str, x_user_id: str = Header(default="")):
    sweep_jobs()
    job = JOBS.get(job_id)
    if job is None:
        # Deliberately does not distinguish expired from never-existed: keeping
        # tombstones to tell them apart would mean retaining a record of every
        # batch ever run, which is the thing the expiry exists to avoid.
        held = f"{JOB_TTL // 60} minutes" if JOB_TTL >= 60 else f"{JOB_TTL} seconds"
        raise HTTPException(404, f"Unknown or expired job_id. Results are kept "
                                 f"for {held} after a batch finishes.")
    # A job id is guessable in principle, and results contain candidate names
    # and CV-derived features. Only hand a job back to the user who created it.
    owner = job.get("uid") or ""
    if owner and x_user_id and owner != x_user_id:
        raise HTTPException(404, "Unknown job_id.")
    return {"status": job["status"],
            "progress": f"{job['done']}/{job['total']}" if job["total"] else "0/?",
            "results": job["results"],
            "cost_usd": round(job.get("cost_usd", 0.0), 6),
            "reason": job.get("reason")}
