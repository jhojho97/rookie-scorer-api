# -*- coding: utf-8 -*-
"""
extract_deepseek.py
-------------------
Shared Stage-1 step for Sets C and D: run the paper's Table OA1 prompt (plus
the gender + research-area/method aux prompts) through DeepSeek for every
candidate, keyed by the dataset ID, and cache one JSON per ID.

Both build_set_C.py and build_set_D.py read these cached JSONs, so DeepSeek is
called once per candidate (not twice).

Output: <out_dir>/raw_json/{ID}.json

Usage:
  set DEEPSEEK_API_KEY=...   (or export on mac/linux)
  python extract_deepseek.py --data_dir ../data --main_csv ../data/2015-2018_rookie_dataset.csv
  python extract_deepseek.py ... --limit 5     # quick test
"""

import os
import re
import sys
import json
import time
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from build_scibert_dataset import (
    extract_text, find_cv_file, find_jmp_file, extract_jmp_sections,
)
from prompts import OA1_PROMPT, GENDER_PROMPT, RESEARCH_CLASSIFY_PROMPT
from common import build_candidate_index

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("pip install openai")

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# Override the extraction model without editing code, e.g.
#   PowerShell:  $env:DEEPSEEK_MODEL = "deepseek-reasoner"
# Default is v4-flash, NOT the `deepseek-chat` alias: since the alias was
# deprecated (2026/07/24) it routes to v4-pro, which roughly DOUBLED scoring
# latency and TRIPLED cost for no measured accuracy gain on this extraction.
DEEPSEEK_MODEL    = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
MAX_CV_CHARS      = 30000
MAX_RETRIES       = 4
# Output ceiling for the OA1 JSON. This was 4096, which is NOT enough: the OA1
# prompt asks for every abstract in the CV plus the full papers / coauthors /
# presentations / awards / reviewer / membership / references lists, and a real
# CV overruns it. The response was then cut off mid-JSON, failed to parse, and
# -- because temperature is 0 and the call is deterministic -- failed identically
# on all four retries before returning an EMPTY extraction. Measured on one real
# candidate: 31,239 tokens and $0.02 spent to produce nothing, 7.8x the cost of
# a single successful call.
MAX_OUTPUT_TOKENS = int(os.environ.get("DEEPSEEK_MAX_OUTPUT_TOKENS", "16384"))
# One growth step if even that overruns, before giving up with a clear error.
MAX_OUTPUT_CEILING = int(os.environ.get("DEEPSEEK_MAX_OUTPUT_CEILING", "32768"))


def _is_transient(exc) -> bool:
    """Worth retrying? Rate limits and server faults are; a bad model id, a bad
    key or a rejected parameter will fail the same way every time, so retrying
    those just multiplies the bill."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        return True                      # network/timeout: no status, retry
    return status == 429 or status >= 500


def get_client():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY environment variable.")
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)


def call_json(client, prompt, user_text="", model=None, meter=None):
    """Run one JSON-mode completion.

    Returns (data, error). `error` is None on success; on failure `data` is {}
    and `error` says why, so the caller can report a degraded extraction instead
    of silently scoring an empty feature row.

    Retries are deliberately narrow. The call is deterministic (temperature 0),
    so repeating it after a truncated or unparseable response reproduces the
    same response exactly -- the old code did that four times and paid for it
    each time. Only a transient fault is worth another attempt; a response that
    ran out of room gets more room instead.
    """
    model = model or DEEPSEEK_MODEL
    budget = MAX_OUTPUT_TOKENS
    last_err = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": prompt + ("\n" + user_text if user_text else "")}],
                response_format={"type": "json_object"},
                temperature=0.0, max_tokens=budget,
            )
            # record token usage as soon as the call succeeds (tokens are spent
            # even if JSON parsing below fails and we fall back to regex).
            if meter is not None and getattr(resp, "usage", None) is not None:
                u = resp.usage
                meter.add("extraction", model,
                          getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))

            choice = resp.choices[0]
            content = choice.message.content or ""

            # Cut off at the output ceiling: the JSON is incomplete by
            # definition. Grow the budget once rather than re-running the
            # identical call and truncating in the identical place.
            if getattr(choice, "finish_reason", None) == "length":
                if budget < MAX_OUTPUT_CEILING:
                    warnings.warn(f"[extract] response hit the {budget}-token output "
                                  f"limit; retrying once at {MAX_OUTPUT_CEILING}.")
                    budget = MAX_OUTPUT_CEILING
                    continue
                return {}, (f"the model's reply exceeded the {budget}-token output "
                            f"limit even after being raised; the CV may be unusually long")

            try:
                return json.loads(content), None
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", content, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0)), None
                    except json.JSONDecodeError:
                        pass
                # Complete but unparseable -> deterministic, so do not hammer it.
                return {}, "the model did not return valid JSON"

        except Exception as e:                       # noqa: BLE001 - reported upward
            last_err = e
            if not _is_transient(e):
                warnings.warn(f"[extract] model={model!r} call failed permanently ({e}); "
                              f"not retrying.")
                return {}, f"the extraction service rejected the request ({e})"
            wait = 2 ** attempt
            warnings.warn(f"[extract] model={model!r} call failed ({e}); retry in {wait}s.")
            time.sleep(wait)

    warnings.warn(f"[extract] model={model!r} FAILED after {MAX_RETRIES} attempts "
                  f"(last error: {last_err}).")
    return {}, f"the extraction service did not respond ({last_err})"


def extract_one(client, cv_text, jmp_text, model=None, meter=None):
    """Extract one candidate.

    Also reports whether the extraction actually worked. A failed OA1 call used
    to return {} silently, which is indistinguishable downstream from a CV that
    genuinely contains nothing: Set C comes out all-defaults, the model scores
    that empty row, and the candidate gets a confident-looking number computed
    from no information at all. Callers must be able to tell the two apart.
    """
    cv_text = (cv_text or "")[:MAX_CV_CHARS]
    problems = []

    if cv_text.strip():
        oa1, err = call_json(client, OA1_PROMPT, cv_text, model=model, meter=meter)
        if err:
            problems.append(f"Could not read the CV: {err}.")
    else:
        oa1, problems = {}, ["No readable text was found in the CV file."]

    # The gender and research-area calls both consume OA1's output, so they must
    # follow it -- but they do not depend on EACH OTHER. Running them
    # concurrently turns a 3-call chain into 2 round-trips, cutting roughly a
    # third off extraction latency (they are pure network waits).
    name = str(oa1.get("name", "") or "").strip()
    research_interest = str(oa1.get("research interest", "") or "")
    papers_str = json.dumps(oa1.get("papers", ""))[:4000] if oa1.get("papers") else ""

    def _gender():
        if not name:
            return {}, None
        return call_json(client, GENDER_PROMPT.format(name=name), model=model, meter=meter)

    def _research():
        if not (research_interest or papers_str):
            return {}, None
        return call_json(client, RESEARCH_CLASSIFY_PROMPT.format(
            research_interest=research_interest[:4000], papers=papers_str),
            model=model, meter=meter)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_gender, f_research = pool.submit(_gender), pool.submit(_research)
        (gender, gender_err), (research, research_err) = f_gender.result(), f_research.result()

    if gender_err:
        problems.append(f"Could not infer gender: {gender_err}.")
    if research_err:
        problems.append(f"Could not classify research area: {research_err}.")

    return {
        "oa1": oa1,
        "gender": gender.get("gender", ""),
        "primary_area": research.get("primary_area", ""),
        "primary_method": research.get("primary_method", ""),
        "jmp_text": jmp_text or "",
        # False => Set C is all-defaults and any score built on it is meaningless.
        "cv_extraction_ok": bool(oa1),
        "extraction_problems": problems,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with {year}/ subfolders")
    ap.add_argument("--main_csv", required=True, help="2015-2018_rookie_dataset.csv (for IDs)")
    ap.add_argument("--out_dir", default="deepseek_output")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    json_dir = Path(args.out_dir) / "raw_json"
    json_dir.mkdir(parents=True, exist_ok=True)

    index = build_candidate_index(args.data_dir, args.main_csv)
    if args.limit:
        index = index.head(args.limit)

    client = get_client()

    for n, row in enumerate(index.itertuples(index=False)):
        cid, folder = int(row.ID), Path(row.folder)
        out_path = json_dir / f"{cid}.json"
        if out_path.exists():
            print(f"[{n+1}/{len(index)}] ID={cid} cached")
            continue

        cv_file  = find_cv_file(folder)
        jmp_file = find_jmp_file(folder, cv_file=cv_file)
        cv_text  = extract_text(cv_file) if cv_file else ""
        jmp_full = extract_text(jmp_file) if jmp_file else ""
        jmp_text = extract_jmp_sections(jmp_full) if jmp_full else ""

        print(f"[{n+1}/{len(index)}] ID={cid} {folder.name!r} "
              f"cv={bool(cv_text)} jmp={bool(jmp_text)} ...", flush=True)

        rec = extract_one(client, cv_text, jmp_text)
        rec["ID"] = cid
        rec["folder_name"] = folder.name
        out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print(f"\nDone -> {json_dir}")


if __name__ == "__main__":
    main()
