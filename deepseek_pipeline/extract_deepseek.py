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
DEEPSEEK_MODEL    = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
MAX_CV_CHARS      = 30000
MAX_RETRIES       = 4


def get_client():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY environment variable.")
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)


def call_json(client, prompt, user_text="", model=None):
    model = model or DEEPSEEK_MODEL
    content = ""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": prompt + ("\n" + user_text if user_text else "")}],
                response_format={"type": "json_object"},
                temperature=0.0, max_tokens=4096,
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content or "", re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            warnings.warn(f"[extract] model={model!r} call failed ({e}); retry in {wait}s.")
            time.sleep(wait)
    warnings.warn(f"[extract] model={model!r} FAILED after {MAX_RETRIES} retries "
                  f"(last error: {last_err}). Returning EMPTY extraction -> Set C "
                  f"will be all-default for this candidate. Check model access / key.")
    return {}


def extract_one(client, cv_text, jmp_text, model=None):
    cv_text = (cv_text or "")[:MAX_CV_CHARS]
    oa1 = call_json(client, OA1_PROMPT, cv_text, model=model) if cv_text.strip() else {}

    name = str(oa1.get("name", "") or "").strip()
    gender = call_json(client, GENDER_PROMPT.format(name=name), model=model) if name else {}

    research_interest = str(oa1.get("research interest", "") or "")
    papers_str = json.dumps(oa1.get("papers", ""))[:4000] if oa1.get("papers") else ""
    research = {}
    if research_interest or papers_str:
        research = call_json(client, RESEARCH_CLASSIFY_PROMPT.format(
            research_interest=research_interest[:4000], papers=papers_str), model=model)

    return {
        "oa1": oa1,
        "gender": gender.get("gender", ""),
        "primary_area": research.get("primary_area", ""),
        "primary_method": research.get("primary_method", ""),
        "jmp_text": jmp_text or "",
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
