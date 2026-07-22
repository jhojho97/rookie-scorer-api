# -*- coding: utf-8 -*-
"""
build_set_C.py
--------------
Build Set C (27 structured CV variables) from the cached DeepSeek JSONs.
Output: set_C.csv  - columns: ID + SET_C_COLS.

Prerequisite: run extract_deepseek.py first (produces raw_json/{ID}.json).

Usage:
  python build_set_C.py --data_dir ../data --main_csv ../data/2015-2018_rookie_dataset.csv
  python build_set_C.py ... --json_dir deepseek_output/raw_json --out set_C.csv
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

from pathlib import Path
from common import (
    SET_C_COLS, build_candidate_index, load_raw_json_by_folder, to_int, as_list,
)


def build_set_c(rec: dict) -> dict:
    oa1 = rec.get("oa1", {}) or {}
    coauthors = as_list(oa1.get("coauthor list"))
    references = as_list(oa1.get("references"))
    languages = as_list(oa1.get("language list"))
    area = str(rec.get("primary_area", "") or "").lower()
    method = str(rec.get("primary_method", "") or "").lower()

    return {
        "gender": to_int(rec.get("gender", ""), default=0),
        "has Bachelor honor": to_int(oa1.get("has Bachelor honor")),
        "has Master honor": to_int(oa1.get("has Master honor")),
        "has PhD honor": to_int(oa1.get("has PhD honor")),
        "number of published papers": to_int(oa1.get("number of published papers")),
        "number of R&R papers": to_int(oa1.get("number of R&R papers")),
        "number of papers in progress": to_int(oa1.get("number of papers in progress")),
        "has_coauthor": 1 if (coauthors or to_int(oa1.get("number of coauthors")) > 0) else 0,
        "has_reference": 1 if references else 0,
        "number of coauthors": to_int(oa1.get("number of coauthors")),
        "number of presentations": to_int(oa1.get("number of presentations")),
        "number of teaching experiences": to_int(oa1.get("number of teaching experiences")),
        "number of awards": to_int(oa1.get("number of awards")),
        "number of reviewers": to_int(oa1.get("number of reviewers")),
        "number of membership": to_int(oa1.get("number of membership")),
        "number of working experiences": to_int(oa1.get("number of working experiences")),
        "had academic work": to_int(oa1.get("had academic work")),
        "had non-academic work": to_int(oa1.get("had non-academic work")),
        "provide abstract": to_int(oa1.get("provide abstract")),
        "PrimaryResearchArea_financial": int(area == "financial"),
        "PrimaryResearchArea_auditing": int(area == "auditing"),
        "PrimaryResearchArea_managerial": int(area == "managerial"),
        "PrimaryResearchArea_tax": int(area == "tax"),
        "PrimaryResearchMethod_archival": int(method == "archival"),
        "PrimaryResearchMethod_experiment": int(method == "experiment"),
        "PrimaryResearchMethod_analytical": int(method == "analytical"),
        "multi_language": int(len(languages) > 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--main_csv", required=True)
    ap.add_argument("--json_dir", default="deepseek_output/raw_json")
    ap.add_argument("--out", default="set_C.csv")
    args = ap.parse_args()

    index = build_candidate_index(args.data_dir, args.main_csv)
    json_by_folder = load_raw_json_by_folder(args.json_dir)   # keyed by folder_name

    rows, missing = [], 0
    for r in index.itertuples(index=False):
        cid = int(r.ID)
        rec = json_by_folder.get(Path(r.folder).name)
        if rec is None:
            missing += 1
            row = {c: 0 for c in SET_C_COLS}   # no extraction -> zeros
        else:
            row = build_set_c(rec)
        row = {"ID": cid, **row}
        rows.append(row)

    if missing:
        warnings.warn(f"{missing} candidates had no raw_json - run extract_deepseek.py "
                      f"first. Their Set C rows are all zeros.")

    df = pd.DataFrame(rows)[["ID"] + SET_C_COLS]
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols -> {args.out}")


if __name__ == "__main__":
    main()
