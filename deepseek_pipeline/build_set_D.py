# -*- coding: utf-8 -*-
"""
build_set_D.py
--------------
Build Set D (16 external public-source variables) from the cached DeepSeek
JSONs joined against the same online sources the paper used:
  - BYU Accounting Research Rankings  (coauthor/reference rank features)
  - US News top-50 business schools   (university *_top flags)
  - the 11 top journals / conference list (field knowledge)

Output: set_D.csv - columns: ID + SET_D_COLS.

Prerequisites:
  - run extract_deepseek.py first (raw_json/{ID}.json)
  - run fetch_byu_rankings.py for the BYU year(s) you need
  - reference_data/us_news_top50.csv, top_journals.csv present

Usage:
  python build_set_D.py --data_dir ../data --main_csv ../data/2015-2018_rookie_dataset.csv \
      --byu_year_for 2015:2014 2016:2015 2017:2016 2018:2017 --fetch_2degree
"""

import re
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    SET_D_COLS, build_candidate_index, load_raw_json_by_folder, to_int, as_list,
    norm_inst, name_key, ASIAN_LANGUAGES, EUROPEAN_LANGUAGES,
)

REF_DIR = Path(__file__).resolve().parent / "reference_data"
_COAUTHOR_CACHE = {}


# -- Reference-table loaders ---------------------------------------------------
def load_top_universities():
    p = REF_DIR / "us_news_top50.csv"
    if not p.exists():
        warnings.warn(f"{p} missing - *_top flags = 0.")
        return set()
    return {norm_inst(u) for u in pd.read_csv(p)["university"].dropna()}


def load_top_journals():
    p = REF_DIR / "top_journals.csv"
    if not p.exists():
        warnings.warn(f"{p} missing - top-journal counts = 0.")
        return set()
    return {norm_inst(j) for j in pd.read_csv(p)["journal"].dropna()}


def load_canonical_map(filename):
    """Load a DeepSeek canonical map (raw_name_key -> canonical or null)."""
    import json
    p = REF_DIR / filename
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _lk(s):
    return str(s or "").strip().lower()


def load_byu_year(year, rank_window="all"):
    """Load reference_data/byu_scholar_ranks_{year}.csv keyed by name_key."""
    rank_col = {"all": "rank_all", "6yr": "rank_6yr", "12yr": "rank_12yr"}.get(rank_window, "rank_all")
    p = REF_DIR / f"byu_scholar_ranks_{year}.csv"
    if not p.exists():
        warnings.warn(f"{p} missing - coauthor/reference ranks = 0 for that year. "
                      f"Run fetch_byu_rankings.py --year {year}.")
        return {}
    df = pd.read_csv(p)
    if rank_col not in df.columns:
        rank_col = "rank" if "rank" in df.columns else None
    out = {}
    li_best = {}        # (last, initial) -> best info seen
    li_ids = {}         # (last, initial) -> set of distinct authorids (for ambiguity)
    for _, r in df.iterrows():
        rank = float(r.get(rank_col, 0) or 0) if rank_col else 0.0
        aid = int(r["authorid"]) if "authorid" in df.columns and pd.notna(r.get("authorid")) else None
        info = {"rank": rank, "authorid": aid,
                "is_top1pct": int(r.get("is_top1pct", 0) or 0)}

        k = name_key(r["name"])
        if k:
            prev = out.get(k)
            if prev is None or (rank > 0 and (prev["rank"] == 0 or rank < prev["rank"])):
                out[k] = info

        lk = li_key_byu(r["name"])
        if lk:
            li_ids.setdefault(lk, set()).add(aid if aid is not None else r["name"])
            prev = li_best.get(lk)
            if prev is None or (rank > 0 and (prev["rank"] == 0 or rank < prev["rank"])):
                li_best[lk] = info

    # keep last+initial keys that map to exactly ONE scholar (unambiguous)
    out["__li__"] = {lk: li_best[lk] for lk, ids in li_ids.items() if len(ids) == 1}
    return out


def fetch_author_coauthor_count(authorid):
    if authorid in _COAUTHOR_CACHE:
        return _COAUTHOR_CACHE[authorid]
    try:
        import requests, re
        url = ("https://www.byuaccounting.net/rankings/indrank/"
               f"per_ind_cnt.php?authorid={authorid}")
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (academic)"}, timeout=30)
        resp.raise_for_status()
        ids = {int(x) for x in re.findall(r"authorid=(\d+)", resp.text)}
        ids.discard(authorid)
        count = len(ids)
    except Exception:
        count = 0
    _COAUTHOR_CACHE[authorid] = count
    return count


# -- Name matching: last-name + first-initial fallback ------------------------
# Recall of full-token matching is low (initials like "A. Mohan", nicknames like
# "Eddie" vs "Edward" fail). This fallback matches on (last-name-token, first
# initial), used ONLY when it resolves to a single BYU scholar (so common names
# like "Kim, S" stay ambiguous and are skipped rather than mis-matched).

def _alpha_tokens(s):
    s = re.sub(r"[^a-z ]", " ", str(s or "").lower())
    return [t for t in s.split() if t]


def li_key_cv(name):
    """(last-token, first-initial) from a CV name written 'First [M] Last'."""
    toks = _alpha_tokens(name)
    if len(toks) < 2:
        return None
    return (toks[-1], toks[0][0])


def li_key_byu(name):
    """(last-token, first-initial) from a BYU name 'Last, First M.'."""
    name = str(name or "")
    if "," in name:
        last_part, given_part = name.split(",", 1)
        lt, gt = _alpha_tokens(last_part), _alpha_tokens(given_part)
        if not lt or not gt:
            return None
        return (lt[-1], gt[0][0])
    return li_key_cv(name)


def resolve_scholar(name, byu):
    """Full-token match first, then unambiguous last-name+first-initial fallback."""
    info = byu.get(name_key(name))
    if info:
        return info
    li = byu.get("__li__")
    if li is not None:
        return li.get(li_key_cv(name))   # None if ambiguous or absent
    return None


# The 7 Set D columns derived from BYU rankings. For dataset REGENERATION these
# should be taken from the original dataset (the 2026 scrape has drifted); the
# live scrape is only for scoring NEW candidates in the web-app pipeline.
BYU_DATASET_COLS = [
    "coauthor_mean", "coauthor_high", "number_of_coauthor_2degree", "coauthor_top",
    "reference_first", "reference_mean", "reference_high",
]


# -- Feature builder -----------------------------------------------------------
def build_set_d(rec, top_unis, top_journals, byu, fetch_2degree=False,
                uni_map=None, journal_map=None, byu_row=None):
    oa1 = rec.get("oa1", {}) or {}

    # If a DeepSeek canonical map is present, a name is "top" when it resolves to
    # a non-null reference entry. Otherwise fall back to normalized exact match.
    def is_top_uni(name):
        if uni_map is not None:
            return uni_map.get(_lk(name)) is not None
        return norm_inst(name) in top_unis if top_unis else False

    def is_top_journal(name):
        if journal_map is not None:
            return journal_map.get(_lk(name)) is not None
        return norm_inst(name) in top_journals if top_journals else False

    def uni_top(key):
        return int(is_top_uni(oa1.get(key, "")))

    visits = as_list(oa1.get("visiting experience"))
    visit_top = int(any(is_top_uni(u) for u in visits))

    n_top_pub = sum(1 for j in as_list(oa1.get("published journal list")) if is_top_journal(j))
    n_top_rr = sum(1 for j in as_list(oa1.get("R&R journal list")) if is_top_journal(j))

    if byu_row is not None:
        # Dataset-regeneration path: take the paper's BYU values directly by ID.
        def _v(c):
            v = byu_row.get(c, 0)
            return 0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        coauthor_mean = _v("coauthor_mean")
        coauthor_high = _v("coauthor_high")
        co_2deg = _v("number_of_coauthor_2degree")
        co_top = int(_v("coauthor_top"))
        reference_first = _v("reference_first")
        reference_mean = _v("reference_mean")
        reference_high = _v("reference_high")
    else:
        # Live path (web-app scoring of a new candidate): compute from the scrape.
        coauthors = as_list(oa1.get("coauthor list"))
        co_ranks, co_top, co_2deg = [], 0, 0.0
        for c in coauthors:
            info = resolve_scholar(c, byu)
            if info:
                if info["rank"] > 0:
                    co_ranks.append(info["rank"])
                if info["is_top1pct"]:
                    co_top = 1
                if fetch_2degree and info.get("authorid"):
                    co_2deg += fetch_author_coauthor_count(info["authorid"])
        coauthor_mean = float(np.mean(co_ranks)) if co_ranks else 0.0
        coauthor_high = float(np.min(co_ranks)) if co_ranks else 0.0   # best = min rank

        references = as_list(oa1.get("references"))
        ref_infos = [resolve_scholar(r, byu) for r in references]
        ref_ranks = [i["rank"] for i in ref_infos if i and i["rank"] > 0]
        first = resolve_scholar(references[0], byu) if references else None
        reference_first = float(first["rank"]) if first else 0.0
        reference_mean = float(np.mean(ref_ranks)) if ref_ranks else 0.0
        reference_high = float(np.min(ref_ranks)) if ref_ranks else 0.0

    langs = [str(l).lower().strip() for l in as_list(oa1.get("language list"))]
    non_eng = [l for l in langs if l != "english"]

    return {
        "Bachelor_top": uni_top("Bachelor degree"),
        "Master_top": uni_top("Master degree"),
        "PhD_top": uni_top("PhD degree"),
        "visit_top": visit_top,
        "number of top published papers": n_top_pub,
        "number of top R&R papers": n_top_rr,
        "coauthor_mean": coauthor_mean,
        "coauthor_high": coauthor_high,
        "number_of_coauthor_2degree": float(co_2deg),
        "coauthor_top": co_top,
        "number of presentations on top conferences":
            to_int(oa1.get("number of presentations at top conferences")),
        "reference_first": reference_first,
        "reference_mean": reference_mean,
        "reference_high": reference_high,
        "second_language_asia": int(any(l in ASIAN_LANGUAGES for l in non_eng)),
        "second_language_euro": int(any(l in EUROPEAN_LANGUAGES for l in non_eng)),
    }


def parse_byu_map(pairs):
    """['2015:2014','2018:2017'] -> {2015:2014, ...} (batch_year -> BYU display year)."""
    m = {}
    for p in pairs or []:
        b, y = p.split(":")
        m[int(b)] = int(y)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--main_csv", required=True)
    ap.add_argument("--json_dir", default="deepseek_output/raw_json")
    ap.add_argument("--out", default="set_D.csv")
    ap.add_argument("--byu_year_for", nargs="*", default=[],
                    help="Map batch year -> BYU display year, e.g. 2018:2017 2017:2016")
    ap.add_argument("--byu_rank_window", choices=["all", "6yr", "12yr"], default="all")
    ap.add_argument("--fetch_2degree", action="store_true")
    ap.add_argument("--byu_from_dataset", action="store_true",
                    help="Take the 7 BYU rank columns from the ORIGINAL dataset "
                         "(main_csv) by ID instead of the 2026 scrape. Use this to "
                         "REGENERATE the training dataset (the scrape has drifted). "
                         "Omit it only for the web-app path scoring NEW candidates.")
    args = ap.parse_args()

    index = build_candidate_index(args.data_dir, args.main_csv)
    top_unis = load_top_universities()
    top_journals = load_top_journals()

    # For dataset regeneration: load the paper's BYU rank columns by ID.
    byu_dataset = None
    if args.byu_from_dataset:
        have = [c for c in BYU_DATASET_COLS
                if c in pd.read_csv(args.main_csv, nrows=0).columns]
        dfm = pd.read_csv(args.main_csv, usecols=["ID"] + have)
        byu_dataset = dfm.set_index("ID").to_dict("index")
        print(f"  Using BYU rank columns FROM THE DATASET for {len(have)}/7 columns "
              f"(scrape not used for regeneration).")

    # DeepSeek canonical maps (built by build_canonical_maps.py). Optional -
    # if absent, build_set_d falls back to normalized exact matching.
    uni_map = load_canonical_map("canonical_universities.json")
    journal_map = load_canonical_map("canonical_journals.json")
    if uni_map is not None:
        print(f"  Using DeepSeek canonical university map ({len(uni_map)} names).")
    else:
        print("  No canonical university map - using exact match "
              "(run build_canonical_maps.py to improve *_top matching).")
    if journal_map is not None:
        print(f"  Using DeepSeek canonical journal map ({len(journal_map)} names).")

    byu_map = parse_byu_map(args.byu_year_for)
    byu_cache = {}   # display_year -> dict
    def byu_for_batch(batch_year):
        # When taking BYU from the dataset, no scrape is needed.
        if byu_dataset is not None:
            return {}
        dy = byu_map.get(batch_year, batch_year - 1)   # default: prior year
        if dy not in byu_cache:
            byu_cache[dy] = load_byu_year(dy, args.byu_rank_window)
            print(f"  BYU year {dy}: {len(byu_cache[dy])} scholars")
        return byu_cache[dy]

    json_by_folder = load_raw_json_by_folder(args.json_dir)   # keyed by folder_name
    rows, missing = [], 0
    for r in index.itertuples(index=False):
        cid, year = int(r.ID), int(r.year)
        rec = json_by_folder.get(Path(r.folder).name)
        if rec is None:
            missing += 1
            row = {c: 0 for c in SET_D_COLS}
        else:
            byu_row = byu_dataset.get(cid) if byu_dataset is not None else None
            row = build_set_d(rec, top_unis, top_journals, byu_for_batch(year),
                              fetch_2degree=args.fetch_2degree,
                              uni_map=uni_map, journal_map=journal_map,
                              byu_row=byu_row)
        rows.append({"ID": cid, **row})

    if missing:
        warnings.warn(f"{missing} candidates had no raw_json - run extract_deepseek.py first.")

    df = pd.DataFrame(rows)[["ID"] + SET_D_COLS]
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols -> {args.out}")


if __name__ == "__main__":
    main()
