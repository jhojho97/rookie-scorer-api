# -*- coding: utf-8 -*-
"""
common.py
---------
Shared pieces for the per-set builders (build_set_C / D / E) and combine_sets.

The critical shared piece is the CANDIDATE INDEX: it assigns each candidate
folder the same ID the training dataset uses, so every per-set CSV is keyed by
the same ID and merges cleanly with the dependent variables.

ID rule (must match how the dataset was built):
  Within each year, dataset rows sorted by ID correspond positionally to the
  data/{year}/ subfolders sorted alphabetically by candidate name.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 console output so Unicode in prints (->, x, [!], box chars) doesn't
# crash on Windows' default cp1252 console. Imported by every pipeline script.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# -- Exact column order expected by the model (mirrors the training dataset) ---
SET_C_COLS = [
    "gender", "has Bachelor honor", "has Master honor", "has PhD honor",
    "number of published papers", "number of R&R papers",
    "number of papers in progress", "has_coauthor", "has_reference",
    "number of coauthors", "number of presentations",
    "number of teaching experiences", "number of awards",
    "number of reviewers", "number of membership",
    "number of working experiences", "had academic work",
    "had non-academic work", "provide abstract",
    "PrimaryResearchArea_financial", "PrimaryResearchArea_auditing",
    "PrimaryResearchArea_managerial", "PrimaryResearchArea_tax",
    "PrimaryResearchMethod_archival", "PrimaryResearchMethod_experiment",
    "PrimaryResearchMethod_analytical", "multi_language",
]
SET_D_COLS = [
    "Bachelor_top", "Master_top", "PhD_top", "visit_top",
    "number of top published papers", "number of top R&R papers",
    "coauthor_mean", "coauthor_high", "number_of_coauthor_2degree",
    "coauthor_top", "number of presentations on top conferences",
    "reference_first", "reference_mean", "reference_high",
    "second_language_asia", "second_language_euro",
]
SET_E_COLS = [f"{i}_dt" for i in range(256)]

DEPENDENT_VARS = ["pub_top_5pct", "pub_w_top_5pct", "research_oriented", "year"]

# -- Language -> region maps (used by Set D) ------------------------------------
ASIAN_LANGUAGES = {
    "chinese", "mandarin", "cantonese", "japanese", "korean", "vietnamese",
    "thai", "hindi", "tamil", "telugu", "bengali", "urdu", "indonesian",
    "malay", "tagalog", "filipino", "burmese", "khmer", "lao", "mongolian",
}
EUROPEAN_LANGUAGES = {
    "french", "german", "spanish", "italian", "portuguese", "dutch",
    "russian", "polish", "greek", "swedish", "norwegian", "danish",
    "finnish", "czech", "hungarian", "romanian", "bulgarian", "ukrainian",
    "serbian", "croatian", "catalan", "turkish",
}


# -- Candidate index: folder -> dataset ID -------------------------------------
def build_candidate_index(data_dir, main_csv, use_ground_truth=True) -> pd.DataFrame:
    """
    Returns a DataFrame [ID, year, folder] mapping each candidate folder to the
    dataset ID.

    PREFERRED: if reference_data-adjacent 'ground_truth_mapping.csv' exists (built
    by build_ground_truth_mapping.py via CV-embedding fingerprinting), use it -
    this is the verified alignment. Used for TRAINING-dataset regeneration.

    FALLBACK: per year, dataset IDs sorted ascending <-> folders sorted
    alphabetically (fragile for mixed-digit folder names; kept only as a default).

    Every per-set builder iterates THIS index, guaranteeing all per-set CSVs
    share the same ID column for a clean merge.
    """
    gt_path = Path(__file__).resolve().parent / "ground_truth_mapping.csv"
    if use_ground_truth and gt_path.exists():
        gt = pd.read_csv(gt_path)
        idx = gt[["ID", "year", "folder"]].sort_values("ID").reset_index(drop=True)
        print(f"  Candidate index: {len(idx)} candidates (GROUND-TRUTH mapping "
              f"from {gt_path.name})")
        return idx

    data_dir = Path(data_dir)
    meta = pd.read_csv(main_csv, usecols=["ID", "year"])

    rows = []
    for year in sorted(meta["year"].unique()):
        ids = sorted(meta.loc[meta["year"] == year, "ID"].tolist())   # ascending
        year_dir = data_dir / str(year)
        if not year_dir.exists():
            continue
        folders = sorted(
            [d for d in year_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name.lower(),
        )
        if len(ids) != len(folders):
            print(f"  [!]  year {year}: {len(ids)} dataset IDs vs {len(folders)} "
                  f"folders - pairing up to min().")
        for i in range(min(len(ids), len(folders))):
            rows.append({"ID": int(ids[i]), "year": int(year),
                         "folder": str(folders[i])})
    idx = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    print(f"  Candidate index: {len(idx)} candidates "
          f"(years {sorted(meta['year'].unique())})")
    return idx


# -- Small shared parsing utils ------------------------------------------------
def to_int(v, default=0) -> int:
    try:
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return default
        return int(float(v))
    except (ValueError, TypeError):
        return default


def as_list(v) -> list:
    if isinstance(v, list):
        return [x for x in v if x not in ("", None)]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def norm_inst(s: str) -> str:
    """Normalize an institution name for matching."""
    s = str(s or "").lower().strip()
    s = re.sub(r"\b(the|university|college|school|of|business|department)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_key(s: str) -> tuple:
    """Order-insensitive person-name key: 'Dan Dhaliwal' == 'Dhaliwal, Dan S.'"""
    s = str(s or "").lower()
    s = re.sub(r"[^a-z, ]", " ", s)
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) >= 2:
            s = " ".join(parts[1:] + parts[:1])
    tokens = [t for t in s.split() if len(t) > 1]
    return tuple(sorted(tokens))


def load_raw_json(json_dir, candidate_id):
    """Load the cached DeepSeek record for an ID, or None."""
    import json
    p = Path(json_dir) / f"{candidate_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_raw_json_by_folder(json_dir):
    """
    Build {folder_name: record} by scanning every cached JSON.

    The JSON files are named by the ID assigned AT EXTRACTION TIME (alphabetical).
    Once the ground-truth mapping reassigns IDs, that filename no longer matches
    the new ID - so downstream builders must look up the extraction by FOLDER,
    not by ID. Each record stores 'folder_name' (written by extract_deepseek).
    """
    import json
    out = {}
    p = Path(json_dir)
    if not p.exists():
        return out
    for f in p.glob("*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        fn = rec.get("folder_name")
        if fn:
            out[fn] = rec
    return out
