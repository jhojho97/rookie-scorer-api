# -*- coding: utf-8 -*-
"""
inference.py
------------
Single-candidate scoring for the web app. Given the raw text of one candidate's
CV and JMP, run the LIVE pipeline end-to-end:

  CV text  --DeepSeek OA1--> Set C (27 structured vars)
                         --> Set D (16 external vars: live BYU scrape + US News
                             + top journals + canonical name maps)
  JMP text --OpenAI embed--> Set E (256 dims, paper's sliding-window method)
           |
           +--> XGBoost C+D+E ensemble --> P(top researcher)
           +--> local TreeSHAP --> top-N contributing factors

This is the WEB-APP path (new candidates): BYU is scraped live, embeddings are
computed live, nothing is copied from the training dataset. Build ONE
CandidateScorer at app startup and call .score() per request.
"""

import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- make the pipeline + project modules importable --------------------------
PIPE = Path(__file__).resolve().parent.parent          # deepseek_pipeline/
ROOT = PIPE.parent                                      # Input/  (has shap_xgb, etc.)
for p in (str(PIPE), str(ROOT)):
    if p not in sys.path:
        sys.path.append(p)

from build_set_C import build_set_c
import build_set_D as D
from build_set_E import embed as embed_jmp_texts
from build_scibert_dataset import extract_jmp_sections
from local_shap_xgb import LocalExplainer
from pricing import UsageMeter


class CandidateScorer:
    """Loads models + reference data once; scores candidates on demand."""

    def __init__(self, data_csv, target="pub_w_top_5pct",
                 sets=("C", "D", "E"), byu_year=None, fetch_2degree=False,
                 provider="deepseek"):
        self.target = target
        self.fetch_2degree = fetch_2degree
        self.provider = provider

        # Run from the project root so the cached models in shap_models/ (a path
        # relative to cwd inside retrain_best_xgb) are found instead of retrained.
        os.chdir(str(ROOT))

        # 1. Extraction client for Set C/D. Provider-swappable so GPT-4 (the
        #    paper's original extractor) can be A/B'd against DeepSeek.
        if provider in ("gpt4", "openai"):
            from extract_gpt4 import get_client as _get_client, extract_one as _extract
            print(f"[scorer] extraction provider = GPT-4 "
                  f"({os.environ.get('GPT4_MODEL', 'gpt-4o')})")
        else:
            from extract_deepseek import get_client as _get_client, extract_one as _extract
            print(f"[scorer] extraction provider = DeepSeek "
                  f"({os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-flash')})")
        self.extract_client = _get_client()
        self._extract_one = _extract

        # 2. Set D reference data (loaded once)
        self.top_unis = D.load_top_universities()
        self.top_journals = D.load_top_journals()
        self.uni_map = D.load_canonical_map("canonical_universities.json")
        self.journal_map = D.load_canonical_map("canonical_journals.json")
        self.byu = self._load_byu(byu_year)

        # 3. Model + SHAP explainer (loads cached C/D/E models from shap_models/)
        print(f"[scorer] building explainer for target={target} ...")
        self.explainer = LocalExplainer(data_csv, target=target, sets=sets)
        # score() is called concurrently by batch workers. Everything before the
        # explain step is network-bound and thread-safe, but shap.TreeExplainer
        # is neither thread-safe nor cheap in memory (it materialises the
        # background matrix), so exactly one candidate may be in it at a time.
        # This is what keeps peak RSS flat as ROOKIE_BATCH_WORKERS rises.
        self._explain_lock = threading.Lock()
        print("[scorer] ready.")

    def _load_byu(self, byu_year):
        """Use the most recent scraped BYU table (live rankings for new candidates)."""
        import glob, re
        if byu_year is not None:
            return D.load_byu_year(byu_year)
        files = sorted(glob.glob(str(PIPE / "reference_data" / "byu_scholar_ranks_*.csv")))
        if not files:
            print("[scorer] WARNING: no BYU table found - coauthor/reference features = 0.")
            return {}
        yr = int(re.search(r"(\d{4})\.csv$", files[-1]).group(1))
        print(f"[scorer] using BYU rankings year {yr} (most recent scraped).")
        return D.load_byu_year(yr)

    # -- core: score one candidate --------------------------------------------
    def score(self, cv_text: str, jmp_text: str, top_n: int = 5,
              byu_row: dict = None) -> dict:
        """byu_row: optional dict of the candidate's dataset BYU columns
        (coauthor_*/reference_*/*_2degree/coauthor_top). When given, those 7
        Set-D columns are copied from the dataset instead of live-scraped -
        used for evaluating KNOWN candidates (the paper's BYU 2019 scrape
        cannot be reproduced live). None => live scrape (new candidates)."""
        cv_text = cv_text or ""
        jmp_text = jmp_text or ""

        # Meter every LLM call in this scoring so the response can report $ cost.
        meter = UsageMeter()

        # 1. OA1 extraction (Set C + names for Set D) via the chosen provider
        rec = self._extract_one(self.extract_client, cv_text, jmp_text, meter=meter)

        # 2. Set C (27 structured CV variables)
        set_c = build_set_c(rec)

        # 3. Set D (live path: BYU scrape + US News + journals + canonical maps)
        set_d = D.build_set_d(
            rec, self.top_unis, self.top_journals, self.byu,
            fetch_2degree=self.fetch_2degree,
            uni_map=self.uni_map, journal_map=self.journal_map,
            byu_row=byu_row,          # None=live scrape; dict=copy from dataset
        )

        # 4. Set E (JMP embedding, paper's method; zero vector if no JMP)
        if jmp_text.strip():
            vec = embed_jmp_texts([jmp_text], meter=meter)[0]
        else:
            vec = np.zeros(256, dtype=float)
        set_e = {f"{i}_dt": float(vec[i]) for i in range(256)}

        # 5. Assemble the model input row and explain
        row = {**set_c, **set_d, **set_e}
        with self._explain_lock:
            result = self.explainer.explain(row, top_n=top_n)

        # When this scoring actually happened. The client cannot derive it: a
        # report is re-rendered, reopened from a table and exported to PDF long
        # after the fact, so a render-time clock would silently relabel it.
        result["scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # attach a little context for the UI
        oa1 = rec.get("oa1", {}) or {}
        result["candidate_name"] = oa1.get("name") or None
        result["extraction"] = {
            "phd": oa1.get("PhD degree"),
            "num_published": oa1.get("number of published papers"),
            "num_awards": oa1.get("number of awards"),
            "research_area": rec.get("primary_area"),
            "cv_chars": len(cv_text),
            "jmp_chars": len(jmp_text),
        }
        # per-scoring token cost (USD) so the frontend can meter user budgets
        result["cost"] = meter.summary()
        return result
