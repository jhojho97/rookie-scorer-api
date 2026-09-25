# -*- coding: utf-8 -*-
"""
Build the web app's per-set models with the Optuna-tuned pipeline.

Run from backend/:
    python training/train_optuna_models.py

Writes shap_models_optuna/xgb_<label>_<set>_<target>.pkl, the same filenames
and tuple format as shap_models/ (the grid-search models), so LocalExplainer
loads either directory unchanged:
    (fitted XGBClassifier, full training matrix as a DataFrame, params dict)

The procedure is main_xg_imp_optuna.py's, step for step, and uses that file's
own compute_ipw_weights and tune_xgboost_optuna rather than a re-implementation
(it is vendored byte-identical beside this script):
  1. Train on 2015-2017, test on 2018.
  2. Stratified 80/20 split of the training years (stratified on pub_top_5pct,
     seed 42, exactly as the research script does for every target).
  3. IPW weights on the 80% fold; Optuna TPE, 50 trials, AUPRC objective,
     RepeatedStratifiedKFold 5x3, on that fold.
  4. Refit the best configuration on the FULL training years with full-data IPW
     (C=0.001 for the 256-dim embedding set, 0.01 otherwise) and
     scale_pos_weight recomputed on the full data.

Optuna is needed only here, never at serving time, so it is deliberately not
in requirements.txt. The versions that matter for the pickles ARE pinned
there (xgboost, scikit-learn, numpy, pandas); train with those exact versions.
"""

import os
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import main_xg_imp_optuna as research          # noqa: E402  (vendored copy)
from shap_xgb import prepare_data              # noqa: E402  (the web app's own loader)

DATA_CSV = ROOT / "2015-2018_rookie_dataset.csv"
OUT_DIR = ROOT / "shap_models_optuna"
SETS = ("C", "D", "E")
TARGETS = ("pub_w_top_5pct", "pub_top_5pct")
N_TRIALS = research.N_OPTUNA_TRIALS if hasattr(research, "N_OPTUNA_TRIALS") else 50


def cache_name(X: pd.DataFrame, set_label: str, target: str) -> str:
    """Same naming rule as shap_xgb.retrain_best_xgb, so the loader finds it."""
    first = X.columns[0].split("_")[-1]
    return f"xgb_{first}_{set_label}_{target}.pkl"


def train_one(X_train: pd.DataFrame, y_train_all: pd.DataFrame, treatment: pd.Series,
              set_label: str, target: str):
    X_train = X_train.reset_index(drop=True)
    y_train_all = y_train_all.reset_index(drop=True)
    treatment = treatment.reset_index(drop=True)

    # Step 2: the research script stratifies on pub_top_5pct for EVERY target.
    X_fold, _, y_fold, _, treat_fold, _ = train_test_split(
        X_train, y_train_all, treatment,
        test_size=0.2, random_state=42,
        stratify=y_train_all["pub_top_5pct"].values,
    )

    # Step 3: tune on the 80% fold.
    ipw_fold = research.compute_ipw_weights(X_fold, treat_fold)
    y_t = y_fold[target]
    spw_fold = float((y_t == 0).sum()) / max(int((y_t == 1).sum()), 1)
    best_params, _ = research.tune_xgboost_optuna(
        X_fold, y_t, sample_weight=ipw_fold, scale_pos_weight=spw_fold,
        n_trials=N_TRIALS, n_splits=5, n_repeats=3, random_state=42,
    )

    # Step 4: refit on the full training years.
    y_full = y_train_all[target]
    c_ipw = 0.001 if set_label in ("B", "E") else 0.01
    ipw_full = research.compute_ipw_weights(X_train, treatment, C_reg=c_ipw)
    best_params["scale_pos_weight"] = float((y_full == 0).sum()) / max(int((y_full == 1).sum()), 1)
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train.values, y_full.values, sample_weight=ipw_full)
    return model, X_train, dict(best_params)


def main():
    data = pd.read_csv(DATA_CSV, index_col=0)
    fm, y_train, y_test, treatment_train, *_ = prepare_data(data, train_test_year=2)
    OUT_DIR.mkdir(exist_ok=True)

    for target in TARGETS:
        for s in SETS:
            X_tr, _ = fm[s]
            print(f"[{s} / {target}] Optuna {N_TRIALS} trials ...", flush=True)
            model, X_bg, params = train_one(X_tr, y_train, treatment_train, s, target)
            path = OUT_DIR / cache_name(X_tr, s, target)
            with open(path, "wb") as f:
                pickle.dump((model, X_bg, params), f)
            print(f"    wrote {path.name}  depth={params['max_depth']} "
                  f"n_est={params['n_estimators']} lr={params['learning_rate']:.4f}", flush=True)


if __name__ == "__main__":
    main()
