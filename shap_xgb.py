"""
shap_analysis_xgboost_ipw.py
============================
Independent SHAP and PDP analysis for the XGBoost No-SMOTE IPW model.

Six analysis types produced per feature set (C, D, E) × target:
  1. Beeswarm summary plot   — normalised SHAP importance (Leinonen et al. 2023)
  2. Waterfall plots         — raw SHAP for each true positive (correct + missed)
  3. Dependence plots        — normalised SHAP vs feature value (top 3)
  4. PDP                     — partial dependence for top 10 features
  5. Interaction analysis    — SHAP interactions for Sets C and D
  6. Cross-set comparison    — normalised SHAP bar chart across C, D, E

Targets evaluated: pub_top_5pct and pub_w_top_5pct only.
Usage
-----
# Case 1: XGBoost model HAS already been run (most common)
python shap_analysis_xgboost_ipw.py \
    --pred_csv output_prediction_main_2018_xgboost_ipw.csv

# Case 2: XGBoost model has NOT been run yet (or force retrain)
python shap_analysis_xgboost_ipw.py --run_model

# Analyse specific sets only
python shap_analysis_xgboost_ipw.py \
    --pred_csv output_prediction_main_2018_xgboost_ipw.csv \
    --sets C D E
"""

import os
import argparse
import pickle
import warnings
import copy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                   # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import xgboost as xgb
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ── Output directory ──────────────────────────────────────────────────────────
SHAP_DIR = 'shap_output'
os.makedirs(SHAP_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    'positive': '#0369A1',    # blue  — pushes prediction higher
    'negative': '#EF4444',    # red   — pushes prediction lower
    'neutral':  '#64748B',    # grey
    'teal':     '#0D9488',
    'navy':     '#0D2137',
}


# ══════════════════════════════════════════════════════════════════════════════
# IPW helper (identical to main_no_smote_xg.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ipw_weights(X_train, treatment, C_reg=0.01):
    treatment = np.asarray(treatment)
    ps_model  = LogisticRegression(
        max_iter=1000, C=C_reg, solver='lbfgs', random_state=42
    )
    ps_model.fit(X_train, treatment)
    p_score = np.clip(ps_model.predict_proba(X_train)[:, 1], 0.05, 0.95)
    p_bar   = treatment.mean()
    weights = np.where(
        treatment == 1,
        p_bar / p_score,
        (1 - p_bar) / (1 - p_score)
    )
    weights = np.clip(weights, np.percentile(weights, 5),
                                np.percentile(weights, 95))
    return weights


def get_rank_decile(s):
    s_pos  = s[s > 0]
    s_pos  = pd.qcut(s_pos.rank(method='first'), q=10, labels=range(1, 11))
    s_zero = s[s == 0].replace(0, 11)
    return pd.concat([s_pos, s_zero]).loc[s.index]


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Data preparation (mirrors get_full_prediction_ensemble exactly)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(data0, train_test_year=2):
    """
    Reproduce the exact data split used in main_no_smote_xg.py so SHAP
    analyses are computed on the same training/test sets as the predictions.
    Returns a dict of feature matrices and label arrays.
    """
    data = data0.copy()

    c = (data.groupby('year')
             .apply(lambda x: get_rank_decile(x.Placerank))
             .reset_index()[['ID', 'Placerank']]
             .set_index('ID'))
    data['Placerank'] = c

    if train_test_year == 2:
        train = data[data.year < 2018]
        test  = data[data.year == 2018]
    elif train_test_year == 1:
        train = data[data.year < 2017]
        test  = data[data.year == 2017]

    feature_matrices = {
        'B': (train.loc[:, '0_cv':'255_cv'],  test.loc[:, '0_cv':'255_cv']),
        'C': (train.loc[:, 'gender':'multi_language'],
              test.loc[:,  'gender':'multi_language']),
        'D': (train.loc[:, 'Bachelor_top':'second_language_euro'],
              test.loc[:,  'Bachelor_top':'second_language_euro']),
        'E': (train.loc[:, '0_dt':'255_dt'],  test.loc[:, '0_dt':'255_dt']),
        'F': (train[['Placerank']],             test[['Placerank']]),
    }

    targets = ['pub_top_5pct', 'pub_w_top_5pct']
    y_train = train[targets]
    y_test  = test[targets]
    treatment_train = train['research_oriented']

    return feature_matrices, y_train, y_test, treatment_train, train, test


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Retrain best XGBoost model for a given feature set and target
#          Mirrors the exact training logic in main_no_smote_xg.py
# ══════════════════════════════════════════════════════════════════════════════

XGB_PARAM_GRID = {
    'max_depth':        [2, 3, 4],
    'learning_rate':    [0.01, 0.05, 0.1],
    'n_estimators':     [50, 100, 150],
    'reg_alpha':        [0.5, 1.0],
    'reg_lambda':       [1.0, 5.0],
    'gamma':            [0.1, 0.5],
    'min_child_weight': [5, 10],
    'tree_method':      ['hist'],
    'subsample':        [0.9],
    'colsample_bytree': [0.5],
}


def retrain_best_xgb(X_train_full, y_train_full, treatment_full,
                     target, cache_dir='shap_models'):
    """
    Reproduce the exact model trained in main_no_smote_xg.py:
      1. Stratified 80/20 split
      2. Hyperparameter search on validation AUC
      3. Retrain best config on full training data with IPW weights

    Uses model caching — if the pkl exists, load it instead of retraining.
    This respects the 'already ran the model' scenario: the predictions CSV
    was saved by the main script, but the trained model object was not.
    We retrain here with identical settings to get the same model.

    Parameters
    ----------
    X_train_full : pd.DataFrame, full training feature matrix for this set
    y_train_full : pd.Series,    binary labels (pub_top_5pct or pub_w_top_5pct)
    treatment_full : pd.Series,  research_oriented binary indicator
    target       : str
    cache_dir    : str, directory to save/load trained model pkl files

    Returns
    -------
    model : trained XGBClassifier (best config, fitted on full training data)
    X_train_full_r : np.ndarray, training features (reset index, for SHAP)
    """
    os.makedirs(cache_dir, exist_ok=True)
    set_label = getattr(X_train_full, 'columns', ['unk'])[0].split('_')[-1]
    cache_path = os.path.join(
        cache_dir, f"xgb_{set_label}_{target}.pkl"
    )

    if os.path.exists(cache_path):
        print(f"    Loading cached model: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"    Training XGBoost on full data for {target}...")

    # ── Step 1: Stratified 80/20 split (mirrors main script exactly) ─────────
    X_r        = X_train_full.reset_index(drop=True)
    y_r        = y_train_full.reset_index(drop=True)
    treat_r    = treatment_full.reset_index(drop=True)

    X_tr, X_val, y_tr, y_val, treat_tr, _ = train_test_split(
        X_r, y_r, treat_r,
        test_size=0.2,
        random_state=42,
        stratify=y_r.values
    )

    # ── Step 2: IPW on training fold ─────────────────────────────────────────
    n_features = X_tr.shape[1]
    C_ipw      = 0.001 if n_features > 50 else 0.01
    ipw_tr     = compute_ipw_weights(X_tr.values, treat_tr.values, C_reg=C_ipw)

    # ── Step 3: Hyperparameter grid search by validation AUC ─────────────────
    from sklearn.metrics import roc_auc_score

    n_neg   = (y_tr == 0).sum()
    n_pos   = (y_tr == 1).sum()
    spw     = n_neg / n_pos if n_pos > 0 else 1.0

    best_auc    = -1.0
    best_params = None

    keys   = list(XGB_PARAM_GRID.keys())
    combos = list(product(*XGB_PARAM_GRID.values()))
    print(f"    Grid search: {len(combos)} configurations...")

    for combo in combos:
        params = dict(zip(keys, combo))
        params['scale_pos_weight'] = spw

        try:
            model_cv = xgb.XGBClassifier(**params, random_state=42,
                                         verbosity=0, use_label_encoder=False,
                                         eval_metric='auc')
            model_cv.fit(X_tr.values, y_tr.values,
                         sample_weight=ipw_tr, verbose=False)
            y_score_val = model_cv.predict_proba(X_val.values)[:, 1]
            if len(np.unique(y_val.values)) > 1:
                auc = roc_auc_score(y_val.values, y_score_val)
                if auc > best_auc:
                    best_auc    = auc
                    best_params = params
        except Exception:
            continue

    if best_params is None:
        # Fallback to sensible small-sample defaults
        best_params = {
            'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 100,
            'reg_alpha': 1.0, 'reg_lambda': 5.0, 'gamma': 0.1,
            'min_child_weight': 10, 'tree_method': 'hist',
            'subsample': 0.9, 'colsample_bytree': 0.5,
            'scale_pos_weight': spw
        }

    print(f"    Best val AUC = {best_auc:.4f}  params = {best_params}")

    # ── Step 4: Retrain on FULL training data ─────────────────────────────────
    X_full_r    = X_train_full.reset_index(drop=True)
    treat_full_r = treatment_full.reset_index(drop=True)
    ipw_full    = compute_ipw_weights(X_full_r.values, treat_full_r.values,
                                       C_reg=C_ipw)

    n_neg_f = (y_train_full == 0).sum()
    n_pos_f = (y_train_full == 1).sum()
    best_params['scale_pos_weight'] = n_neg_f / n_pos_f if n_pos_f > 0 else 1.0

    final_model = xgb.XGBClassifier(
        **best_params, random_state=42,
        verbosity=0, use_label_encoder=False, eval_metric='auc'
    )
    final_model.fit(X_full_r.values, y_train_full.reset_index(drop=True).values,
                    sample_weight=ipw_full, verbose=False)

    # Cache
    payload = (final_model, X_full_r, best_params)
    with open(cache_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"    Cached: {cache_path}")

    return final_model, X_full_r, best_params


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — SHAP computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap(model, X_train, X_test, feature_names):
    """
    Compute SHAP values using TreeExplainer (exact, not approximate).
    Returns shap_values for test set and expected_value from explainer.

    TreeExplainer is the correct choice for XGBoost:
      - Exact Shapley values (not sampling approximation)
      - O(TLD) complexity per sample (T=trees, L=leaves, D=depth)
      - Consistent with how the model actually makes decisions
    """
    print("    Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(
        model,
        data=X_train.values,           # background dataset for expected value
        feature_perturbation='interventional',  # marginal not conditional
        model_output='probability'      # output in probability space not log-odds
    )
    shap_values = explainer(X_test.values if hasattr(X_test, 'values')
                             else X_test)

    # Convert to Explanation object if raw array returned
    if isinstance(shap_values, np.ndarray):
        shap_values = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_test.values if hasattr(X_test, 'values') else X_test,
            feature_names=list(feature_names)
        )
    else:
        shap_values.feature_names = list(feature_names)

    return shap_values, explainer


# ══════════════════════════════════════════════════════════════════════════════
# Step 3b — SHAP normalisation (Leinonen et al. 2023 methodology)
# ══════════════════════════════════════════════════════════════════════════════

def normalise_shap(shap_values):
    """
    Normalise SHAP values within each base model so that the sum of absolute
    values equals 1, following Leinonen et al. (2023).

    This is required for comparing feature importance ACROSS the three
    predictor sets (C, D, E) in the late-fusion ensemble. Without it:
      - Set E (256 embedding dims) would have many small SHAP values
        whose total magnitude dwarfs Sets C and D
      - A feature with mean|SHAP|=0.001 in Set C cannot fairly be
        compared to mean|SHAP|=0.001 in Set E because the scales differ

    Procedure (per test year, per base model):
      1. Compute raw SHAP values φ_i for each feature i and candidate j
      2. For each candidate j: normalise by Σ_i |φ_ij|
         → φ̃_ij = φ_ij / Σ_i |φ_ij|
      3. After normalisation: Σ_i |φ̃_ij| = 1 for every candidate

    This ensures that the sum of absolute SHAP values for each base model
    equals 1, making relative feature contributions directly comparable
    across models trained on different predictor sets.

    Parameters
    ----------
    shap_values : shap.Explanation object
                  Raw SHAP values from TreeExplainer.
                  shap_values.values has shape (n_candidates, n_features)

    Returns
    -------
    shap_norm : shap.Explanation object
                Normalised SHAP values with the same structure.
                shap_norm.values[j, :] sums to 1 in absolute value
                for every candidate j.

    Notes
    -----
    - The base_values (expected value) is preserved unchanged — it is a
      scalar property of the model, not a per-candidate SHAP value
    - The data array is also preserved — normalisation only affects values
    - A small epsilon (1e-10) guards against division by zero for candidates
      where all SHAP values are exactly zero (very rare edge case)
    """
    raw_values = shap_values.values.copy()   # shape: (n_candidates, n_features)

    # Sum of absolute values per candidate (row-wise)
    # Shape: (n_candidates,)
    abs_sum_per_candidate = np.abs(raw_values).sum(axis=1, keepdims=True)

    # Avoid division by zero — epsilon is negligible relative to any real SHAP
    abs_sum_per_candidate = np.where(
        abs_sum_per_candidate == 0, 1e-10, abs_sum_per_candidate
    )

    # Normalise: divide each candidate's SHAP vector by its total |SHAP| sum
    normalised_values = raw_values / abs_sum_per_candidate

    # Verification: each row should now sum to 1 in absolute value
    row_check = np.abs(normalised_values).sum(axis=1)
    assert np.allclose(row_check, 1.0, atol=1e-6), \
        "Normalisation failed: row absolute sums are not all 1.0"

    # Return a new Explanation object with normalised values
    shap_norm = shap.Explanation(
        values       = normalised_values,
        base_values  = shap_values.base_values,   # unchanged
        data         = shap_values.data,             # unchanged
        feature_names= shap_values.feature_names,
    )
    return shap_norm


def mean_abs_normalised_shap(shap_norm):
    """
    Compute mean absolute normalised SHAP per feature across all test candidates.

    This is the key statistic for feature importance ranking and cross-set
    comparison. For feature i:
      importance_i = (1/N) × Σ_j |φ̃_ij|
    where φ̃_ij is the normalised SHAP value of feature i for candidate j.

    Parameters
    ----------
    shap_norm : shap.Explanation, normalised SHAP values

    Returns
    -------
    mean_abs : np.ndarray, shape (n_features,)
               Mean absolute normalised SHAP per feature.
               Values are in [0, 1] and sum to approximately 1 across features
               (exactly 1 when averaged across all candidates with equal weight).
    """
    return np.abs(shap_norm.values).mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Plot functions
# ══════════════════════════════════════════════════════════════════════════════
def plot_beeswarm(shap_norm, feature_names, set_label, target, max_display=20):
    """
    Beeswarm summary plot using NORMALISED SHAP values.
    X-axis = normalised SHAP value (fraction of total |SHAP| for that candidate).
    Colour = raw feature value (high=blue, low=red).

    Normalisation (Leinonen et al. 2023): each candidate's SHAP vector is
    divided by the sum of its absolute values so that Σ|φ̃_i|=1 per candidate.
    This makes the x-axis directly comparable across Sets C, D, and E.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    shap.plots.beeswarm(
        shap_norm,
        max_display=max_display,
        show=False,
        color_bar=True,
        plot_size=None,
        ax=ax if hasattr(shap.plots.beeswarm, '__code__') else None,
    )

    ax = plt.gca()
    ax.set_title(
        f'XGBoost SHAP — Normalised Feature Importance\n'
        f'Set {set_label} · {target} · Test 2018',
        fontsize=13, fontweight='bold', color=COLORS['navy'], pad=12
    )
    # x-axis label reflects normalisation
    ax.set_xlabel(
        'Normalised SHAP value  (fraction of total |SHAP| per candidate)',
        fontsize=10
    )

    plt.tight_layout()
    path = os.path.join(SHAP_DIR, f'beeswarm_{set_label}_{target}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_waterfall_true_positives(shap_values_raw, shap_norm, y_test, y_score,
                                   set_label, target, max_features=15):
    """
    Waterfall plot for each true positive correctly identified by the model.

    Uses RAW (un-normalised) SHAP values for the waterfall:
      - The waterfall must sum to the predicted probability for that candidate
      - Normalised values no longer sum to the probability — they sum to 1
      - Using normalised values here would show incorrect base/final values

    The normalised values are used for ranking which features to display
    (top features by normalised importance), ensuring the most meaningful
    features appear even if raw magnitudes differ.

    For each correctly identified candidate:
      - Positive bars: features that pushed probability UP
      - Negative bars: features that pushed probability DOWN
      - Base: average model prediction (E[f(X)])
      - Final: this candidate's predicted probability
    """
    y_true_arr  = np.asarray(y_test, dtype=float)
    y_score_arr = np.asarray(y_score, dtype=float)

    k           = int(y_true_arr.sum())
    top_k_idx   = np.argsort(y_score_arr)[::-1][:k]
    correct_tp  = [i for i in top_k_idx if y_true_arr[i] == 1]
    missed_tp   = [i for i in np.where(y_true_arr == 1)[0]
                   if i not in top_k_idx]

    print(f"    True positives: {int(y_true_arr.sum())} total | "
          f"{len(correct_tp)} correctly found | {len(missed_tp)} missed")

    paths = []
    for rank, idx in enumerate(correct_tp):
        fig, ax = plt.subplots(figsize=(10, 6))
        # Raw SHAP for waterfall — preserves correct probability scale
        shap.plots.waterfall(shap_values_raw[idx], max_display=max_features,
                              show=False)
        ax = plt.gca()
        ax.set_title(
            f'XGBoost SHAP — True Positive #{rank+1} (correctly identified)\n'
            f'Set {set_label} · {target} · '
            f'Predicted prob = {y_score_arr[idx]:.3f}\n'
            f'(Raw SHAP values — bars sum to predicted probability)',
            fontsize=10, fontweight='bold', color=COLORS['navy']
        )
        plt.tight_layout()
        path = os.path.join(
            SHAP_DIR, f'waterfall_{set_label}_{target}_tp{rank+1}.png'
        )
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {path}")
        paths.append(path)

    for rank, idx in enumerate(missed_tp):
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values_raw[idx], max_display=max_features,
                              show=False)
        ax = plt.gca()
        ax.set_title(
            f'XGBoost SHAP — Missed True Positive #{rank+1}\n'
            f'Set {set_label} · {target} · '
            f'Predicted prob = {y_score_arr[idx]:.3f} (below threshold)\n'
            f'(Raw SHAP values — bars sum to predicted probability)',
            fontsize=10, fontweight='bold', color=COLORS['negative']
        )
        plt.tight_layout()
        path = os.path.join(
            SHAP_DIR, f'waterfall_{set_label}_{target}_missed{rank+1}.png'
        )
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {path}")
        paths.append(path)

    return paths, correct_tp, missed_tp


def plot_dependence(shap_norm, X_test, feature_names, set_label, target,
                    top_n=3):
    """
    Dependence plot for top N features by mean absolute NORMALISED SHAP.
    X-axis = feature value. Y-axis = normalised SHAP value.
    Colour = strongest interacting feature.

    Using normalised SHAP ensures feature rankings and y-axis scales are
    comparable across Sets C, D, and E. The non-linearity pattern (shape
    of the curve) is identical to raw SHAP — normalisation is a per-row
    scaling that does not change the relative ordering or shape.
    """
    mean_abs   = mean_abs_normalised_shap(shap_norm)
    top_idx    = np.argsort(mean_abs)[::-1][:top_n]

    paths = []
    for rank, feat_idx in enumerate(top_idx):
        feat_name = feature_names[feat_idx]
        fig, ax   = plt.subplots(figsize=(8, 5))

        shap.plots.scatter(
            shap_norm[:, feat_idx],
            color=shap_norm,
            show=False,
            ax=ax,
        )

        ax.set_title(
            f'XGBoost SHAP Dependence — {feat_name}\n'
            f'Set {set_label} · {target}\n'
            f'Colour = strongest interacting feature',
            fontsize=11, fontweight='bold', color=COLORS['navy']
        )
        ax.set_xlabel(feat_name, fontsize=10)
        ax.set_ylabel(
            f'Normalised SHAP  (fraction of total |SHAP| per candidate)',
            fontsize=9
        )
        ax.axhline(0, color=COLORS['neutral'], linewidth=0.8, linestyle='--')

        plt.tight_layout()
        safe_name = feat_name.replace(' ', '_').replace('/', '_')[:30]
        path = os.path.join(
            SHAP_DIR, f'dependence_{set_label}_{target}_{safe_name}.png'
        )
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {path}")
        paths.append(path)

    return paths


def plot_comparison_bar(shap_results_by_set, target):
    """
    Horizontal bar chart comparing mean absolute NORMALISED SHAP across
    feature sets C, D, and E.

    Uses normalised SHAP (Leinonen et al. 2023) so that values are directly
    comparable across sets. Without normalisation, Set E (256 dims) would
    dominate because individual features have smaller raw SHAP values that
    collectively total a much larger sum than Sets C or D.

    shap_results_by_set: {set_label: (sv_raw, shap_norm, feat_names)}
    """
    all_features = {}
    for set_label, (sv_raw, shap_norm, feat_names) in shap_results_by_set.items():
        mean_abs  = mean_abs_normalised_shap(shap_norm)
        top10_idx = np.argsort(mean_abs)[::-1][:10]
        for i in top10_idx:
            name  = feat_names[i]
            score = mean_abs[i]
            if name not in all_features:
                all_features[name] = {}
            all_features[name][set_label] = score

    df = pd.DataFrame(all_features).T.fillna(0)
    df['max'] = df.max(axis=1)
    df = df.sort_values('max', ascending=True).tail(20)
    df = df.drop(columns='max')

    n_sets = len(df.columns)
    colors = ['#0369A1', '#0D9488', '#F59E0B', '#6366F1', '#EF4444']
    bar_h  = 0.8 / max(n_sets, 1)

    fig, ax = plt.subplots(figsize=(11, max(6, len(df) * 0.4 + 1)))
    y_pos   = np.arange(len(df))

    for i, (col, color) in enumerate(zip(df.columns, colors)):
        offset = (i - n_sets / 2 + 0.5) * bar_h
        ax.barh(y_pos + offset, df[col], height=bar_h,
                label=f'Set {col}', color=color, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index, fontsize=9)

    ax.set_xlabel('Mean |Normalised SHAP value|', fontsize=10)
    ax.set_title(
        f'XGBoost SHAP — Normalised Feature Importance Across Sets\n'
        f'{target} · Test 2018\n'
        f'Normalised per Leinonen et al. (2023): Σ|φ̃_i|=1 per candidate',
        fontsize=11, fontweight='bold', color=COLORS['navy']
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(SHAP_DIR, f'comparison_bar_{target}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved cross-set comparison: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Diagnostic summary 
# ══════════════════════════════════════════════════════════════════════════════

def print_top_features_summary(shap_results_by_set, target, top_n=10):
    """
    Print a structured summary of XGBoost normalised SHAP top features.

    Uses normalised SHAP values (Leinonen et al. 2023) so that rankings
    within each set are based on comparable scales.
    """
    print(f"\n{'='*65}")
    print(f"SHAP SUMMARY — Top Features ({target})")
    print(f"Normalised SHAP: Σ|φ̃_i|=1 per candidate (Leinonen et al. 2023)")
    print(f"{'='*65}")

    for set_label, (sv_raw, shap_norm, feat_names) in shap_results_by_set.items():
        mean_abs   = mean_abs_normalised_shap(shap_norm)
        top_idx    = np.argsort(mean_abs)[::-1][:top_n]
        xgb_top    = [feat_names[i] for i in top_idx]

        print(f"\nSet {set_label} — XGBoost top {top_n} features (normalised):")
        for rank, (i, name) in enumerate(zip(top_idx, xgb_top)):
            print(f"  {rank+1:2d}. {name:<40} mean|norm-SHAP|={mean_abs[i]:.5f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Interaction analysis: validates the non-linearity hypothesis
# ══════════════════════════════════════════════════════════════════════════════

def analyse_interactions(model, X_train, X_test, feature_names,
                          set_label, target, top_pairs=3):
    """
    Compute SHAP interaction values to directly test whether XGBoost
    is exploiting feature combinations that logistic regression misses.

    Interaction values decompose each prediction into:
      - Main effect of feature i on candidate j
      - Interaction effect of feature pair (i,k) on candidate j

    If interaction values are large relative to main effects, XGBoost's
    non-linearity advantage over logistic regression is validated.

    Note: shap_interaction_values() is memory-intensive for 256-dim inputs.
    Only run for Sets C and D (low-dimensional). Skipped for B and E.
    """
    n_features = X_train.shape[1] if hasattr(X_train, 'shape') else len(feature_names)
    if n_features > 50:
        print(f"    Skipping interaction analysis for Set {set_label} "
              f"({n_features} features — too memory intensive).")
        print(f"    Run dependence plots instead to see interaction colours.")
        return None

    print(f"    Computing SHAP interaction values for Set {set_label}...")
    X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_arr  = X_test.values  if hasattr(X_test, 'values')  else X_test

    explainer    = shap.TreeExplainer(model, data=X_train_arr)
    interaction_vals = explainer.shap_interaction_values(X_test_arr)

    # Mean absolute interaction for each pair (i, j) where i != j
    mean_interaction = np.abs(interaction_vals).mean(axis=0)
    np.fill_diagonal(mean_interaction, 0)   # zero out main effects diagonal

    # Find top interacting pairs
    n = len(feature_names)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((mean_interaction[i, j], i, j))
    pairs.sort(reverse=True)

    print(f"\n    Top {top_pairs} feature interactions (Set {set_label}, {target}):")
    print(f"    {'Pair':<50} {'Mean |interaction|':>20}")
    print(f"    {'─'*70}")

    results = []
    
    # --- FIX: Recover full 2D SHAP values from the 3D interaction matrix ---
    # Summing interactions across axis=2 mathematically yields standard SHAP values
    # Shape becomes (n_samples, n_features), which perfectly matches X_test_arr
    shap_values_2d = interaction_vals.sum(axis=2)
    
    for val, i, j in pairs[:top_pairs]:
        fname_i = feature_names[i]
        fname_j = feature_names[j]
        print(f"    {fname_i} × {fname_j:<30} {val:.6f}")
        results.append({'feature_1': fname_i, 'feature_2': fname_j,
                         'interaction': val})

        # Plot the interaction as a dependence plot coloured by interaction
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pass the full 2D shap_values_2d array. The function will automatically 
        # extract column 'i' for the y-axis and color by column 'j'
        shap.dependence_plot(i, shap_values_2d, X_test_arr, 
                             feature_names=list(feature_names),
                             interaction_index=j, show=False, ax=axes[0])
        axes[0].set_title(f'{fname_i}\ncoloured by {fname_j}', fontsize=10)

        shap.dependence_plot(j, shap_values_2d, X_test_arr, 
                             feature_names=list(feature_names),
                             interaction_index=i, show=False, ax=axes[1])
        axes[1].set_title(f'{fname_j}\ncoloured by {fname_i}', fontsize=10)

        fig.suptitle(
            f'XGBoost Interaction: {fname_i} × {fname_j}\n'
            f'Set {set_label} · {target}',
            fontsize=12, fontweight='bold', color=COLORS['navy']
        )
        plt.tight_layout()
        safe = f"{fname_i[:15]}_{fname_j[:15]}".replace(' ', '_')
        path = os.path.join(
            SHAP_DIR, f'interaction_{set_label}_{target}_{safe}.png'
        )
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {path}")

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Partial Dependence Plots (PDP) for top 10 features
# ══════════════════════════════════════════════════════════════════════════════

def plot_pdp(model, X_train, feature_names, shap_norm,
             set_label, target, top_n=10):
    """
    Partial Dependence Plots (PDP) for the top N features by mean absolute
    normalised SHAP importance.

    What PDP shows vs what SHAP shows
    -----------------------------------
    SHAP (dependence plot):  How much does feature i contribute to the
                             prediction for each individual candidate?
                             Shows the actual spread across all test candidates.

    PDP:                     What is the average predicted probability as
                             feature i varies from its minimum to maximum,
                             holding all other features at their observed values?
                             Shows the marginal effect of the feature on
                             the model's output, averaged over the training data.

    Why both are needed
    --------------------
    - SHAP dependence tells you HOW the model used feature i for actual candidates
    - PDP tells you WHAT the model would predict if only feature i changed
    - SHAP can show heterogeneous effects (different SHAP values for same feature
      value) that PDP smooths over — comparing both validates whether the XGBoost
      non-linearity is consistent across candidates or driven by specific outliers

    Computation method: manual grid sweep over percentile values of each feature
    using the trained model's predict_proba. This is model-agnostic and matches
    exactly how sklearn's PartialDependenceDisplay works internally, but allows
    full control over the grid, confidence intervals, and styling.

    Parameters
    ----------
    model        : trained XGBClassifier
    X_train      : pd.DataFrame or np.ndarray, training feature matrix
                   (used for the background distribution)
    feature_names: list of str
    shap_norm    : normalised shap.Explanation, used to rank features
    set_label    : str, e.g. 'C', 'D', 'E'
    target       : str, e.g. 'pub_w_top_5pct'
    top_n        : int, number of features to plot (default 10)

    Outputs
    -------
    One combined figure with top_n subplots saved as:
      pdp_{set_label}_{target}_top{top_n}.png

    Plus one individual figure per feature for higher resolution:
      pdp_{set_label}_{target}_{feature_name}.png
    """
    X_arr = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)

    # ── Select top N features by mean |normalised SHAP| ───────────────────────
    mean_abs  = mean_abs_normalised_shap(shap_norm)
    top_idx   = np.argsort(mean_abs)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]

    print(f"    PDP top {top_n} features by normalised SHAP importance:")
    for rank, (i, name) in enumerate(zip(top_idx, top_names)):
        print(f"      {rank+1:2d}. {name:<40} mean|norm-SHAP|={mean_abs[i]:.5f}")

    # ── Build PDP grid ─────────────────────────────────────────────────────────
    # For each feature: 50 evenly-spaced percentile grid points
    # Using percentiles (not linear spacing) to handle skewed distributions,
    # which is important for features like coauthor_mean (range 0-5614)
    N_GRID = 50

    pdp_results = {}  # {feature_name: (grid_values, mean_proba, ci_lower, ci_upper)}

    for feat_idx, feat_name in zip(top_idx, top_names):
        col_values = X_arr[:, feat_idx]

        # Grid: 50 values from 5th to 95th percentile
        # (avoids extrapolation into extreme tails with no training data)
        grid = np.linspace(
            np.percentile(col_values, 5),
            np.percentile(col_values, 95),
            N_GRID
        )

        mean_proba = np.zeros(N_GRID)
        std_proba  = np.zeros(N_GRID)

        for g_idx, grid_val in enumerate(grid):
            # Copy entire training matrix and set feature i to grid value
            X_modified = X_arr.copy()
            X_modified[:, feat_idx] = grid_val

            # Predict probability for all training candidates with this fixed value
            probas = model.predict_proba(X_modified)[:, 1]

            mean_proba[g_idx] = probas.mean()
            std_proba[g_idx]  = probas.std()

        # 95% confidence interval: ±1.96 × std / sqrt(n)
        n                    = X_arr.shape[0]
        ci                   = 1.96 * std_proba / np.sqrt(n)
        pdp_results[feat_name] = (grid, mean_proba, mean_proba - ci,
                                   mean_proba + ci, col_values)

    # ── Combined figure: top_n subplots in a grid ─────────────────────────────
    n_cols = 2
    n_rows = int(np.ceil(top_n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(12, 3.5 * n_rows),
                              constrained_layout=True)
    axes_flat = axes.flatten() if top_n > 1 else [axes]

    for rank, (feat_name, ax) in enumerate(zip(top_names, axes_flat)):
        grid, mean_p, ci_lo, ci_hi, raw_vals = pdp_results[feat_name]

        # PDP line
        ax.plot(grid, mean_p, color=COLORS['positive'],
                linewidth=2.0, label='Mean predicted P(top researcher)')

        # Confidence interval band
        ax.fill_between(grid, ci_lo, ci_hi,
                         alpha=0.20, color=COLORS['positive'],
                         label='95% CI')

        # Rug plot: actual training data distribution (top edge, small ticks)
        rug_vals = raw_vals[
            (raw_vals >= np.percentile(raw_vals, 5)) &
            (raw_vals <= np.percentile(raw_vals, 95))
        ]
        y_rug = ax.get_ylim()[1] if rank == 0 else mean_p.max() * 1.02
        ax.plot(rug_vals, np.full_like(rug_vals, mean_p.max()),
                '|', color=COLORS['neutral'], alpha=0.4,
                markersize=4, label='Training data distribution')

        # Formatting
        ax.set_xlabel(feat_name, fontsize=9)
        ax.set_ylabel('Mean P(top researcher)', fontsize=8)
        ax.set_title(
            f'{rank+1}. {feat_name}\nmean|norm-SHAP|={mean_abs[top_idx[rank]]:.4f}',
            fontsize=9, fontweight='bold', color=COLORS['navy']
        )
        ax.axhline(0, color=COLORS['neutral'], linewidth=0.5, linestyle=':')
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)

        # Only show legend on first subplot
        if rank == 0:
            ax.legend(fontsize=7, loc='upper left')

    # Hide unused subplots
    for ax in axes_flat[len(top_names):]:
        ax.set_visible(False)

    fig.suptitle(
        f'Partial Dependence Plots — XGBoost IPW\n'
        f'Set {set_label} · {target} · Top {top_n} features by normalised SHAP\n'
        f'Grid: 50 points from 5th–95th percentile of each feature',
        fontsize=11, fontweight='bold', color=COLORS['navy'], y=1.01
    )

    path_combined = os.path.join(
        SHAP_DIR, f'pdp_{set_label}_{target}_top{top_n}.png'
    )
    plt.savefig(path_combined, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved combined PDP: {path_combined}")

    # ── Individual high-resolution plots ──────────────────────────────────────
    individual_paths = []
    for rank, feat_name in enumerate(top_names):
        grid, mean_p, ci_lo, ci_hi, raw_vals = pdp_results[feat_name]

        fig, ax = plt.subplots(figsize=(8, 4.5))

        ax.plot(grid, mean_p, color=COLORS['positive'],
                linewidth=2.5, label='Mean predicted P(top researcher)')
        ax.fill_between(grid, ci_lo, ci_hi,
                         alpha=0.20, color=COLORS['positive'],
                         label='95% CI (training set)')

        # Rug: full distribution
        rug_vals = raw_vals[
            (raw_vals >= np.percentile(raw_vals, 5)) &
            (raw_vals <= np.percentile(raw_vals, 95))
        ]
        ax.plot(rug_vals, np.full_like(rug_vals, mean_p.min() - 0.002),
                '|', color=COLORS['neutral'], alpha=0.5,
                markersize=5, label='Observed values')

        # Mark the feature's mean value
        ax.axvline(raw_vals.mean(), color=COLORS['teal'],
                    linewidth=1.2, linestyle='--',
                    label=f'Training mean = {raw_vals.mean():.2f}')

        ax.set_xlabel(feat_name, fontsize=11)
        ax.set_ylabel('Mean P(top researcher | feature set to x)', fontsize=10)
        ax.set_title(
            f'PDP — {feat_name}\n'
            f'Set {set_label} · {target} · '
            f'Rank {rank+1} by normalised SHAP importance',
            fontsize=11, fontweight='bold', color=COLORS['navy']
        )
        ax.legend(fontsize=9, loc='best')
        ax.axhline(0, color=COLORS['neutral'], linewidth=0.5, linestyle=':')
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        safe_name = feat_name.replace(' ', '_').replace('/', '_')[:35]
        path_ind  = os.path.join(
            SHAP_DIR, f'pdp_{set_label}_{target}_{safe_name}.png'
        )
        plt.savefig(path_ind, dpi=150, bbox_inches='tight')
        plt.close()
        individual_paths.append(path_ind)

    print(f"    Saved {len(individual_paths)} individual PDP plots")
    return path_combined, individual_paths


# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — SHAP bar chart: mean |normalised SHAP| for top 10 features
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_bar(shap_norm, feature_names, set_label, target, top_n=10):
    """
    Horizontal bar chart of mean absolute normalised SHAP for top N features.

    This is the canonical SHAP importance plot — allowing rank-order 
    comparison of which features XGBoost relied on most.

    Two layers of information per bar:
      ─ Total bar length   = mean |normalised SHAP| across all test candidates
      ─ Error bar (±1 std) = variability in |SHAP| across candidates
                             Wide error bars → feature is important for some
                             candidates but not others (heterogeneous effect)
                             Narrow error bars → feature consistently matters

    Parameters
    ----------
    shap_norm     : shap.Explanation, normalised SHAP values
    feature_names : list of str
    set_label     : str, e.g. 'C', 'D', 'E'
    target        : str, e.g. 'pub_w_top_5pct'
    top_n         : int, number of features to show (default 10)

    Outputs
    -------
    shap_bar_{set_label}_{target}_top{top_n}.png
    """
    mean_abs = mean_abs_normalised_shap(shap_norm)       # shape (n_features,)
    std_abs  = np.abs(shap_norm.values).std(axis=0)      # std across candidates

    # Top N features sorted ascending (so top feature is at top of chart)
    top_idx   = np.argsort(mean_abs)[::-1][:top_n]
    top_idx   = top_idx[::-1]                            # reverse for horizontal bar
    top_names = [feature_names[i] for i in top_idx]
    top_means = mean_abs[top_idx]
    top_stds  = std_abs[top_idx]

    bar_colors = [COLORS['positive'] for _ in top_names]

    fig, ax = plt.subplots(figsize=(9, 0.55 * top_n + 1.8))

    y_pos = np.arange(len(top_names))
    bars  = ax.barh(y_pos, top_means, xerr=top_stds,
                    color=bar_colors, alpha=0.85,
                    error_kw={'ecolor': COLORS['neutral'],
                               'capsize': 4, 'linewidth': 1.2},
                    height=0.62)

    # ── Annotate bar values ────────────────────────────────────────────────────
    for bar, mean_val in zip(bars, top_means):
        ax.text(
            mean_val + top_stds[list(top_means).index(mean_val)] * 0.05,
            bar.get_y() + bar.get_height() / 2,
            f'{mean_val:.4f}',
            va='center', ha='left', fontsize=8.5, color=COLORS['navy']
        )

    # ── Axes formatting ────────────────────────────────────────────────────────
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10)

    for tick, color in zip(ax.get_yticklabels(), bar_colors):
        tick.set_color(color)

    ax.set_xlabel('Mean |Normalised SHAP value|  (± 1 std)', fontsize=10)
    ax.set_title(
        f'XGBoost SHAP Feature Importance — Top {top_n} Features\n'
        f'Set {set_label} · {target} · Test 2018\n'
        f'Normalised per Leinonen et al. (2023): Σ|φ̃_i|=1 per candidate',
        fontsize=11, fontweight='bold', color=COLORS['navy'], pad=10
    )

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=COLORS['positive'],
                        label='XGBoost feature'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8.5)

    # Clean spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(SHAP_DIR, f'shap_bar_{set_label}_{target}_top{top_n}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved SHAP bar chart: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — SHAP interaction heatmap: pairwise interaction strengths
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_interaction_heatmap(model, X_train, X_test, feature_names,
                                   shap_norm, set_label, target, top_n=10):
    """
    Heatmap of mean absolute SHAP interaction values for top N features.

    What this shows
    ---------------
    SHAP interaction values (Lundberg et al. 2020) decompose each SHAP value
    into main effects and pairwise interaction effects:
      φ_i(x) = main_effect_i + Σ_{j≠i} interaction_ij

    The heatmap shows mean|interaction_ij| averaged across all test candidates,
    giving the average strength of every feature pair's interaction.

    Why this matters for your analysis
    ------------------------------------
    - Diagonal entries = main effects (how much feature i matters on its own)
    - Off-diagonal entries = interaction effects (how much features i and j
      jointly shift predictions beyond their individual contributions)
    - If off-diagonal values are large relative to diagonal:
      XGBoost is exploiting non-linear feature interactions
      → validates the non-linearity hypothesis vs logistic regression

    This is only computed for low-dimensional sets (C, D) where n_features ≤ 50.
    For Set E (256 dims), the interaction matrix is 256×256 = 65,536 entries
    which is memory-intensive. The top-N subset heatmap is computed instead
    by restricting to the top N features before calling shap_interaction_values.

    Parameters
    ----------
    model        : trained XGBClassifier
    X_train      : pd.DataFrame, full training feature matrix (background)
    X_test       : pd.DataFrame, test feature matrix
    feature_names: list of str
    shap_norm    : normalised shap.Explanation, used to select top N features
    set_label    : str
    target       : str
    top_n        : int, number of top features to include in heatmap (default 10)

    Outputs
    -------
    shap_interaction_heatmap_{set_label}_{target}.png
    """
    X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_arr  = X_test.values  if hasattr(X_test,  'values') else X_test
    n_features  = X_train_arr.shape[1]

    # ── Select top N features by normalised SHAP importance ───────────────────
    mean_abs = mean_abs_normalised_shap(shap_norm)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]

    # ── For high-dimensional sets: restrict to top N columns ──────────────────
    # This reduces Set E from 256×256 to top_n×top_n interaction computation
    if n_features > 50:
        print(f"    Set {set_label} has {n_features} features — restricting "
              f"interaction heatmap to top {top_n} features.")
        X_train_sub = X_train_arr[:, top_idx]
        X_test_sub  = X_test_arr[:,  top_idx]
        sub_names   = top_names

        # Build a surrogate model on the top-N feature subset for interaction values
        # This avoids memory issues with 256-dim interaction matrices
        from sklearn.metrics import roc_auc_score as _roc

        # Get scale_pos_weight from original model
        spw = getattr(model, 'scale_pos_weight', 1.0)

        surrogate = xgb.XGBClassifier(
            max_depth=model.max_depth if hasattr(model, 'max_depth') else 3,
            n_estimators=model.n_estimators if hasattr(model, 'n_estimators') else 100,
            learning_rate=model.learning_rate if hasattr(model, 'learning_rate') else 0.05,
            scale_pos_weight=spw,
            tree_method='hist',
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='auc'
        )

        # Surrogate trained with IPW weights not available here — use uniform weights
        # This is acceptable for the interaction heatmap which shows relative strengths
        print(f"    Training top-{top_n} surrogate model for interaction values...")
        # We need labels — use model's predictions on training set as soft labels
        y_soft = (model.predict_proba(X_train_arr)[:, 1] >= 0.5).astype(int)
        surrogate.fit(X_train_sub, y_soft, verbose=False)
    else:
        X_train_sub = X_train_arr
        X_test_sub  = X_test_arr
        sub_names   = top_names
        surrogate   = model
        # Re-restrict to top N even for low-dim sets
        X_train_sub = X_train_arr[:, top_idx]
        X_test_sub  = X_test_arr[:,  top_idx]

    print(f"    Computing SHAP interaction values ({len(sub_names)} × {len(sub_names)} matrix)...")

    try:
        explainer        = shap.TreeExplainer(surrogate, data=X_train_sub)
        interaction_vals = explainer.shap_interaction_values(X_test_sub)
        # interaction_vals shape: (n_test, n_features_sub, n_features_sub)
    except Exception as e:
        print(f"    Interaction values failed: {e}. Skipping heatmap.")
        return None

    # ── Mean absolute interaction matrix ──────────────────────────────────────
    # shape: (n_features_sub, n_features_sub)
    mean_abs_interaction = np.abs(interaction_vals).mean(axis=0)

    # Normalise the matrix so diagonal + off-diagonal sum to 1
    # Makes the heatmap readable regardless of absolute SHAP magnitude
    total = mean_abs_interaction.sum()
    if total > 0:
        mean_abs_interaction_norm = mean_abs_interaction / total
    else:
        mean_abs_interaction_norm = mean_abs_interaction

    # ── Plot heatmap ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, top_n * 0.75),
                                     max(7, top_n * 0.7)))

    # Use a diverging palette: white = 0 (no interaction), dark = strong
    import matplotlib.colors as mcolors
    cmap = plt.cm.Blues

    im = ax.imshow(mean_abs_interaction_norm, cmap=cmap, aspect='auto',
                    vmin=0, vmax=mean_abs_interaction_norm.max())

    # ── Annotate cells with values ─────────────────────────────────────────────
    thresh = mean_abs_interaction_norm.max() * 0.5
    for i in range(len(sub_names)):
        for j in range(len(sub_names)):
            val      = mean_abs_interaction_norm[i, j]
            txt_col  = 'white' if val > thresh else COLORS['navy']
            # Only annotate cells above a minimum threshold to reduce clutter
            if val > mean_abs_interaction_norm.max() * 0.05:
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=7.5, color=txt_col)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks(range(len(sub_names)))
    ax.set_yticks(range(len(sub_names)))

    # Truncate long feature names for readability
    short_names = [n[:22] + '…' if len(n) > 22 else n for n in sub_names]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8.5)
    ax.set_yticklabels(short_names, fontsize=8.5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label('Normalised mean |SHAP interaction|', fontsize=9)

    ax.set_title(
        f'SHAP Interaction Heatmap — XGBoost IPW\n'
        f'Set {set_label} · {target} · Top {top_n} features\n'
        f'Diagonal = main effects  |  Off-diagonal = pairwise interactions',
        fontsize=10, fontweight='bold', color=COLORS['navy'], pad=10
    )

    plt.tight_layout()
    path = os.path.join(
        SHAP_DIR, f'shap_interaction_heatmap_{set_label}_{target}.png'
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved SHAP interaction heatmap: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Main SHAP analysis pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_shap_analysis(data_csv, pred_csv=None, train_test_year=2,
                      sets_to_analyse=('C', 'D', 'E'),
                      targets=('pub_top_5pct', 'pub_w_top_5pct')):
    """
    Full SHAP + PDP analysis pipeline.

    Targets are fixed to pub_top_5pct and pub_w_top_5pct.
    Eight plot types produced per feature set × target:
      1. Beeswarm summary          — normalised SHAP (all features)
      2. SHAP bar chart            — mean |norm SHAP| top 10 features ± std
      3. Waterfall plots           — raw SHAP for each true positive
      4. Dependence plots          — normalised SHAP vs feature value (top 3)
      5. PDP                       — partial dependence for top 10 features
      6. SHAP interaction heatmap  — pairwise interaction strengths (top 10)
      7. SHAP interaction analysis — detailed plots for top interacting pairs
      8. Cross-set comparison      — normalised SHAP bar chart across C, D, E

    Parameters
    ----------
    data_csv         : str, path to 2015-2018_rookie_dataset.csv
    pred_csv         : str or None
                       If the XGBoost model has already been run:
                         path to output_prediction_main_2018_xgboost_ipw.csv
                       If None or not found: models are retrained from scratch.
    train_test_year  : int, 1=test2017, 2=test2018
    sets_to_analyse  : tuple of set labels ('C', 'D', 'E' recommended)
    targets          : tuple of target names
    """
    print("=" * 65)
    print("XGBoost IPW SHAP Analysis")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading data from {data_csv}...")
    data = pd.read_csv(data_csv, index_col=0)

    feature_matrices, y_train, y_test, treatment_train, train, test = \
        prepare_data(data, train_test_year=train_test_year)

    # ── Load predictions if available ─────────────────────────────────────────
    pred_df = None
    if pred_csv and os.path.exists(pred_csv):
        print(f"Loading existing predictions from {pred_csv}...")
        pred_df = pd.read_csv(pred_csv, index_col=0)
    else:
        print("No prediction CSV found — models will be retrained for SHAP.")

    # ── Main loop: feature set × target ───────────────────────────────────────
    for target in targets:
        print(f"\n{'─'*65}")
        print(f"TARGET: {target}")
        print(f"{'─'*65}")

        shap_results_by_set = {}   # {set_label: (sv_raw, shap_norm, feature_names)}

        for set_label in sets_to_analyse:
            print(f"\n  Set {set_label}:")
            X_train_full, X_test_full = feature_matrices[set_label]
            feature_names = list(X_train_full.columns)

            # ── Step 1: Retrain model ──────────────────────────────────────────
            print(f"  Step 1/8: Retrain best XGBoost (Set {set_label}, {target})")
            result = retrain_best_xgb(
                X_train_full,
                y_train[target],
                treatment_train,
                target=f'{set_label}_{target}'
            )
            model, X_full_r, best_params = result

            # ── Step 2: Compute raw SHAP ───────────────────────────────────────
            print(f"  Step 2/8: Compute SHAP values")
            X_test_r = X_test_full.reset_index(drop=True)
            shap_values_raw, explainer = compute_shap(
                model, X_full_r, X_test_r, feature_names
            )

            # ── Step 3: Normalise SHAP (Leinonen et al. 2023) ─────────────────
            print(f"  Step 3/8: Normalise SHAP (Leinonen et al. 2023)")
            shap_norm = normalise_shap(shap_values_raw)

            raw_sum    = np.abs(shap_values_raw.values).sum(axis=1).mean()
            normed_sum = np.abs(shap_norm.values).sum(axis=1).mean()
            print(f"    Raw   mean Σ|φ_i| per candidate: {raw_sum:.6f}")
            print(f"    Normed mean Σ|φ̃_i| per candidate: {normed_sum:.6f}  "
                  f"(should be 1.000000)")

            # Store both — raw for waterfall, norm for everything else
            shap_results_by_set[set_label] = (shap_values_raw, shap_norm,
                                               feature_names)

            # Load predictions if available, else compute fresh
            if pred_df is not None and f'{set_label}_{target}_score' in pred_df.columns:
                y_score = pred_df[f'{set_label}_{target}_score'].values
                y_true  = pred_df[f'{set_label}_{target}'].values
            else:
                y_score = model.predict_proba(X_test_r.values)[:, 1]
                y_true  = y_test[target].reset_index(drop=True).values

            # ── Step 4: Beeswarm (normalised) ─────────────────────────────────
            print(f"  Step 4/8: Beeswarm summary plot")
            plot_beeswarm(
                shap_norm, feature_names, set_label, target,
                max_display=15 if len(feature_names) > 50 else 20
            )

            # ── Step 5: SHAP bar chart — top 10 features ──────────────────────
            print(f"  Step 5/8: SHAP bar chart (top 10 features)")
            plot_shap_bar(
                shap_norm, feature_names, set_label, target,
                top_n=10
            )

            # ── Step 6: Waterfall (raw SHAP) ──────────────────────────────────
            print(f"  Step 6/8: Waterfall plots (true positives)")
            _, correct_tp, missed_tp = plot_waterfall_true_positives(
                shap_values_raw, shap_norm, y_true, y_score, set_label, target
            )

            # ── Step 7: Dependence plots (normalised, top 3) ──────────────────
            print(f"  Step 7/8: Dependence + PDP + interaction plots")
            plot_dependence(
                shap_norm, X_test_r, feature_names, set_label, target,
                top_n=3
            )

            # ── PDP for top 10 features ────────────────────────────────────────
            plot_pdp(
                model, X_full_r, feature_names, shap_norm,
                set_label, target, top_n=10
            )

            # ── SHAP interaction heatmap — top 10 features ────────────────────
            print(f"  Step 8/8: SHAP interaction heatmap (top 10 features)")
            plot_shap_interaction_heatmap(
                model, X_full_r, X_test_r, feature_names,
                shap_norm, set_label, target, top_n=10
            )

            # ── SHAP interaction detailed plots — top pairs ────────────────────
            analyse_interactions(
                model, X_full_r, X_test_r, feature_names,
                set_label, target, top_pairs=3
            )

        # ── Cross-set comparison (normalised, per target) ──────────────────────
        print(f"\n  Cross-set comparison plot (normalised SHAP)...")
        plot_comparison_bar(shap_results_by_set, target)

        # ── Console diagnostic summary ─────────────────────────────────────────
        print_top_features_summary(shap_results_by_set, target, top_n=10)

    print(f"\n{'='*65}")
    print(f"SHAP analysis complete. All plots saved to: {SHAP_DIR}/")
    print(f"{'='*65}")
    _print_file_summary()


def _print_file_summary():
    """Print all generated files grouped by type."""
    files = sorted(os.listdir(SHAP_DIR))
    groups = {
        'Beeswarm':            [f for f in files if f.startswith('beeswarm')],
        'SHAP bar chart':      [f for f in files if f.startswith('shap_bar')],
        'Waterfall':           [f for f in files if f.startswith('waterfall')],
        'Dependence':          [f for f in files if f.startswith('dependence')],
        'PDP':                 [f for f in files if f.startswith('pdp')],
        'Interaction heatmap': [f for f in files if f.startswith('shap_interaction_heatmap')],
        'Interaction detail':  [f for f in files if f.startswith('interaction_')],
        'Comparison bar':      [f for f in files if f.startswith('comparison')],
    }
    total = 0
    for group, flist in groups.items():
        if flist:
            print(f"\n  {group} ({len(flist)} files):")
            for f in flist:
                print(f"    {os.path.join(SHAP_DIR, f)}")
            total += len(flist)
    print(f"\n  Total files generated: {total}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SHAP analysis for XGBoost No-SMOTE IPW model'
    )
    parser.add_argument(
        '--data_csv',
        default='2015-2018_rookie_dataset.csv',
        help='Path to the rookie dataset CSV (default: 2015-2018_rookie_dataset.csv)'
    )
    parser.add_argument(
        '--pred_csv',
        default='output_prediction_main_2018_xgboost_ipw.csv',
        help='Path to existing XGBoost prediction CSV (optional). '
             'If found, predictions are loaded from here to identify '
             'correctly classified candidates without re-running the main model. '
             'If not found, models are retrained from scratch.'
    )
    parser.add_argument(
        '--run_model',
        action='store_true',
        help='Force retrain from scratch even if pred_csv exists'
    )
    parser.add_argument(
        '--sets',
        nargs='+',
        default=['C', 'D', 'E'],
        choices=['B', 'C', 'D', 'E', 'F'],
        help='Feature sets to analyse (default: C D E)'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        default=['pub_w_top_5pct', 'pub_top_5pct'],
        help='Target variables to analyse (default: pub_w_top_5pct pub_top_5pct)'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2,
        choices=[1, 2],
        help='Test year config: 1=test2017, 2=test2018 (default: 2)'
    )
    args = parser.parse_args()

    pred_path = None if args.run_model else args.pred_csv

    run_shap_analysis(
        data_csv         = args.data_csv,
        pred_csv         = pred_path,
        train_test_year  = args.year,
        sets_to_analyse  = tuple(args.sets),
        targets          = tuple(args.targets),
    )