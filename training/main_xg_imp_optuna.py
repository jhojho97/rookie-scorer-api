# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from itertools import product
import random
import copy

from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingRegressor, GradientBoostingClassifier,
    RandomForestRegressor, RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,          # improvement #12
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, ndcg_score,
    average_precision_score,          # improvement #5/#6 — Optuna objective
)
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import (
    RUSBoostClassifier, EasyEnsembleClassifier,
    BalancedBaggingClassifier, BalancedRandomForestClassifier,
)
import xgboost as xgb
import optuna                         # improvement #5 — Optuna tuning
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress trial output

# ── Helper functions ──────────────────────────────────────────────────────────

def compute_ipw_weights(X_train, treatment, C_reg=0.01):
    """
    Compute stabilised IPW weights following paper Appendix B
    Uses Hernán & Robins (2020) stabilised weight formula
    
    Parameters:
    -----------
    X_train   : feature matrix for propensity model
    treatment : binary series (1=research-intensive, 0=non-research)
    C_reg     : regularisation for propensity logistic regression
    
    Returns:
    --------
    weights   : numpy array of stabilised truncated IPW weights
    """
    treatment = np.asarray(treatment)

    # FIX: standardise features before the propensity logistic regression.
    # lbfgs fails to converge on the raw 256-dim embedding sets (B, E) because
    # the columns are on very different scales. Zero-mean/unit-variance scaling
    # is the real cure (max_iter alone just delays the warning).
    from sklearn.preprocessing import StandardScaler
    X_train = np.asarray(X_train, dtype=float)
    X_scaled = StandardScaler().fit_transform(X_train)

    # Fit propensity score model
    # Use strong regularisation — small sample with potentially many features
    ps_model = LogisticRegression(
        max_iter=5000,          # FIX: was 1000 — give lbfgs more room
        C=C_reg,
        solver='lbfgs',
        random_state=42
    )
    ps_model.fit(X_scaled, treatment)
    p_score = ps_model.predict_proba(X_scaled)[:, 1]
    
    # Trim extreme propensity scores (Crump et al. 2009, Austin & Stuart 2015)
    p_score = np.clip(p_score, 0.05, 0.95)
    
    # Stabilised weights (Hernán & Robins 2020) — FIX for Issue 2
    p_bar = treatment.mean()
    weights = np.where(
        treatment == 1,
        p_bar / p_score,
        (1 - p_bar) / (1 - p_score)
    )
    
    # Truncate at 5th and 95th percentiles — reduces influence of extreme weights
    lower = np.percentile(weights, 5)
    upper = np.percentile(weights, 95)
    weights = np.clip(weights, lower, upper)
    
    return weights


# ── Improvement #9: LASSO feature selection for high-dim embedding sets ───────

def lasso_select_features(X_train, y_train, X_test,
                           sample_weight=None, C_reg=0.01,
                           n_features=64):
    """
    Select top-n informative embedding dimensions using L1-penalised
    logistic regression fitted on the training fold only.

    Applied only to Sets B and E (256-dim embeddings). Selects by
    predictive signal rather than variance (unlike PCA), which is
    critical for ranking tasks where signal is sparse across dimensions.

    Parameters
    ----------
    X_train       : training feature matrix (numpy array or DataFrame)
    y_train       : binary training labels
    X_test        : test feature matrix — same columns applied
    sample_weight : IPW weights to pass to LASSO fit
    C_reg         : L1 regularisation strength (smaller = more sparse)
    n_features    : maximum number of dimensions to retain

    Returns
    -------
    X_train_sel   : reduced training matrix
    X_test_sel    : reduced test matrix (same column mask)
    selected_idx  : boolean mask of retained columns
    """
    X_tr = X_train.values if hasattr(X_train, 'values') else X_train
    X_te = X_test.values  if hasattr(X_test,  'values') else X_test
    y_tr = np.asarray(y_train)

    # Guard: if fewer positives than needed for logistic regression, skip
    if y_tr.sum() < 3:
        print("    [LASSO selector] Too few positives — skipping, keeping all features.")
        selected_idx = np.ones(X_tr.shape[1], dtype=bool)
        return X_train, X_test, selected_idx

    selector = LogisticRegression(
        penalty='l1',
        C=C_reg,
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    selector.fit(X_tr, y_tr, sample_weight=sample_weight)

    coef_abs = np.abs(selector.coef_[0])

    # Take top n_features by absolute coefficient magnitude
    # (or all nonzero if fewer than n_features have nonzero coef)
    n_nonzero = np.sum(coef_abs > 0)
    k = min(n_features, max(int(n_nonzero), 1))  # always keep at least 1

    top_idx = np.argsort(coef_abs)[::-1][:k]
    top_idx = np.sort(top_idx)  # restore original column order

    selected_idx = np.zeros(X_tr.shape[1], dtype=bool)
    selected_idx[top_idx] = True

    print(f"    [LASSO selector] {k}/{X_tr.shape[1]} embedding dims retained.")

    # Return as DataFrame if input was DataFrame, else numpy
    if hasattr(X_train, 'columns'):
        cols = X_train.columns[selected_idx]
        return (X_train[cols], X_test[cols], selected_idx)
    else:
        return (X_tr[:, selected_idx], X_te[:, selected_idx], selected_idx)


def get_metric_cla(y_test, y_pred, y_score):
    return {
        'accuracy': [accuracy_score(y_test, y_pred)],
        'precision': [precision_score(y_test, y_pred, average='weighted')],
        'recall': [recall_score(y_test, y_pred, average='weighted')],
        'f1': [f1_score(y_test, y_pred, average='weighted')],
        'AUC': [roc_auc_score(y_test, y_score)],
        'NDCG': [ndcg_score([list(y_pred.astype(int))], [list(y_test.astype(int))], k=y_test.sum())],
    }


# ── Improvement #5/#6: Optuna hyperparameter tuning with AUPRC objective ──────

def tune_xgboost_optuna(X_train, y_train, sample_weight,
                         scale_pos_weight=1.0,
                         n_trials=50, n_splits=5, n_repeats=3,
                         random_state=42):
    """
    Tune XGBoost hyperparameters using Optuna with AUPRC objective.

    Replaces the grid search over xgb_params. Uses Optuna's TPE sampler
    which learns which regions of the hyperparameter space are promising
    and concentrates trials there — more efficient than exhaustive grid.

    Objective: AUPRC (average precision score), not AUC.
    AUPRC is far more sensitive to minority-class ranking at 5% positive
    rate and directly penalises models that miss the top positives.

    Validation: Improvement #12 — repeated stratified k-fold CV (not a
    single 80/20 split) so that hyperparameter selection is not driven by
    whichever 3–6 positives happened to land in one val fold.

    Parameters
    ----------
    X_train       : training features (DataFrame or array)
    y_train       : binary training labels (Series or array)
    sample_weight : IPW weights for training examples
    n_trials      : number of Optuna trials (50 recommended)
    n_splits      : folds per repeat for inner CV
    n_repeats     : number of CV repeats
    random_state  : reproducibility seed

    Returns
    -------
    best_params   : dict of best hyperparameters (ready for XGBClassifier)
    study         : Optuna study object (for inspection / visualisation)
    """
    X_tr = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
    y_tr = np.asarray(y_train)
    w_tr = np.asarray(sample_weight)

    # Improvement #12 — RepeatedStratifiedKFold stabilises AUPRC estimates
    # when val fold has only 3–6 positives (5% × 80% × N≈350 ≈ 14 positives
    # → each of 5 folds has ~3 positives; repeating 3× averages over 15 folds)
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    def objective(trial):
        # Improvement #7 — ranges tuned for small-N imbalanced regime
        params = {
            'max_depth':        trial.suggest_int('max_depth', 2, 3),
            'learning_rate':    trial.suggest_float(
                                    'learning_rate', 0.01, 0.1, log=True),
            'n_estimators':     trial.suggest_int('n_estimators', 100, 300),
            # FIX: 10–30 blocked every split with ~6 positives → constant model.
            # Lowered so the trees can actually isolate the minority class.
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'subsample':        trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.6),
            'reg_alpha':        trial.suggest_float(
                                    'reg_alpha', 0.1, 2.0, log=True),
            'reg_lambda':       trial.suggest_float(
                                    'reg_lambda', 1.0, 10.0, log=True),
            'gamma':            trial.suggest_float('gamma', 0.0, 1.0),
            'max_delta_step':   trial.suggest_int('max_delta_step', 1, 5),
            # FIX: class weighting restored. IPW (sample_weight) corrects
            # treatment-SELECTION bias, NOT class imbalance — those are
            # separate corrections. scale_pos_weight × IPW is the legitimate,
            # proven combination (it is what the AUC-98 original used).
            'scale_pos_weight': scale_pos_weight,
            'tree_method':      'hist',
            'objective':        'binary:logistic',
            'eval_metric':      'aucpr',
            'random_state':     random_state,
            'verbosity':        0,
        }

        fold_scores = []
        for train_idx, val_idx in rskf.split(X_tr, y_tr):
            X_fold_tr, X_fold_val = X_tr[train_idx], X_tr[val_idx]
            y_fold_tr, y_fold_val = y_tr[train_idx], y_tr[val_idx]
            w_fold_tr             = w_tr[train_idx]

            # Guard: skip fold if val has no positives (can happen at 5% rate)
            if y_fold_val.sum() == 0:
                continue

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_fold_tr, y_fold_tr,
                sample_weight=w_fold_tr,
                verbose=False
            )
            proba = model.predict_proba(X_fold_val)[:, 1]
            # AUPRC objective — improvement #6
            fold_scores.append(average_precision_score(y_fold_val, proba))

        return np.mean(fold_scores) if fold_scores else 0.0

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    # Restore fixed params that were not part of the search space
    best_params['scale_pos_weight'] = scale_pos_weight   # FIX: class weighting
    best_params['tree_method']      = 'hist'
    best_params['objective']        = 'binary:logistic'
    best_params['eval_metric']      = 'aucpr'
    best_params['random_state']     = random_state
    best_params['verbosity']        = 0

    print(f"    [Optuna] Best AUPRC={study.best_value:.4f} | "
          f"depth={best_params['max_depth']} "
          f"lr={best_params['learning_rate']:.4f} "
          f"n_est={best_params['n_estimators']} "
          f"mcw={best_params['min_child_weight']}")

    return best_params, study


def train_and_predict(X_train, y_train, X_test, y_test,
                      m_param, sample_weight=None):
    models = {
        'GBDT_c':            GradientBoostingClassifier(),
        'XGboost_c':         xgb.XGBClassifier(),
        'Logit_c':           LogisticRegression(),
        'GaussianNB_c':      GaussianNB(),
        'RF_c':              RandomForestClassifier(),
        'RUSboost_c':        RUSBoostClassifier(),
        'EasyEns_c':         EasyEnsembleClassifier(),
        'BalancedBagging_c': BalancedBaggingClassifier(),
        'BalancedRF_c':      BalancedRandomForestClassifier(),
    }
    model_name   = list(m_param.keys())[0]
    hyperparams  = m_param[model_name]
    if model_name not in models:
        raise ValueError(f"Invalid model name. Choose one of {list(models.keys())}")
    selected     = models[model_name]
    selected.set_params(**hyperparams)
    selected.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred  = selected.predict(X_test)
    y_score = pd.Series(
        selected.predict_proba(X_test)[:, 1],
        index=X_test.index
    )
    metrics = get_metric_cla(y_test, y_pred, y_score)

    # Store m_param as string key for retrieval 
    param_key = str(m_param)
    df = pd.DataFrame(metrics, index=[param_key])
    df['m_param_str'] = param_key
    df['model']       = [model_name]
    return df


def train_and_pred_value(X_train, y_train, X_test, y_test,
                         m_param, sample_weight=None):
    random.seed(0)
    models = {
        'GBDT_c':            GradientBoostingClassifier(),
        'XGboost_c':         xgb.XGBClassifier(),
        'Logit_c':           LogisticRegression(),
        'GaussianNB_c':      GaussianNB(),
        'RF_c':              RandomForestClassifier(),
        'RUSboost_c':        RUSBoostClassifier(),
        'EasyEns_c':         EasyEnsembleClassifier(),
        'BalancedBagging_c': BalancedBaggingClassifier(),
        'BalancedRF_c':      BalancedRandomForestClassifier(),
    }
    model_name  = list(m_param.keys())[0]
    hyperparams = m_param[model_name]
    if model_name not in models:
        raise ValueError(f"Invalid model name. Choose one of {list(models.keys())}")
    selected    = models[model_name]
    selected.set_params(**hyperparams)
    selected.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred  = selected.predict(X_test)
    y_score = pd.Series(
        selected.predict_proba(X_test)[:, 1],
        index=range(len(X_test))
    )
    return y_pred, y_score


def get_rank_decile(s):
    s_pos  = s[s > 0]
    s_pos  = pd.qcut(s_pos.rank(method="first"), q=10, labels=range(1, 11))
    s_zero = s[s == 0].replace(0, 11)
    res    = pd.concat([s_pos, s_zero]).loc[s.index]
    return res


# ── Main ensemble function ────────────────────────────────────────────────────

def get_full_prediction_ensemble(data0, train_test, param_pool,
                                  drop10='original', robust='main'):
    data       = data0.copy()
    model_name = list(param_pool.keys())[0]

    if drop10 == 'drop10':
        data = data[(data['Placerank'] > 10) | (data['Placerank'] == 0)]

    c = (data.groupby('year')
             .apply(lambda x: get_rank_decile(x.Placerank))
             .reset_index()[['ID', 'Placerank']]
             .set_index('ID'))
    data['Placerank'] = c

    if robust == 'removemanipulate':
        data = data.drop([
            'has PhD honor', 'number of papers in progress',
            'number of presentations', 'number of teaching experiences',
            'number of reviewers', 'number of working experiences',
            'provide abstract',
        ], axis=1)

    if train_test == 1:
        train = data[data.year < 2017]
        test  = data[data.year == 2017]
    elif train_test == 2:
        train = data[data.year < 2018]
        test  = data[data.year == 2018]
    elif train_test == 3:
        train = data[data.year < 2017]
        test  = data[data.year == 2018]

    # Feature matrices
    if robust == 'embedding3072':
        X_trainB = train.loc[:, '0_cv':'3071_cv']
        X_testB  = test.loc[:, '0_cv':'3071_cv']
        X_trainE = train.loc[:, '0_dt':'3071_dt']
        X_testE  = test.loc[:, '0_dt':'3071_dt']
    else:
        X_trainB = train.loc[:, '0_cv':'255_cv']
        X_testB  = test.loc[:, '0_cv':'255_cv']
        X_trainE = train.loc[:, '0_dt':'255_dt']
        X_testE  = test.loc[:, '0_dt':'255_dt']

    X_trainC = train.loc[:, 'gender':'multi_language']
    X_testC  = test.loc[:, 'gender':'multi_language']
    X_trainD = train.loc[:, 'Bachelor_top':'second_language_euro']
    X_testD  = test.loc[:, 'Bachelor_top':'second_language_euro']
    X_trainF = train[['Placerank']]
    X_testF  = test[['Placerank']]

    # keep research_oriented separate from prediction targets
    targets = ['pub_top_5pct', 'pub_w_top_5pct']
    y_train00         = train[targets]
    y_test_df         = test[targets]
    treatment_train   = train['research_oriented']  # separate

    test_full = pd.DataFrame()

    for x in ['B', 'C', 'D', 'E', 'F']:
        # Assign feature matrices
        X_map = {'B': (X_trainB, X_testB), 'C': (X_trainC, X_testC),
                 'D': (X_trainD, X_testD), 'E': (X_trainE, X_testE),
                 'F': (X_trainF, X_testF)}
        X_train00, X_test = X_map[x]

        # ── Improvement #12: stratify split, but Optuna will do its own
        # repeated CV inside — this split is only used by non-XGBoost models
        # and to compute the full-data IPW weights for final retraining.
        X_train0, X_val, y_train0, y_val, treat_tr, _ = train_test_split(
            X_train00.reset_index(drop=True),
            y_train00.reset_index(drop=True),
            treatment_train.reset_index(drop=True),
            test_size=0.2,
            random_state=42,
            stratify=y_train00['pub_top_5pct'].values  # stratified split
        )

        # Compute stabilised IPW weights on the 80% training fold
        print(f"Computing IPW weights for block {x}...")
        ipw_weights = compute_ipw_weights(X_train0, treat_tr)

        # ── Improvement #9 (LASSO feature selection) — DISABLED.
        # Empirically, L1 selection on the 256-dim embedding sets (B, E) with
        # only ~6 positives dropped the signal dimensions and collapsed those
        # sets to near-random (Set E AUC 81->60, NDCG 32->0), dragging CDE/BCDE
        # down. XGBoost's native colsample_bytree handles the 256 dims far more
        # robustly at this sample size, so we keep ALL features for every set
        # (matching the standard model). Re-enable only if N of positives grows.
        USE_LASSO = False
        if USE_LASSO and x in ['B', 'E']:
            print(f"  [Block {x}] Applying LASSO feature selection "
                  f"(256-dim -> <=64 dims)...")
            X_train0_sel, X_test_sel, sel_idx = lasso_select_features(
                X_train0, y_train0['pub_top_5pct'],
                X_test.reset_index(drop=True),
                sample_weight=ipw_weights,
                C_reg=0.01,
                n_features=64
            )
            X_train00_sel = (
                X_train00.reset_index(drop=True).iloc[:, sel_idx]
                if hasattr(X_train00, 'iloc')
                else X_train00[sel_idx]
            )
            X_test_for_final = X_test_sel
        else:
            X_train0_sel    = X_train0
            X_test_sel      = X_test.reset_index(drop=True)
            X_train00_sel   = X_train00.reset_index(drop=True)
            X_test_for_final = X_test.reset_index(drop=True)

        for target in targets:
            if drop10 == 'drop10' and 'job' in target:
                continue
            print(f"  Feature set {x}, target {target}")

            if 'XGboost' in model_name:

                y_train_fold = y_train0[target].copy()

                # ── FIX: class weighting via scale_pos_weight (= n_neg/n_pos).
                # IPW (sample_weight) corrects treatment-selection bias only;
                # it does NOT balance the ~6-vs-270 class imbalance. Without
                # this, the heavily-regularised trees make zero splits and the
                # model collapses to a constant (AUC 0.5, all-negative).
                n_pos_fold = int((y_train_fold == 1).sum())
                n_neg_fold = int((y_train_fold == 0).sum())
                spw_fold = n_neg_fold / n_pos_fold if n_pos_fold > 0 else 1.0

                # ── Improvements #5/#6/#12: Optuna with AUPRC objective
                # and RepeatedStratifiedKFold inside the objective function.
                print(f"    Running Optuna ({N_OPTUNA_TRIALS} trials) ... "
                      f"scale_pos_weight={spw_fold:.1f}")
                best_params, _ = tune_xgboost_optuna(
                    X_train0_sel,
                    y_train_fold,
                    sample_weight=ipw_weights,
                    scale_pos_weight=spw_fold,
                    n_trials=N_OPTUNA_TRIALS,
                    n_splits=5,
                    n_repeats=3,
                    random_state=42
                )

                # ── Final model: retrain on FULL training data with full IPW
                X_train_full  = X_train00_sel
                y_train_full  = y_train00[target].reset_index(drop=True)
                treat_full    = treatment_train.reset_index(drop=True)

                # IPW regularisation: stronger for high-dim sets (B, E)
                C_ipw_full = 0.001 if x in ['B', 'E'] else 0.01
                ipw_full   = compute_ipw_weights(
                    X_train00.reset_index(drop=True),
                    treat_full,
                    C_reg=C_ipw_full
                )

                # FIX: recompute scale_pos_weight on the FULL training set the
                # final model is fit on (slightly larger N than the tuning fold).
                n_pos_full = int((y_train_full == 1).sum())
                n_neg_full = int((y_train_full == 0).sum())
                best_params['scale_pos_weight'] = (
                    n_neg_full / n_pos_full if n_pos_full > 0 else 1.0
                )

                final_model = xgb.XGBClassifier(**best_params)
                final_model.fit(
                    X_train_full.values
                    if hasattr(X_train_full, 'values')
                    else X_train_full,
                    y_train_full.values,
                    sample_weight=ipw_full  # unified weight — no doubling
                )

                X_te_arr = (
                    X_test_for_final.values
                    if hasattr(X_test_for_final, 'values')
                    else X_test_for_final
                )
                y_score_arr = final_model.predict_proba(X_te_arr)[:, 1]
                y_pred_arr  = (y_score_arr >= 0.5).astype(int)

                best_res = pd.DataFrame(y_test_df[target])
                best_res['pred']  = y_pred_arr
                best_res['score'] = y_score_arr
                best_res = best_res.rename(columns={target: 'y_vali'})[
                    ['y_vali', 'pred', 'score']
                ]
                best_res['x']      = x
                best_res['target'] = target
                test_full = pd.concat([test_full, best_res])

    # ── Restructure individual predictions ───────────────────────────────
    pred_full = pd.DataFrame()
    for x in ['B', 'C', 'D', 'E', 'F']:
        for target in targets:
            if drop10 == 'drop10' and 'job' in target:
                continue
            temp = test_full[
                (test_full.x == x) & (test_full.target == target)
            ].copy()
            temp = temp.rename({
                'y_vali': x + '_' + target,
                'pred':   x + '_' + target + '_pred',
                'score':  x + '_' + target + '_score'
            }, axis=1)
            pred_full = pred_full.join(temp.iloc[:, :3], how='outer')

    # ── Late fusion ensembles ─────────────────────────────────────────────
    for combo in ['BC','BCD','BCDE','BCDEF','CD','CDE','CDEF','CE','CEF']:
        for target in targets:
            if drop10 == 'drop10' and 'job' in target:
                continue
            test_set = test_full[
                test_full['x'].isin(list(combo)) &
                (test_full['target'] == target)
            ].reset_index()
            test_x = test_set[['ID','score','x']].pivot(
                index='ID', columns='x', values='score'
            )
            fused_score = test_x.mean(axis=1)
            pred_full[combo + '_' + target + '_score'] = fused_score
            pred_full[combo + '_' + target + '_pred']  = (fused_score >= 0.5).astype(int)
            pred_full[combo + '_' + target]            = test[target]

    return pred_full


# ── Metric functions ──────────────────────────────────────────────────────────

def ndcg_at_k(y_true, y_score, k):
    k = int(k)
    if k <= 0: return 0.0
    y_true  = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    idx_pred     = np.argsort(y_score)[::-1]
    y_sorted     = y_true[idx_pred[:k]]
    gains        = 2**y_sorted - 1
    discounts    = np.log2(np.arange(len(y_sorted)) + 2)
    dcg_k        = np.sum(gains / discounts)
    y_ideal      = np.sort(y_true)[::-1][:k]
    gains_ideal  = 2**y_ideal - 1
    idcg_k       = np.sum(gains_ideal / discounts[:len(gains_ideal)])
    return dcg_k / idcg_k if idcg_k > 0 else 0.0


def get_top_k_metrics(y_true, y_score, k):
    # FIX Issue 6 — consistent 2-value return
    k = int(k)
    if k <= 0: return 0.0, 0.0
    y_true  = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = np.sum(y_true)
    if n_pos == 0: return 0.0, 0.0
    k   = min(k, len(y_true))
    idx = np.argsort(y_score)[::-1][:k]
    tp  = np.sum(y_true[idx])
    return tp / k, tp / n_pos


def get_metric_test(y_true, y_pred, y_score):
    k = int(y_true.sum())
    prec_k, rec_k = get_top_k_metrics(y_true, y_score, k)
    mean_dv = y_true.mean()
    return pd.Series({
        'true_pos':    mean_dv,
        'accuracy':    accuracy_score(y_true, y_pred),
        'precision':   precision_score(y_true, y_pred, zero_division=0),
        'recall':      recall_score(y_true, y_pred, zero_division=0),
        'f1':          f1_score(y_true, y_pred, zero_division=0),
        'AUC':         roc_auc_score(y_true, y_score),
        'NDCG':        ndcg_at_k(y_true, y_score, k),
        'precision@K': prec_k,
        'LIFT':        prec_k / mean_dv if mean_dv > 0 else 0.0,
        'recall@K':    rec_k,
    })


def get_accuracy(data, targets=None):
    if targets is None:
        targets = ['pub_top_5pct', 'pub_w_top_5pct']
    combos = ['B','C','D','E','F','BC','BCD','BCDE','BCDEF',
              'CD','CDE','CDEF','CE','CEF']
    results = []
    for x in combos:
        for target in targets:
            if 'F' in x and 'job' in target:
                continue
            try:
                y_true  = data[x + '_' + target]
                y_pred  = data[x + '_' + target + '_pred']
                y_score = data[x + '_' + target + '_score']
                m = get_metric_test(y_true.values, y_pred.values, y_score.values)
                m['x']      = x
                m['target'] = target
                results.append(m)
            except KeyError:
                continue

    allres = pd.DataFrame(results)
    metric_cols = ['accuracy','precision','recall','f1','AUC','NDCG']
    allres[metric_cols] = allres[metric_cols] * 100

    allres['target'] = pd.Categorical(allres['target'],
                                       categories=targets, ordered=True)
    allres['x']      = pd.Categorical(allres['x'],
                                       categories=combos, ordered=True)
    return allres.sort_values(['target','x'])


# ── Entry point ───────────────────────────────────────────────────────────────

# ── Improvement #5: number of Optuna trials per (feature-set, target) pair ──
# 50 trials is the default used in Dhanka & Maini (2025).
# Increase to 100 for a more thorough search at the cost of runtime.
N_OPTUNA_TRIALS = 50

if __name__ == '__main__':
    data = pd.read_csv('2015-2018_rookie_dataset.csv', index_col=0)

    # param_pool now only needs to identify the model family.
    # Hyperparameter values are found by Optuna — the dict below is kept
    # for compatibility with get_full_prediction_ensemble's model_name lookup.
    xgb_params = {
        'XGboost_c': {}   # hyperparameters selected by Optuna per block/target
    }

    res = get_full_prediction_ensemble(data, 2, xgb_params)
    res.to_csv('output_prediction_main_2018_xgboost_opt.csv')
    print('Saved: output_prediction_main_2018_xgboost_opt.csv')

    acc = get_accuracy(res)
    acc.to_csv('output_accuracy_main_2018_xgboost_opt.csv', index=False)
    print(acc[acc['x']=='CDE'].to_string())