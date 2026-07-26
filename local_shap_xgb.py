"""
local_shap_xgb.py
=================
LOCAL (per-candidate) SHAP explanation for the XGBoost No-SMOTE IPW ensemble,
for use in the web app: given one candidate's features, return the top factors
that pushed their predicted research potential up or down, with the underlying
feature values.

Why local SHAP (and is there a better way?)
-------------------------------------------
For a tree model, the right tool for per-instance attribution is **TreeSHAP**,
which is exactly what `shap.TreeExplainer` computes. It is:
  • exact (not a sampling approximation like KernelSHAP/LIME),
  • additive — the factor contributions sum to (prediction − baseline), so the
    explanation is faithful to the model, and
  • fast — milliseconds per candidate once the explainer is built.

Alternatives considered:
  • LIME — local linear surrogate; slower, unstable across runs, only
    approximate. Worse than TreeSHAP for trees. Not recommended.
  • XGBoost native `pred_contribs=True` — this IS TreeSHAP, computed inside
    XGBoost with no `shap` dependency. Fastest option for production serving.
    See `explain_native()` below for that path; it returns the same numbers as
    TreeExplainer (in log-odds space).
  • Counterfactuals ("raise top-journal pubs by 1 → +6pp") — a nice complement
    for UX, not a replacement. Easy to add later from the same model.

Recommendation: TreeSHAP is the correct and best choice here. Use the
`LocalExplainer` class (probability-space, matches shap_xgb.py) for rich
explanations, or `explain_native()` for the leanest production endpoint.

The ensemble combines Sets C, D, E by averaging their probabilities, so each
feature's contribution to the final score is its in-set SHAP value divided by
the number of sets. Set E's 256 embedding dims are collapsed into one
non-technical "Job market paper content" factor.

Usage (CLI)
-----------
# Explain every candidate in the DeepSeek inference dataset:
python local_shap_xgb.py \
    --candidate_csv deepseek_pipeline/deepseek_output/inference_dataset.csv \
    --data_csv data/2015-2018_rookie_dataset.csv \
    --target pub_w_top_5pct --top_n 5 --out explanations.json

# Explain a single candidate (row 0) and also save a bar chart:
python local_shap_xgb.py --candidate_csv ... --row 0 --plot

Usage (web backend)
-------------------
    from local_shap_xgb import LocalExplainer
    explainer = LocalExplainer(data_csv="data/2015-2018_rookie_dataset.csv",
                               target="pub_w_top_5pct")      # build once
    result = explainer.explain(candidate_row, top_n=5)        # per request
"""

import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import shap

# Reuse the exact training / data-prep logic from the global SHAP script
from shap_xgb import retrain_best_xgb, prepare_data

warnings.filterwarnings("ignore")


# ── Human-readable labels for the interpretable feature sets (C and D) ────────
FRIENDLY_LABELS = {
    # Set C
    "gender": "Gender (male)",
    "has Bachelor honor": "Bachelor's honors",
    "has Master honor": "Master's honors",
    "has PhD honor": "PhD honors",
    "number of published papers": "Number of published papers",
    "number of R&R papers": "Number of R&R papers",
    "number of papers in progress": "Papers in progress",
    "has_coauthor": "Has coauthor(s)",
    "has_reference": "Has reference(s)",
    "number of coauthors": "Number of coauthors",
    "number of presentations": "Number of presentations",
    "number of teaching experiences": "Teaching experiences",
    "number of awards": "Number of awards",
    "number of reviewers": "Reviewer roles",
    "number of membership": "Professional memberships",
    "number of working experiences": "Work experiences",
    "had academic work": "Prior academic work",
    "had non-academic work": "Prior non-academic work",
    "provide abstract": "Provides paper abstracts in CV",
    "PrimaryResearchArea_financial": "Primary area: Financial",
    "PrimaryResearchArea_auditing": "Primary area: Auditing",
    "PrimaryResearchArea_managerial": "Primary area: Managerial",
    "PrimaryResearchArea_tax": "Primary area: Tax",
    "PrimaryResearchMethod_archival": "Primary method: Archival",
    "PrimaryResearchMethod_experiment": "Primary method: Experimental",
    "PrimaryResearchMethod_analytical": "Primary method: Analytical",
    "multi_language": "Speaks multiple languages",
    # Set D
    "Bachelor_top": "Bachelor from a top-50 school",
    "Master_top": "Master from a top-50 school",
    "PhD_top": "PhD from a top-50 school",
    "visit_top": "Visited a top-50 school",
    "number of top published papers": "Top-journal publications",
    "number of top R&R papers": "Top-journal R&Rs",
    "coauthor_mean": "Average coauthor rank",
    "coauthor_high": "Best coauthor rank",
    "number_of_coauthor_2degree": "2nd-degree coauthor network size",
    "coauthor_top": "Has a top-1% coauthor",
    "number of presentations on top conferences": "Top-conference presentations",
    "reference_first": "First reference rank",
    "reference_mean": "Average reference rank",
    "reference_high": "Best reference rank",
    "second_language_asia": "Second language (Asian)",
    "second_language_euro": "Second language (European)",
}

# Binary features → render value as Yes/No
BINARY_FEATURES = {
    "gender", "has Bachelor honor", "has Master honor", "has PhD honor",
    "has_coauthor", "has_reference", "had academic work", "had non-academic work",
    "provide abstract", "multi_language",
    "PrimaryResearchArea_financial", "PrimaryResearchArea_auditing",
    "PrimaryResearchArea_managerial", "PrimaryResearchArea_tax",
    "PrimaryResearchMethod_archival", "PrimaryResearchMethod_experiment",
    "PrimaryResearchMethod_analytical",
    "Bachelor_top", "Master_top", "PhD_top", "visit_top", "coauthor_top",
    "second_language_asia", "second_language_euro",
}

# Column ranges per feature set (mirror prepare_data in shap_xgb.py)
SET_SLICES = {
    "C": ("gender", "multi_language"),
    "D": ("Bachelor_top", "second_language_euro"),
    "E": ("0_dt", "255_dt"),
}
EMBEDDING_LABEL = "Job market paper content"   # collapsed Set E


def _fmt_value(feature: str, value):
    """Format a raw feature value for display."""
    if feature in BINARY_FEATURES:
        return "Yes" if float(value) >= 0.5 else "No"
    v = float(value)
    return int(v) if v == int(v) else round(v, 3)


class LocalExplainer:
    """
    Build the per-set models + TreeExplainers once, then explain many
    candidates cheaply. Designed to be instantiated once at web-app startup.
    """

    def __init__(self, data_csv, target="pub_w_top_5pct",
                 sets=("C", "D", "E"), train_test_year=2):
        self.target = target
        self.sets = tuple(sets)
        self.models = {}        # set_label -> fitted model
        self.backgrounds = {}   # set_label -> training matrix (DataFrame)
        self.feature_names = {} # set_label -> [col names]
        self.explainers = {}    # set_label -> shap.TreeExplainer

        print(f"Building local explainer for target='{target}', sets={self.sets} ...")
        data = pd.read_csv(data_csv, index_col=0)
        feature_matrices, y_train, _, treatment_train, _, _ = prepare_data(
            data, train_test_year=train_test_year
        )

        # Interventional probability SHAP integrates over the background, so its
        # memory/compute scale with the background size (heaviest for the 256-dim
        # embedding model). Subsample it to keep the serving process well under
        # small-instance RAM limits (e.g. Render 512MB). SHAP stays additive, so
        # base + sum(phi) == model output regardless of this size — the PREDICTION
        # is unchanged; only the baseline/attribution split shifts slightly.
        bg_n = int(os.environ.get("ROOKIE_SHAP_BG", "40"))
        for s in self.sets:
            X_train_full, _ = feature_matrices[s]
            # Reuse cached model if shap_xgb.py already trained it (same key)
            model, X_bg, _ = retrain_best_xgb(
                X_train_full, y_train[target], treatment_train,
                target=f"{s}_{target}",
            )
            if len(X_bg) > bg_n:
                X_bg = X_bg.sample(n=bg_n, random_state=42)
            self.models[s] = model
            self.backgrounds[s] = X_bg
            self.feature_names[s] = list(X_train_full.columns)
            # Probability-space interventional TreeExplainer (matches shap_xgb.py)
            self.explainers[s] = shap.TreeExplainer(
                model,
                data=X_bg.values,
                feature_perturbation="interventional",
                model_output="probability",
            )
        print("  Explainer ready.")

    # ── Core: explain one candidate ──────────────────────────────────────────
    def explain(self, candidate_row, top_n=5, collapse_embeddings=True):
        """
        candidate_row : pd.Series or dict containing at least the Set C/D/E
                        columns (e.g. a row of the DeepSeek inference dataset).

        Returns a dict:
          {
            "target", "prediction", "baseline",
            "top_factors": [
               {"feature","label","value","contribution","direction","set"}, ...
            ]
          }
        Contributions are on the final (ensemble-averaged) probability scale and
        sum (across ALL features, not just the top_n) to prediction − baseline.
        """
        if isinstance(candidate_row, dict):
            candidate_row = pd.Series(candidate_row)

        n_sets = len(self.sets)
        factors = []          # list of dicts (one per interpretable feature)
        set_probs = []        # per-set predicted probability
        set_bases = []        # per-set baseline E[f(x)]
        skipped = []

        for s in self.sets:
            names = self.feature_names[s]
            # Align the candidate to this set's columns
            try:
                x = candidate_row[names].astype(float).values.reshape(1, -1)
            except KeyError:
                skipped.append(s)
                continue
            if np.isnan(x).any():
                skipped.append(s)         # e.g. Set E missing (--no_embeddings)
                continue

            expl = self.explainers[s](x)
            phi = np.asarray(expl.values).reshape(-1)        # (n_features,)
            base = float(np.asarray(expl.base_values).reshape(-1)[0])
            set_probs.append(base + phi.sum())
            set_bases.append(base)

            if s == "E" and collapse_embeddings:
                # Collapse 256 embedding dims into one non-technical factor
                factors.append({
                    "feature": "_embedding_E",
                    "label": EMBEDDING_LABEL,
                    "value": None,                       # text — no scalar value
                    "contribution": float(phi.sum()),    # in-set, pre-averaging
                    "set": s,
                })
            else:
                for i, name in enumerate(names):
                    factors.append({
                        "feature": name,
                        "label": FRIENDLY_LABELS.get(name, name),
                        "value": candidate_row[name],
                        "contribution": float(phi[i]),
                        "set": s,
                    })

        if not set_probs:
            raise ValueError("No usable feature sets for this candidate "
                             f"(skipped: {skipped}). Check the input columns.")

        # Ensemble = average of per-set probabilities; scale contributions by 1/n
        n_used = len(set_probs)
        prediction = float(np.mean(set_probs))
        baseline = float(np.mean(set_bases))
        for f in factors:
            f["contribution"] /= n_used
            f["direction"] = "increases" if f["contribution"] >= 0 else "decreases"
            if f["value"] is not None:
                f["value"] = _fmt_value(f["feature"], f["value"])

        factors.sort(key=lambda d: abs(d["contribution"]), reverse=True)

        return {
            "target": self.target,
            "prediction": round(prediction, 4),
            "baseline": round(baseline, 4),
            "sets_used": [s for s in self.sets if s not in skipped],
            "sets_skipped": skipped,
            "top_factors": [
                {
                    "label": f["label"],
                    "value": f["value"],
                    "contribution": round(f["contribution"], 4),
                    "direction": f["direction"],
                    "set": f["set"],
                }
                for f in factors[:top_n]
            ],
        }

    # ── Optional: a simple horizontal bar chart for the web/report ───────────
    def plot(self, result, out_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        facs = result["top_factors"][::-1]   # smallest at bottom
        labels = [f"{f['label']}" + (f"  ({f['value']})" if f["value"] is not None else "")
                  for f in facs]
        vals = [f["contribution"] for f in facs]
        colors = ["#0369A1" if v >= 0 else "#EF4444" for v in vals]

        fig, ax = plt.subplots(figsize=(9, 0.6 * len(facs) + 1.5))
        ax.barh(range(len(facs)), vals, color=colors, alpha=0.9)
        ax.set_yticks(range(len(facs)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlabel("Contribution to predicted probability")
        ax.set_title(
            f"Top factors — {result['target']}\n"
            f"Predicted P(top researcher) = {result['prediction']:.2f}  "
            f"(baseline {result['baseline']:.2f})",
            fontsize=11, fontweight="bold", color="#0D2137",
        )
        # Legend
        import matplotlib.patches as mpatches
        ax.legend(handles=[
            mpatches.Patch(color="#0369A1", label="increases potential"),
            mpatches.Patch(color="#EF4444", label="decreases potential"),
        ], loc="lower right", fontsize=8.5)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return out_path


# ── Leanest production path: XGBoost native TreeSHAP (no shap dependency) ─────
def explain_native(model, candidate_x, feature_names, top_n=5):
    """
    Same TreeSHAP attribution computed natively by XGBoost (log-odds space).
    Useful for a minimal serving endpoint that doesn't import `shap`.

    Returns the top_n (feature, value, contribution) tuples for ONE set/model.
    For the full ensemble, average across sets exactly as LocalExplainer does.
    """
    import xgboost as xgb
    booster = model.get_booster()
    dm = xgb.DMatrix(np.asarray(candidate_x, dtype=float).reshape(1, -1),
                     feature_names=list(feature_names))
    contribs = booster.predict(dm, pred_contribs=True)[0]   # last entry = bias
    phi = contribs[:-1]
    order = np.argsort(np.abs(phi))[::-1][:top_n]
    return [(feature_names[i], float(phi[i])) for i in order]


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Local per-candidate SHAP for the XGBoost ensemble")
    ap.add_argument("--candidate_csv", required=True,
                    help="CSV of candidates with Set C/D/E columns "
                         "(e.g. the DeepSeek inference dataset)")
    ap.add_argument("--data_csv", default="data/2015-2018_rookie_dataset.csv",
                    help="Training dataset (for the model + SHAP background)")
    ap.add_argument("--target", default="pub_w_top_5pct",
                    choices=["pub_w_top_5pct", "pub_top_5pct"])
    ap.add_argument("--sets", nargs="+", default=["C", "D", "E"],
                    choices=["C", "D", "E"])
    ap.add_argument("--top_n", type=int, default=5)
    ap.add_argument("--row", type=int, default=None,
                    help="Explain only this row index (default: all rows)")
    ap.add_argument("--out", default="local_explanations.json")
    ap.add_argument("--plot", action="store_true",
                    help="Also save a bar chart per explained candidate")
    args = ap.parse_args()

    cand_df = pd.read_csv(args.candidate_csv)
    id_col = "candidate" if "candidate" in cand_df.columns else None

    explainer = LocalExplainer(args.data_csv, target=args.target, sets=tuple(args.sets))

    rows = [args.row] if args.row is not None else range(len(cand_df))
    results = []
    os.makedirs("local_shap_output", exist_ok=True)

    for r in rows:
        row = cand_df.iloc[r]
        res = explainer.explain(row, top_n=args.top_n)
        res["candidate"] = str(row[id_col]) if id_col else f"row_{r}"
        results.append(res)

        print(f"\n=== {res['candidate']} — P({args.target})={res['prediction']:.3f} "
              f"(baseline {res['baseline']:.3f}) ===")
        for f in res["top_factors"]:
            val = "" if f["value"] is None else f"  [{f['value']}]"
            sign = "+" if f["contribution"] >= 0 else "−"
            print(f"  {f['direction']:9s} {f['label']:<38}{val:<10} "
                  f"{sign}{abs(f['contribution']):.4f}")

        if args.plot:
            safe = "".join(ch if ch.isalnum() else "_" for ch in res["candidate"])[:40]
            p = explainer.plot(res, os.path.join("local_shap_output", f"factors_{safe}.png"))
            print(f"  chart → {p}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results if args.row is None else results[0], f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(results)} explanation(s) → {args.out}")


if __name__ == "__main__":
    main()
