"""
src/evaluation/router_ablation.py — feature ablation for the adaptive router.

WHY THIS EXISTS
===============
The router beats a question-TEXT lookup baseline on held-out data, which
implies it uses signal beyond question identity. But XGBoost's aggregate
gain attributes only ~1.6% of importance to the structural patient features,
which superficially contradicts that. The 2026-08-14 audit reconciled the
two by hand (n_labs perfectly separates T from T+E on lab questions), but a
hand-picked example is weaker evidence than an ablation.

This retrains the router on explicit feature subsets and reports what each
group contributes:

    question_only    384 BGE dims             — can only learn question -> mode
    patient_only     5 structural features    — no question information at all
    full             both (the deployed model)

If `full` > `question_only`, patient features carry information the question
alone does not. That is the direct quantitative test of the H1 mechanism.

Nothing is overwritten: the deployed router in models/router/ is untouched.
Results go to experiments/results/final_eval/router_ablation.csv.

Usage:
    python -m src.evaluation.router_ablation
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sentence_transformers import SentenceTransformer

TRAIN = Path("data/router/router_train_oracle.parquet")
VAL   = Path("data/router/router_val_oracle.parquet")
OUT   = Path("experiments/results/final_eval/router_ablation.csv")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
STRUCT_COLS = ["n_labs", "n_diag", "n_meds", "sparsity_score", "sparsity_bucket"]
BUCKET_MAP  = {"low": 0, "medium": 1, "high": 2, "unknown": -1}
SEED = 42

# Same hyperparameters the deployed router was tuned to, so differences
# between rows reflect FEATURES rather than a different search landing spot.
XGB_PARAMS = dict(objective="multi:softprob", random_state=SEED,
                  eval_metric="mlogloss", subsample=1.0, reg_lambda=5.0,
                  reg_alpha=0.5, n_estimators=300, min_child_weight=1,
                  max_depth=4, learning_rate=0.15, gamma=0.3,
                  colsample_bytree=0.8)


def _clean(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    return df[~df["best_mode"].isin(["FAILED", "FAILED_GENERATION", "MISSING_MODES"])].copy()


def _structural(df: pd.DataFrame) -> np.ndarray:
    out = []
    for c in STRUCT_COLS:
        if c == "sparsity_bucket":
            out.append(df[c].map(BUCKET_MAP).fillna(-1).to_numpy())
        else:
            out.append(pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0).to_numpy())
    return np.vstack(out).T.astype(np.float32)


def main() -> None:
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    tr, va = _clean(TRAIN), _clean(VAL)
    print(f"[INFO] train={len(tr)}  val={len(va)}")
    print(f"[INFO] val class support: {va['best_mode'].value_counts().to_dict()}")

    le = LabelEncoder()
    y_tr = le.fit_transform(tr["best_mode"])
    y_va = le.transform(va["best_mode"])

    print(f"[INFO] Encoding questions with {EMBED_MODEL} (CPU)...")
    emb = SentenceTransformer(EMBED_MODEL, device="cpu")
    E_tr = emb.encode(tr["question"].tolist(), show_progress_bar=False)
    E_va = emb.encode(va["question"].tolist(), show_progress_bar=False)

    scaler = StandardScaler()
    S_tr = scaler.fit_transform(_structural(tr))
    S_va = scaler.transform(_structural(va))

    variants = {
        "question_only (384 BGE dims)": (E_tr, E_va),
        "patient_only (5 structural)":  (S_tr, S_va),
        "full (BGE + structural)":      (np.hstack([E_tr, S_tr]), np.hstack([E_va, S_va])),
    }

    # Reference points that use no learning at all.
    maj = tr["best_mode"].value_counts().index[0]
    rows = [{
        "variant": "majority-class baseline", "n_features": 0,
        "accuracy": round(accuracy_score(va["best_mode"], [maj] * len(va)), 4),
        "macro_f1": round(f1_score(va["best_mode"], [maj] * len(va),
                                    average="macro", zero_division=0), 4),
        "balanced_acc": round(balanced_accuracy_score(va["best_mode"], [maj] * len(va)), 4),
        "f1_T": np.nan, "f1_T+E": np.nan, "f1_T+E+K": np.nan,
    }]

    for name, (Xtr, Xva) in variants.items():
        clf = xgb.XGBClassifier(**XGB_PARAMS)
        clf.fit(Xtr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
        pred = le.inverse_transform(clf.predict(Xva))
        per = f1_score(va["best_mode"], pred, average=None,
                       labels=list(le.classes_), zero_division=0)
        rows.append({
            "variant": name,
            "n_features": Xtr.shape[1],
            "accuracy": round(accuracy_score(va["best_mode"], pred), 4),
            "macro_f1": round(f1_score(va["best_mode"], pred, average="macro",
                                        zero_division=0), 4),
            "balanced_acc": round(balanced_accuracy_score(va["best_mode"], pred), 4),
            **{f"f1_{c}": round(v, 4) for c, v in zip(le.classes_, per)},
        })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("\n" + "=" * 88)
    print("ROUTER FEATURE ABLATION (val, n=%d)" % len(va))
    print("=" * 88)
    print(res.to_string(index=False))

    q = res[res.variant.str.startswith("question_only")].iloc[0]
    f = res[res.variant.str.startswith("full")].iloc[0]
    p = res[res.variant.str.startswith("patient_only")].iloc[0]
    print("\n  CONTRIBUTION OF PATIENT FEATURES (full - question_only):")
    print(f"    accuracy     {f['accuracy'] - q['accuracy']:+.4f}")
    print(f"    macro-F1     {f['macro_f1'] - q['macro_f1']:+.4f}")
    print(f"    T+E+K F1     {f['f1_T+E+K'] - q['f1_T+E+K']:+.4f}")
    print("\n  Patient features ALONE (no question text at all):")
    print(f"    accuracy {p['accuracy']:.4f}  macro-F1 {p['macro_f1']:.4f}")
    print(f"\n[INFO] Saved: {OUT}")


if __name__ == "__main__":
    main()
