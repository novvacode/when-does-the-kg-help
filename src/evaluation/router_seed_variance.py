"""
src/evaluation/router_seed_variance.py — how stable is the router's 0.9200?

WHY THIS EXISTS
===============
The paper reports router accuracy 0.9200 and macro-F1 0.8692 from a single
training run with a single seed, and Limitations concedes "a single random
seed". This quantifies two DIFFERENT uncertainties that are easy to conflate:

  1. TRAINING STOCHASTICITY --- retrain on identical data across seeds. Answers
     "would a different run have given a different number?"
  2. EVALUATION UNCERTAINTY --- bootstrap the 100 validation rows for the
     DEPLOYED model. Answers "how precisely do we know 0.9200 at all?"

These are not interchangeable, and on this router the second dominates. Both
are reported; neither is presented as the other.

WHAT THIS DOES NOT MEASURE
--------------------------
Only the final stage varies. The QLoRA fine-tune, the oracle labels, and the
train/validation partition are all held fixed --- re-running those costs hours
of GPU time on a machine with a documented crash history. So this is variance
in router training, NOT end-to-end pipeline variance, and must be reported that
way. A second base model (Phi-3-mini) is deferred for the same reason and
remains a stated limitation.

Deployed hyperparameters are reused exactly (max_depth 4, lr 0.15, 300 trees,
subsample 1.0, colsample_bytree 0.8, gamma 0.3, reg_alpha 0.5, reg_lambda 5.0),
so differences between rows reflect the SEED and nothing else. Note that
subsample=1.0 disables row subsampling, leaving column subsampling as the only
source of randomness --- so a small spread here is partly a property of the
deployed configuration, not evidence of unusual stability.

NOTHING IS OVERWRITTEN. models/router/ is read-only here; retrained models are
discarded after scoring, exactly as router_ablation.py does.

Usage:
    python -m src.evaluation.router_seed_variance
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

TRAIN = Path("data/router/router_train_oracle.parquet")
VAL = Path("data/router/router_val_oracle.parquet")
DEPLOYED_PREDS = Path("models/router/router_predictions.csv")
OUT_DIR = Path("experiments/results/final_eval")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
STRUCT_COLS = ["n_labs", "n_diag", "n_meds", "sparsity_score", "sparsity_bucket"]
BUCKET_MAP = {"low": 0, "medium": 1, "high": 2, "unknown": -1}

SEEDS = list(range(42, 52))          # 10 seeds; the deployed model used 42
N_BOOT = 10_000
BOOT_SEED = 42

XGB_PARAMS = dict(objective="multi:softprob", eval_metric="mlogloss",
                  max_depth=4, learning_rate=0.15, n_estimators=300,
                  subsample=1.0, colsample_bytree=0.8, gamma=0.3,
                  min_child_weight=1, reg_alpha=0.5, reg_lambda=5.0)


def _clean(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    return df[~df.best_mode.isin(["FAILED", "FAILED_GENERATION", "MISSING_MODES"])].copy()


def _structural(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for c in STRUCT_COLS:
        if c == "sparsity_bucket":
            cols.append(df[c].map(BUCKET_MAP).fillna(-1).to_numpy())
        else:
            cols.append(pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0).to_numpy())
    return np.vstack(cols).T.astype(np.float32)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    tr, va = _clean(TRAIN), _clean(VAL)
    print(f"[INFO] train={len(tr)}  val={len(va)}")

    le = LabelEncoder()
    y_tr, y_va = le.fit_transform(tr.best_mode), le.transform(va.best_mode)
    classes = list(le.classes_)

    print(f"[INFO] encoding questions with {EMBED_MODEL} (CPU)...")
    emb = SentenceTransformer(EMBED_MODEL, device="cpu")
    E_tr = emb.encode(tr.question.tolist(), show_progress_bar=False)
    E_va = emb.encode(va.question.tolist(), show_progress_bar=False)
    sc = StandardScaler()
    X_tr = np.hstack([E_tr, sc.fit_transform(_structural(tr))])
    X_va = np.hstack([E_va, sc.transform(_structural(va))])

    # ── 1. training stochasticity ────────────────────────────────────────────
    rows = []
    w = compute_sample_weight("balanced", y_tr)
    for s in SEEDS:
        clf = xgb.XGBClassifier(random_state=s, **XGB_PARAMS)
        clf.fit(X_tr, y_tr, sample_weight=w)
        pred = clf.predict(X_va)
        per = f1_score(y_va, pred, average=None, zero_division=0)
        rows.append({"seed": s,
                     "accuracy": accuracy_score(y_va, pred),
                     "macro_f1": f1_score(y_va, pred, average="macro", zero_division=0),
                     "balanced_acc": balanced_accuracy_score(y_va, pred),
                     **{f"f1_{c}": v for c, v in zip(classes, per)}})
        print(f"       seed {s}: acc {rows[-1]['accuracy']:.4f}  "
              f"macro-F1 {rows[-1]['macro_f1']:.4f}")
    seeds_df = pd.DataFrame(rows)

    # ── 2. evaluation uncertainty on the DEPLOYED model ──────────────────────
    boot = {}
    if DEPLOYED_PREDS.exists():
        dp = pd.read_csv(DEPLOYED_PREDS)
        yt, yp = dp.best_mode.to_numpy(), dp.predicted_mode.to_numpy()
        rng = np.random.default_rng(BOOT_SEED)
        acc, mf1 = [], []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(yt), len(yt))
            acc.append(accuracy_score(yt[idx], yp[idx]))
            mf1.append(f1_score(yt[idx], yp[idx], average="macro", zero_division=0))
        boot = {
            "n_val_rows": int(len(yt)),
            "accuracy_point": float(accuracy_score(yt, yp)),
            "accuracy_ci95": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
            "macro_f1_point": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "macro_f1_ci95": [float(np.percentile(mf1, 2.5)), float(np.percentile(mf1, 97.5))],
        }

    # ── report ───────────────────────────────────────────────────────────────
    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print(f"1. TRAINING STOCHASTICITY — {len(SEEDS)} seeds, identical data")
    print("=" * 78)
    print(seeds_df.round(4).to_string(index=False))
    summ = seeds_df.drop(columns="seed").agg(["mean", "std", "min", "max"]).round(4)
    print("\n" + summ.to_string())
    print(f"\n  accuracy  {summ.loc['mean','accuracy']:.4f} ± {summ.loc['std','accuracy']:.4f}"
          f"   (range {summ.loc['min','accuracy']:.4f}–{summ.loc['max','accuracy']:.4f})")
    print(f"  macro-F1  {summ.loc['mean','macro_f1']:.4f} ± {summ.loc['std','macro_f1']:.4f}"
          f"   (range {summ.loc['min','macro_f1']:.4f}–{summ.loc['max','macro_f1']:.4f})")

    if boot:
        print("\n" + "=" * 78)
        print(f"2. EVALUATION UNCERTAINTY — deployed model, {N_BOOT:,} bootstrap "
              f"resamples of {boot['n_val_rows']} validation rows")
        print("=" * 78)
        print(f"  accuracy  {boot['accuracy_point']:.4f}  95% CI "
              f"[{boot['accuracy_ci95'][0]:.4f}, {boot['accuracy_ci95'][1]:.4f}]")
        print(f"  macro-F1  {boot['macro_f1_point']:.4f}  95% CI "
              f"[{boot['macro_f1_ci95'][0]:.4f}, {boot['macro_f1_ci95'][1]:.4f}]")
        acc_w = boot["accuracy_ci95"][1] - boot["accuracy_ci95"][0]
        seed_w = 4 * summ.loc["std", "accuracy"]        # ~95% span under normality
        print(f"\n  CI width from the 100-row validation set : {acc_w:.4f}")
        print(f"  ~95% span from training seed             : {seed_w:.4f}")
        if seed_w > 0:
            print(f"  ratio                                    : {acc_w / seed_w:.1f}x")
        print("\n  The validation set, not the seed, dominates. Reporting seed")
        print("  variance alone would understate how uncertain 0.9200 actually is.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds_df.to_csv(OUT_DIR / "router_seed_variance.csv", index=False)
    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "scope": "router training stage only; QLoRA fine-tune, oracle labels and "
                 "the train/val partition are held fixed",
        "deferred": "second base model (Phi-3-mini) not attempted — GPU cost and "
                    "documented crash history; remains a stated limitation",
        "seeds": SEEDS,
        "hyperparameters": XGB_PARAMS,
        "note_on_stochasticity": "subsample=1.0 disables row subsampling, so column "
                                 "subsampling is the only randomness; a small spread "
                                 "is partly a property of this configuration",
        "training_stochasticity": {
            m: {"mean": float(summ.loc["mean", m]), "std": float(summ.loc["std", m]),
                "min": float(summ.loc["min", m]), "max": float(summ.loc["max", m])}
            for m in ["accuracy", "macro_f1", "balanced_acc"]},
        "evaluation_uncertainty": boot,
    }
    (OUT_DIR / "router_seed_variance.json").write_text(json.dumps(out, indent=2),
                                                       encoding="utf-8")
    print(f"\n[INFO] Wrote router_seed_variance.csv and .json to {OUT_DIR}")
    print("[INFO] models/router/ was not modified.")


if __name__ == "__main__":
    main()
