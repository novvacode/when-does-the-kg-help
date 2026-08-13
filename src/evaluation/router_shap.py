"""
src/evaluation/router_shap.py — SHAP explainability for the deployed router.

WHY THIS EXISTS
===============
Table VI (the feature ablation in src/evaluation/router_ablation.py) shows
that the 5 structural patient features contribute nothing to routing:
question-only 0.9200 acc / 0.8873 macro-F1 vs full 0.9200 / 0.8692. That is
a *performance-level* result — it says removing the patient features does not
hurt. It does not say, at the level of individual decisions, where the
router's evidence actually comes from.

This script answers that with feature attributions. It runs shap.TreeExplainer
against the DEPLOYED XGBoost router (models/router/router_xgb_model.json) over
its real 389-dim feature vector (384 BGE question-embedding dims + 5 structural
patient features), and reports:

  1. group attribution  — mean |SHAP| summed over the 384 embedding dims vs the
                          5 patient features, per class. This is the
                          attribution-level statement of the Table VI result.
  2. per-decision        — signed top-20 attributions for a sample of decisions.
  3. faithfulness        — a deletion test. Ablate the top-k SHAP-ranked
                          features (those pushing TOWARD the predicted class)
                          and confirm the predicted-class probability falls, as
                          SHAP predicts. Scored against a random-k control,
                          because a flip rate with no null is uninterpretable.

The 2026-08-14 log entry noted that a per-case attribution "would be needed to
confirm" how the router reconciles 98.4% embedding importance with genuine
per-patient behaviour, and that shap was not installed. This is that analysis.

TWO TIERS
---------
  A  router_val (n=100)   — the exact rows Table VI is computed on, so the
                            corroboration is on identical data. All 5
                            structural columns are already in the parquet.
  B  final eval (n=300)   — the DEPLOYED routing decisions the paper reports.
                            per_question_results.csv carries only
                            sparsity_score/sparsity_bucket, so n_labs/n_diag/
                            n_meds are reconstructed via PatientSnapshot
                            exactly as run_evaluation.py does.

Both tiers are gated on reproducing the decision function before any
attribution is computed:
  A  re-predicted probabilities must match models/router/router_predictions.csv
  B  re-predicted modes must match `mode_used` on the Router rows
A tier that fails its gate is reported, not pushed through — a SHAP explanation
of a feature vector the router never actually saw would be worse than none.

NOTHING IS RETRAINED. models/router/ is opened read-only; the generator and
QLoRA adapter are not touched.

Usage:
    python -m src.evaluation.router_shap              # both tiers
    python -m src.evaluation.router_shap --tier A     # one tier
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

MODEL_DIR = Path("models/router")
TRAIN_PARQUET = Path("data/router/router_train_oracle.parquet")
VAL_PARQUET = Path("data/router/router_val_oracle.parquet")
EVAL_RESULTS = Path("experiments/results/final_eval/per_question_results.csv")
OUT_DIR = Path("experiments/results/final_eval")
FIG_DIR = OUT_DIR / "figures"

STRUCT_COLS = ["n_labs", "n_diag", "n_meds", "sparsity_score", "sparsity_bucket"]
N_EMBED_DIMS = 384
SEED = 42

# Deletion test: how many top-ranked features to ablate.
K_VALUES = [1, 2, 5, 10, 20, 50]
N_RANDOM_REPEATS = 5

# Gate tolerances. Tier A compares stored probabilities, so it can be strict.
# Tier B compares discrete modes; anything below this means the reconstructed
# vector is not the one the router saw and the tier is abandoned.
PROB_TOLERANCE = 1e-5
MIN_MODE_AGREEMENT = 0.995


def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
# Loading the deployed decision function
# ══════════════════════════════════════════════════════════════════════════════


def load_router():
    """Load the deployed classifier, label encoder and fitted feature pipeline."""
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_DIR / "router_xgb_model.json")

    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # The pickle carries the FITTED StandardScaler and the SentenceTransformer,
    # so vectors built through it are identical to the deployed ones. If it
    # cannot be unpickled (library drift since 2026-08-10), fall back to
    # rebuilding the same pipeline and refitting the scaler on router_train —
    # deterministic, and the gates below will catch it if it is not equivalent.
    try:
        with open(MODEL_DIR / "feature_pipeline.pkl", "rb") as f:
            pipe = pickle.load(f)
        _log("feature_pipeline.pkl unpickled (deployed scaler + embedder).")
        return clf, le, pipe, "pickle"
    except Exception as e:
        _log(f"[WARN] Could not unpickle feature_pipeline.pkl ({type(e).__name__}: {e}).")
        _log("[WARN] Rebuilding pipeline and refitting scaler on router_train.")
        from src.router.feature_pipeline import HybridFeaturePipeline

        class _Cfg:
            embed_model = "BAAI/bge-small-en-v1.5"
            ehr_feature_cols = STRUCT_COLS
            seed = SEED

        logger = logging.getLogger("router_shap")
        logger.addHandler(logging.NullHandler())
        pipe = HybridFeaturePipeline(_Cfg(), logger)
        pipe.fit_transform(_read_oracle(TRAIN_PARQUET))
        return clf, le, pipe, "rebuilt"


def _read_oracle(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[~df["best_mode"].isin(["FAILED", "FAILED_GENERATION", "MISSING_MODES"])].copy()


def feature_names() -> list[str]:
    """Must match train_router.py's naming so figures line up with existing artifacts."""
    return [f"BGE_Dim_{i}" for i in range(N_EMBED_DIMS)] + STRUCT_COLS


# ══════════════════════════════════════════════════════════════════════════════
# Tier construction + correctness gates
# ══════════════════════════════════════════════════════════════════════════════


def build_tier_a(pipe, clf, le) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """router_val (n=100) — the rows Table VI is computed on."""
    df = _read_oracle(VAL_PARQUET)
    X = pipe.transform(df.copy())
    probs = clf.predict_proba(X)
    pred = le.inverse_transform(np.argmax(probs, axis=1))

    gate = {"type": "stored_probabilities", "n": int(len(df))}
    stored_path = MODEL_DIR / "router_predictions.csv"
    if stored_path.exists():
        stored = pd.read_csv(stored_path)
        prob_cols = [f"prob_{c}" for c in le.classes_]
        if len(stored) == len(df) and all(c in stored.columns for c in prob_cols):
            delta = np.abs(stored[prob_cols].to_numpy() - probs).max()
            mode_match = float((stored["predicted_mode"].to_numpy() == pred).mean())
            gate.update({
                "max_abs_prob_delta": float(delta),
                "mode_agreement": mode_match,
                "passed": bool(delta < PROB_TOLERANCE and mode_match >= MIN_MODE_AGREEMENT),
            })
        else:
            gate.update({"passed": False, "reason": "router_predictions.csv shape/columns mismatch"})
    else:
        gate.update({"passed": False, "reason": "router_predictions.csv absent"})

    meta = df[["question", "best_mode"]].copy()
    meta["predicted_mode"] = pred
    meta["confidence"] = probs.max(axis=1)
    if "question_type" in df.columns:
        meta["question_type"] = df["question_type"].to_numpy()
    return X, meta.reset_index(drop=True), gate


def build_tier_b(pipe, clf, le) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Final held-out eval (n=300) — the deployed decisions the paper reports."""
    from src.lakehouse.patient_snapshot import PatientSnapshot

    res = pd.read_csv(EVAL_RESULTS)
    router_rows = res[res["system"].str.lower() == "router"].copy()
    if router_rows.empty:
        return None, None, {"passed": False, "reason": "no Router rows in per_question_results.csv"}
    router_rows = router_rows.sort_values("q_idx").reset_index(drop=True)

    # Reconstruct the 5 structural features exactly as run_evaluation.py does
    # (see its get_struct_features): counts off PatientSnapshot, not sparsity.parquet.
    snap_api = PatientSnapshot()
    cache: dict[int, dict] = {}

    def struct(hadm_id: int) -> dict:
        if hadm_id not in cache:
            try:
                s = snap_api.get(hadm_id)
                cache[hadm_id] = {
                    "n_labs": float(len(s.get("labs", []))),
                    "n_diag": float(len(s.get("diagnoses", []))),
                    "n_meds": float(len(s.get("medications", []))),
                    "sparsity_score": float(s.get("sparsity_score") or 0.0),
                    "sparsity_bucket": str(s.get("sparsity_bucket") or "unknown"),
                }
            except Exception as e:
                _log(f"[WARN] PatientSnapshot failed for hadm_id={hadm_id}: {e}")
                cache[hadm_id] = {"n_labs": 0.0, "n_diag": 0.0, "n_meds": 0.0,
                                  "sparsity_score": 0.0, "sparsity_bucket": "unknown"}
        return cache[hadm_id]

    _log(f"Reconstructing structural features for {router_rows['hadm_id'].nunique()} admissions...")
    feat_rows = []
    for _, r in router_rows.iterrows():
        s = struct(int(r["hadm_id"]))
        feat_rows.append({"question": r["question"], **s})
    X = pipe.transform(pd.DataFrame(feat_rows))

    probs = clf.predict_proba(X)
    pred = le.inverse_transform(np.argmax(probs, axis=1))

    # THE GATE: does the reconstructed vector reproduce the routing decision
    # that was actually taken and reported?
    agreement = float((pred == router_rows["mode_used"].to_numpy()).mean())
    gate = {
        "type": "mode_used_agreement",
        "n": int(len(router_rows)),
        "mode_agreement": agreement,
        "n_disagreements": int((pred != router_rows["mode_used"].to_numpy()).sum()),
        "passed": bool(agreement >= MIN_MODE_AGREEMENT),
    }

    meta = router_rows[["q_idx", "hadm_id", "question", "question_type",
                        "sparsity_bucket", "mode_used"]].copy()
    meta["predicted_mode"] = pred
    meta["confidence"] = probs.max(axis=1)
    return X, meta.reset_index(drop=True), gate


# ══════════════════════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════════════════════


def compute_shap(clf, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (shap_values [n, n_feat, n_class], base_values [n_class], additivity_err)."""
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # older shap returns a list per class
        sv = np.stack(sv, axis=-1)
    sv = np.asarray(sv)

    base = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))

    # Additivity: base + sum(phi) must equal the raw margin. If this does not
    # hold the attributions are not explaining this model.
    margin = clf.predict(X, output_margin=True)
    recon = base[None, :] + sv.sum(axis=1)
    err = float(np.abs(recon - margin).max())
    return sv, base, err


def group_attribution(sv: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """mean |SHAP| summed over the embedding block vs the patient block, per class."""
    rows = []
    for ci, cname in enumerate(classes):
        m = np.abs(sv[:, :, ci]).mean(axis=0)      # mean |phi| per feature
        emb, pat = m[:N_EMBED_DIMS].sum(), m[N_EMBED_DIMS:].sum()
        total = emb + pat
        rows.append({
            "class": cname,
            "embedding_sum_mean_abs_shap": round(float(emb), 6),
            "patient_sum_mean_abs_shap": round(float(pat), 6),
            "embedding_share": round(float(emb / total), 6) if total else np.nan,
            "patient_share": round(float(pat / total), 6) if total else np.nan,
            **{f"mean_abs_shap_{c}": round(float(v), 6)
               for c, v in zip(STRUCT_COLS, m[N_EMBED_DIMS:])},
        })

    m_all = np.abs(sv).mean(axis=0).mean(axis=1)   # pooled over classes
    emb, pat = m_all[:N_EMBED_DIMS].sum(), m_all[N_EMBED_DIMS:].sum()
    total = emb + pat
    rows.append({
        "class": "POOLED",
        "embedding_sum_mean_abs_shap": round(float(emb), 6),
        "patient_sum_mean_abs_shap": round(float(pat), 6),
        "embedding_share": round(float(emb / total), 6) if total else np.nan,
        "patient_share": round(float(pat / total), 6) if total else np.nan,
        **{f"mean_abs_shap_{c}": round(float(v), 6)
           for c, v in zip(STRUCT_COLS, m_all[N_EMBED_DIMS:])},
    })
    return pd.DataFrame(rows)


def feature_ranking(sv: np.ndarray, top_n: int = 25) -> pd.DataFrame:
    """
    Rank all 389 features by pooled mean |SHAP|.

    The block comparison above sums 384 embedding dims against 5 patient
    features, which flatters the block that simply has more members. This
    ranking is the per-feature view: it says where each patient feature sits
    among all 389, and how it compares to the AVERAGE embedding dim. Both
    numbers are needed — they answer different questions and, on this router,
    they do not say the same thing.
    """
    m = np.abs(sv).mean(axis=0).mean(axis=1)          # pooled over rows and classes
    names = feature_names()
    df = pd.DataFrame({
        "feature": names,
        "block": ["patient" if i >= N_EMBED_DIMS else "embedding" for i in range(len(names))],
        "mean_abs_shap": m,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["n_nonzero_rows"] = [int((np.abs(sv[:, names.index(f), :]).sum(axis=1) > 1e-9).sum())
                            for f in df["feature"]]
    # Keep the head plus every patient feature, wherever it landed.
    keep = df.head(top_n).copy()
    pat = df[df["block"] == "patient"]
    out = pd.concat([keep, pat[~pat["feature"].isin(keep["feature"])]], ignore_index=True)
    out["mean_abs_shap"] = out["mean_abs_shap"].round(6)
    return out.sort_values("rank").reset_index(drop=True)


def per_decision_table(sv, X, meta, clf, le, n_sample: int, top_n: int = 20) -> pd.DataFrame:
    """Signed top-N attributions toward the predicted class for a sample of decisions."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    idx.sort()
    names = feature_names()
    pred_idx = np.argmax(clf.predict_proba(X), axis=1)

    out = []
    for i in idx:
        c = pred_idx[i]
        phi = sv[i, :, c]
        order = np.argsort(-np.abs(phi))[:top_n]
        for rank, f in enumerate(order, 1):
            row = {
                "row": int(i),
                "predicted_mode": le.classes_[c],
                "rank": rank,
                "feature": names[f],
                "feature_block": "patient" if f >= N_EMBED_DIMS else "embedding",
                "shap_value": round(float(phi[f]), 6),
                "feature_value": round(float(X[i, f]), 6),
            }
            for col in ("question", "question_type", "best_mode", "mode_used"):
                if col in meta.columns:
                    row[col] = meta.iloc[i][col]
            out.append(row)
    return pd.DataFrame(out)


# ══════════════════════════════════════════════════════════════════════════════
# Faithfulness (deletion test)
# ══════════════════════════════════════════════════════════════════════════════


def faithfulness(clf, X, sv, baseline: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ablate the top-k features SHAP says push TOWARD the predicted class and check
    the predicted-class probability falls. Scored against a random-k control.

    Ablation = replace the feature with its training-set mean. For dense BGE
    dimensions this is an off-manifold perturbation, so this measures the
    model's local sensitivity to the features SHAP credits — not a real-world
    intervention. That is the standard limitation of deletion-based
    faithfulness and is stated with the result.
    """
    rng = np.random.default_rng(SEED)
    probs0 = clf.predict_proba(X)
    pred = np.argmax(probs0, axis=1)
    p_orig = probs0[np.arange(len(X)), pred]

    # Rank features by signed contribution toward each row's predicted class.
    contrib = np.take_along_axis(sv, pred[:, None, None], axis=2)[:, :, 0]
    order = np.argsort(-contrib, axis=1)   # most positive first
    n_pos = (contrib > 0).sum(axis=1)

    rows, per_row = [], []
    for k in K_VALUES:
        Xa = X.copy()
        for i in range(len(X)):
            take = min(k, int(n_pos[i]))     # only ablate genuine positive contributors
            if take:
                Xa[i, order[i, :take]] = baseline[order[i, :take]]
        pa = clf.predict_proba(Xa)
        p_top = pa[np.arange(len(X)), pred]
        flip_top = (np.argmax(pa, axis=1) != pred)
        drop_top = p_orig - p_top

        # Random-k control: same budget, features chosen at random.
        drops_r, flips_r = [], []
        for _ in range(N_RANDOM_REPEATS):
            Xr = X.copy()
            for i in range(len(X)):
                take = min(k, int(n_pos[i]))
                if take:
                    sel = rng.choice(X.shape[1], size=take, replace=False)
                    Xr[i, sel] = baseline[sel]
            pr = clf.predict_proba(Xr)
            drops_r.append(p_orig - pr[np.arange(len(X)), pred])
            flips_r.append(np.argmax(pr, axis=1) != pred)
        drop_rand = np.mean(drops_r, axis=0)
        flip_rand = np.mean(flips_r, axis=0)

        rows.append({
            "k": k,
            "directional_agreement_top_k": round(float((drop_top > 0).mean()), 4),
            "flip_rate_top_k": round(float(flip_top.mean()), 4),
            "mean_prob_drop_top_k": round(float(drop_top.mean()), 4),
            "directional_agreement_random_k": round(float((drop_rand > 0).mean()), 4),
            "flip_rate_random_k": round(float(flip_rand.mean()), 4),
            "mean_prob_drop_random_k": round(float(drop_rand.mean()), 4),
            "drop_ratio_top_over_random": (
                round(float(drop_top.mean() / drop_rand.mean()), 3)
                if drop_rand.mean() > 1e-12 else np.nan
            ),
        })
        for i in range(len(X)):
            per_row.append({"row": int(i), "k": k,
                            "prob_drop_top_k": round(float(drop_top[i]), 6),
                            "flipped_top_k": bool(flip_top[i]),
                            "prob_drop_random_k": round(float(drop_rand[i]), 6)})

    df = pd.DataFrame(rows)
    aopc_top = float(df["mean_prob_drop_top_k"].mean())
    aopc_rand = float(df["mean_prob_drop_random_k"].mean())
    summary = pd.DataFrame([{
        "aopc_top_k": round(aopc_top, 4),
        "aopc_random_k": round(aopc_rand, 4),
        "aopc_ratio": round(aopc_top / aopc_rand, 3) if aopc_rand > 1e-12 else np.nan,
        "faithfulness_score_directional": round(float(df["directional_agreement_top_k"].mean()), 4),
        "mean_flip_rate_top_k": round(float(df["flip_rate_top_k"].mean()), 4),
        "mean_flip_rate_random_k": round(float(df["flip_rate_random_k"].mean()), 4),
    }])
    return df, summary, pd.DataFrame(per_row)


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════


def plot_beeswarm(sv, X, classes, tier: str) -> list[str]:
    written = []
    names = feature_names()
    for ci, cname in enumerate(classes):
        safe = cname.replace("+", "").replace(" ", "")
        path = FIG_DIR / f"shap_beeswarm_tier{tier}_{safe}.png"
        plt.figure()
        shap.summary_plot(sv[:, :, ci], X, feature_names=names,
                          max_display=20, show=False)
        plt.title(f"SHAP attributions — class {cname} (tier {tier})")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        written.append(str(path))
    return written


def plot_faithfulness(df: pd.DataFrame, tier: str) -> str:
    path = FIG_DIR / f"shap_faithfulness_curve_tier{tier}.png"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df["k"], df["mean_prob_drop_top_k"], "o-", label="top-k (SHAP)")
    ax[0].plot(df["k"], df["mean_prob_drop_random_k"], "s--", label="random-k (control)")
    ax[0].set_xlabel("k features ablated"); ax[0].set_ylabel("mean drop in P(predicted class)")
    ax[0].set_title("Deletion curve"); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(df["k"], df["flip_rate_top_k"], "o-", label="top-k (SHAP)")
    ax[1].plot(df["k"], df["flip_rate_random_k"], "s--", label="random-k (control)")
    ax[1].set_xlabel("k features ablated"); ax[1].set_ylabel("decision flip rate")
    ax[1].set_title("Routing decision flips"); ax[1].legend(); ax[1].grid(alpha=.3)

    plt.suptitle(f"Router SHAP faithfulness (tier {tier})")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(path)


# ══════════════════════════════════════════════════════════════════════════════
# Tier driver
# ══════════════════════════════════════════════════════════════════════════════


def run_tier(tier: str, clf, le, pipe, X_train_mean: np.ndarray) -> dict:
    print("\n" + "=" * 78)
    print(f"TIER {tier}")
    print("=" * 78)

    X, meta, gate = (build_tier_a(pipe, clf, le) if tier == "A"
                     else build_tier_b(pipe, clf, le))

    print(f"[GATE] {json.dumps(gate)}")
    if not gate.get("passed"):
        print(f"[STOP] Tier {tier} correctness gate FAILED — no attributions computed.")
        return {"tier": tier, "gate": gate, "status": "gate_failed"}
    if X is None:
        return {"tier": tier, "gate": gate, "status": "no_data"}

    classes = list(le.classes_)
    _log(f"Computing SHAP over {X.shape[0]} decisions x {X.shape[1]} features...")
    sv, base, add_err = compute_shap(clf, X)
    _log(f"Additivity check (base + sum(phi) vs raw margin): max abs err {add_err:.2e}")

    ga = group_attribution(sv, classes)
    ga.insert(0, "tier", tier)
    print("\n--- GROUP ATTRIBUTION (mean |SHAP|, summed within block) ---")
    print(ga[["class", "embedding_sum_mean_abs_shap", "patient_sum_mean_abs_shap",
              "embedding_share", "patient_share"]].to_string(index=False))

    n_emb_dim = ga.loc[ga["class"] == "POOLED", "embedding_sum_mean_abs_shap"].iloc[0] / N_EMBED_DIMS
    n_pat_dim = ga.loc[ga["class"] == "POOLED", "patient_sum_mean_abs_shap"].iloc[0] / len(STRUCT_COLS)
    print(f"\n  per-feature (pooled): avg embedding dim {n_emb_dim:.6f} | "
          f"avg patient feature {n_pat_dim:.6f} | ratio {n_pat_dim / n_emb_dim:.2f}x")

    rank = feature_ranking(sv)
    rank.insert(0, "tier", tier)
    print("\n--- WHERE THE 5 PATIENT FEATURES RANK AMONG ALL 389 ---")
    print(rank[rank["block"] == "patient"][
        ["feature", "rank", "mean_abs_shap", "n_nonzero_rows"]].to_string(index=False))

    pdt = per_decision_table(sv, X, meta, clf, le, n_sample=15)
    pdt.insert(0, "tier", tier)

    _log("Running deletion-based faithfulness test (+ random-k control)...")
    faith, faith_summary, faith_rows = faithfulness(clf, X, sv, X_train_mean)
    faith.insert(0, "tier", tier)
    faith_summary.insert(0, "tier", tier)
    print("\n--- FAITHFULNESS (deletion test) ---")
    print(faith.drop(columns=["tier"]).to_string(index=False))
    print("\n--- FAITHFULNESS SUMMARY ---")
    print(faith_summary.drop(columns=["tier"]).to_string(index=False))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = plot_beeswarm(sv, X, classes, tier) + [plot_faithfulness(faith, tier)]

    return {
        "tier": tier, "status": "ok", "gate": gate,
        "n_decisions": int(X.shape[0]), "n_features": int(X.shape[1]),
        "shap_additivity_max_abs_error": add_err,
        "base_values": {c: float(b) for c, b in zip(classes, base)} if base.size == len(classes) else base.tolist(),
        "figures": figs,
        "per_feature_pooled": {"avg_embedding_dim": round(float(n_emb_dim), 6),
                               "avg_patient_feature": round(float(n_pat_dim), 6),
                               "ratio_patient_over_embedding": round(float(n_pat_dim / n_emb_dim), 3)},
        "_tables": {"group": ga, "ranking": rank, "per_decision": pdt,
                    "faithfulness": faith, "faithfulness_summary": faith_summary,
                    "faithfulness_per_row": faith_rows, "meta": meta},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SHAP explainability for the deployed router")
    ap.add_argument("--tier", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clf, le, pipe, pipe_source = load_router()
    _log(f"Classes: {list(le.classes_)}")

    # Reference distribution for ablation = the router's own training data.
    X_train = pipe.transform(_read_oracle(TRAIN_PARQUET).copy())
    X_train_mean = X_train.mean(axis=0)

    tiers = ["A", "B"] if args.tier == "both" else [args.tier]
    results = []
    for t in tiers:
        try:
            r = run_tier(t, clf, le, pipe, X_train_mean)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {"tier": t, "status": "error", "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        # Tier B is only attempted if Tier A landed clean.
        if t == "A" and r.get("status") != "ok" and args.tier == "both":
            print("\n[STOP] Tier A did not land clean — not proceeding to Tier B.")
            break

    # Concatenate tables across tiers and write once.
    def _cat(key: str) -> pd.DataFrame:
        parts = [r["_tables"][key] for r in results if r.get("status") == "ok"]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    written = {}
    for key, fname in [("group", "shap_group_attribution.csv"),
                       ("ranking", "shap_feature_ranking.csv"),
                       ("per_decision", "shap_per_decision_sample.csv"),
                       ("faithfulness", "shap_faithfulness.csv"),
                       ("faithfulness_summary", "shap_faithfulness_summary.csv"),
                       ("faithfulness_per_row", "shap_faithfulness_per_row.csv")]:
        df = _cat(key)
        if not df.empty:
            df.to_csv(OUT_DIR / fname, index=False)
            written[fname] = int(len(df))

    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "src/evaluation/router_shap.py",
        "seed": SEED,
        "versions": {"shap": shap.__version__, "xgboost": xgb.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "python": sys.version.split()[0]},
        "model": {
            "path": str(MODEL_DIR / "router_xgb_model.json"),
            "sha256_16": _sha256(MODEL_DIR / "router_xgb_model.json"),
            "feature_pipeline_source": pipe_source,
        },
        "config": {"k_values": K_VALUES, "n_random_repeats": N_RANDOM_REPEATS,
                   "n_embed_dims": N_EMBED_DIMS, "struct_cols": STRUCT_COLS,
                   "ablation_baseline": "router_train feature mean"},
        "tiers": [{k: v for k, v in r.items() if k != "_tables"} for r in results],
        "outputs": written,
    }
    with open(OUT_DIR / "shap_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print("WRITTEN")
    print("=" * 78)
    for k, v in written.items():
        print(f"  {OUT_DIR / k}  ({v} rows)")
    print(f"  {OUT_DIR / 'shap_metadata.json'}")


if __name__ == "__main__":
    main()
