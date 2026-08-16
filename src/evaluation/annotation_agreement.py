"""
src/evaluation/annotation_agreement.py — human vs. automatic detector agreement.

WHY THIS EXISTS
===============
Every hallucination number in the paper comes from two automatic heuristics
(`ehr_contradiction_score`, `unsupported_score`). The Limitations section
concedes they are unvalidated. This scores them against human judgement on the
75-row stratified sample built by build_annotation_sample.py, and reports not
just whether they agree but WHICH WAY they fail — a detector that over-flags
and one that under-flags are different problems for the paper's claims.

Reports per check:
  · Cohen's kappa + bootstrap 95% CI (10,000 resamples, seed 42 — same
    convention as src/evaluation/analysis.py)
  · observed agreement, and both marginals (prevalence)
  · 2x2 confusion matrix, human as reference
  · McNemar's exact test on the discordant cells — the direct test of
    false-positive vs false-negative skew

THE THRESHOLD IS PRE-REGISTERED
-------------------------------
`unsupported_score` is continuous, so it needs a cut to compare against a
yes/no human label. That cut was fixed at >= 0.5 on 2026-08-13 BEFORE any
annotation, precisely so it could not be tuned against the human labels
afterwards. The 0.1-0.9 sweep this script prints is SENSITIVITY ANALYSIS
ONLY. Do not adopt whichever threshold maximises kappa — that would be
fitting the metric to the reference standard it is being judged against, and
it would invalidate the whole exercise. Report the 0.5 row as the result and
the sweep as robustness.

EHR-contradiction is scored only on rows where it is defined: mode T rows are
n/a by the standing rule (the detector scans an EHR snapshot mode T never
receives — RESEARCH_LOG 2026-08-15), so they are excluded, not counted as
agreement.

Usage:
    python -m src.evaluation.annotation_agreement
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

ANNOT_DIR = Path("experiments/results/annotation")
FILLED = ANNOT_DIR / "annotations_filled.csv"
SAMPLE = ANNOT_DIR / "sample_75.csv"
META = ANNOT_DIR / "sample_metadata.json"

PREREG_THRESHOLD = 0.5          # LOCKED 2026-08-13. Do not retune.
SWEEP = [round(x, 1) for x in np.arange(0.1, 0.95, 0.1)]
N_BOOT = 10_000
SEED = 42


def _kappa(h: np.ndarray, d: np.ndarray) -> float:
    """Cohen's kappa, nan when a resample is degenerate (one cell holds all mass)."""
    if len(set(h)) == 1 and len(set(d)) == 1:
        return 1.0 if h[0] == d[0] else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        k = cohen_kappa_score(h, d, labels=["no", "yes"])
    return float(k)


def bootstrap_kappa(h: np.ndarray, d: np.ndarray) -> tuple[float, float, int]:
    """Percentile 95% CI over paired row resamples. Returns (lo, hi, n_degenerate)."""
    rng = np.random.default_rng(SEED)
    n = len(h)
    vals, degen = [], 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        k = _kappa(h[idx], d[idx])
        if np.isnan(k):
            degen += 1
        else:
            vals.append(k)
    if not vals:
        return float("nan"), float("nan"), degen
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), degen


def analyse(name: str, h: pd.Series, d: pd.Series) -> dict:
    """Full agreement analysis for one binary check."""
    h = h.to_numpy()
    d = d.to_numpy()
    n = len(h)

    # Confusion cells, human as reference.
    tp = int(((h == "yes") & (d == "yes")).sum())
    tn = int(((h == "no") & (d == "no")).sum())
    fp = int(((h == "no") & (d == "yes")).sum())   # detector over-flags
    fn = int(((h == "yes") & (d == "no")).sum())   # detector misses

    k = _kappa(h, d)
    lo, hi, degen = bootstrap_kappa(h, d)

    # McNemar: are the two discordant cells asymmetric? Exact binomial, since
    # fp+fn is small here and the chi-square approximation would not hold.
    if fp + fn > 0:
        p_mcnemar = float(binomtest(fp, fp + fn, 0.5).pvalue)
        if fp > fn:
            skew = f"false-POSITIVE skew (detector over-flags: {fp} vs {fn})"
        elif fn > fp:
            skew = f"false-NEGATIVE skew (detector misses: {fn} vs {fp})"
        else:
            skew = f"symmetric ({fp} vs {fn})"
    else:
        p_mcnemar, skew = float("nan"), "no discordant pairs"

    return {
        "check": name, "n": n,
        "human_yes": int((h == "yes").sum()), "human_yes_rate": round(float((h == "yes").mean()), 4),
        "detector_yes": int((d == "yes").sum()), "detector_yes_rate": round(float((d == "yes").mean()), 4),
        "observed_agreement": round(float((h == d).mean()), 4),
        "cohens_kappa": round(k, 4),
        "kappa_ci_low": round(lo, 4), "kappa_ci_high": round(hi, 4),
        "bootstrap_degenerate_resamples": degen,
        "tp": tp, "tn": tn, "false_positives": fp, "false_negatives": fn,
        "mcnemar_p": round(p_mcnemar, 6) if p_mcnemar == p_mcnemar else np.nan,
        "skew": skew,
    }


def interpret(k: float) -> str:
    """Landis & Koch bands. Stated as a convention, not a verdict."""
    if k != k:
        return "undefined"
    for lim, lab in [(0.0, "poor"), (0.20, "slight"), (0.40, "fair"),
                     (0.60, "moderate"), (0.80, "substantial"), (1.01, "almost perfect")]:
        if k < lim:
            return lab
    return "almost perfect"


def main() -> None:
    # The report uses non-cp1252 characters (Δ, ·, —) and the Windows console
    # defaults to cp1252, which raises UnicodeEncodeError mid-report — after the
    # CSVs are already written, so it looks like a failure but is only a print.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not FILLED.exists():
        raise SystemExit(f"[FATAL] {FILLED} not found. Export from annotate.html first.")

    a = pd.read_csv(FILLED, keep_default_na=False)
    s = pd.read_csv(SAMPLE)
    df = a.merge(s[["annot_id", "mode_used", "unsupported_rebuilt",
                    "ehr_contradiction_rebuilt", "question_type"]],
                 on="annot_id", suffixes=("", "_s"))

    # ── Completeness / validity gates ─────────────────────────────────────────
    blank = df[(df.human_ehr_contradiction == "") | (df.human_unsupported == "")]
    if len(blank):
        raise SystemExit(
            f"[FATAL] {len(blank)} row(s) not annotated: "
            f"{', '.join(blank.annot_id)}. Finish them and re-export.")

    valid_ehr = {"yes", "no", "na"}
    valid_uns = {"yes", "no"}
    if not set(df.human_ehr_contradiction) <= valid_ehr:
        raise SystemExit(f"[FATAL] bad human_ehr_contradiction values: "
                         f"{set(df.human_ehr_contradiction) - valid_ehr}")
    if not set(df.human_unsupported) <= valid_uns:
        raise SystemExit(f"[FATAL] bad human_unsupported values: "
                         f"{set(df.human_unsupported) - valid_uns}")

    # The n/a rule must hold exactly, in both directions.
    bad_t = df[(df.mode_used == "T") & (df.human_ehr_contradiction != "na")]
    bad_nt = df[(df.mode_used != "T") & (df.human_ehr_contradiction == "na")]
    if len(bad_t) or len(bad_nt):
        raise SystemExit(f"[FATAL] n/a rule violated: {len(bad_t)} mode-T rows not n/a, "
                         f"{len(bad_nt)} non-T rows marked n/a.")

    print(f"[INFO] {len(df)} annotated rows, all valid.")

    # ── Check 1: EHR-contradiction (defined only off mode T) ──────────────────
    ehr = df[df.mode_used != "T"]
    r_ehr = analyse("ehr_contradiction", ehr.human_ehr_contradiction, ehr.det_ehr_label)

    # ── Check 2: unsupported claim, at the PRE-REGISTERED threshold ───────────
    det_uns = np.where(df.unsupported_rebuilt >= PREREG_THRESHOLD, "yes", "no")
    r_uns = analyse(f"unsupported (threshold>={PREREG_THRESHOLD}, PRE-REGISTERED)",
                    df.human_unsupported, pd.Series(det_uns))

    results = pd.DataFrame([r_ehr, r_uns])
    results.insert(0, "kappa_band", [interpret(r["cohens_kappa"]) for r in (r_ehr, r_uns)])

    # ── Sensitivity sweep (NOT for threshold selection) ───────────────────────
    sweep_rows = []
    for t in SWEEP:
        d = np.where(df.unsupported_rebuilt >= t, "yes", "no")
        r = analyse(f"unsupported@{t}", df.human_unsupported, pd.Series(d))
        sweep_rows.append({"threshold": t, "is_prereg": t == PREREG_THRESHOLD,
                           "detector_yes": r["detector_yes"],
                           "observed_agreement": r["observed_agreement"],
                           "cohens_kappa": r["cohens_kappa"],
                           "false_positives": r["false_positives"],
                           "false_negatives": r["false_negatives"]})
    sweep = pd.DataFrame(sweep_rows)

    # ── Confusion matrices ────────────────────────────────────────────────────
    def cm(name, h, d):
        return pd.DataFrame({
            "check": name,
            "cell": ["human_yes/det_yes", "human_yes/det_no",
                     "human_no/det_yes", "human_no/det_no"],
            "count": [int(((h == "yes") & (d == "yes")).sum()),
                      int(((h == "yes") & (d == "no")).sum()),
                      int(((h == "no") & (d == "yes")).sum()),
                      int(((h == "no") & (d == "no")).sum())],
        })
    conf = pd.concat([
        cm("ehr_contradiction", ehr.human_ehr_contradiction.to_numpy(), ehr.det_ehr_label.to_numpy()),
        cm(f"unsupported@{PREREG_THRESHOLD}", df.human_unsupported.to_numpy(), det_uns),
    ], ignore_index=True)

    # ── Disagreement listing, for qualitative follow-up ───────────────────────
    dis = []
    for _, r in ehr.iterrows():
        if r.human_ehr_contradiction != r.det_ehr_label:
            dis.append({"check": "ehr_contradiction", "annot_id": r.annot_id,
                        "q_idx": r.q_idx, "system": r.system, "mode_used": r.mode_used,
                        "question_type": r.question_type, "human": r.human_ehr_contradiction,
                        "detector": r.det_ehr_label, "score": r.ehr_contradiction_rebuilt,
                        "note": r.human_note})
    for (_, r), d in zip(df.iterrows(), det_uns):
        if r.human_unsupported != d:
            dis.append({"check": "unsupported", "annot_id": r.annot_id,
                        "q_idx": r.q_idx, "system": r.system, "mode_used": r.mode_used,
                        "question_type": r.question_type, "human": r.human_unsupported,
                        "detector": d, "score": round(float(r.unsupported_rebuilt), 4),
                        "note": r.human_note})
    disagreements = pd.DataFrame(dis)

    results.to_csv(ANNOT_DIR / "annotation_agreement.csv", index=False)
    conf.to_csv(ANNOT_DIR / "annotation_confusion_matrix.csv", index=False)
    sweep.to_csv(ANNOT_DIR / "annotation_threshold_sweep.csv", index=False)
    disagreements.to_csv(ANNOT_DIR / "annotation_disagreements.csv", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("HUMAN vs AUTOMATIC DETECTOR — AGREEMENT")
    print("=" * 78)
    for r in (r_ehr, r_uns):
        print(f"\n  {r['check']}   (n = {r['n']})")
        print(f"    human says yes     : {r['human_yes']:>3}  ({r['human_yes_rate']:.1%})")
        print(f"    detector says yes  : {r['detector_yes']:>3}  ({r['detector_yes_rate']:.1%})")
        print(f"    observed agreement : {r['observed_agreement']:.4f}")
        print(f"    Cohen's kappa      : {r['cohens_kappa']:.4f}"
              f"  95% CI [{r['kappa_ci_low']:.4f}, {r['kappa_ci_high']:.4f}]"
              f"  ({interpret(r['cohens_kappa'])})")
        print(f"    confusion          : TP {r['tp']}  TN {r['tn']}  "
              f"FP {r['false_positives']}  FN {r['false_negatives']}")
        print(f"    McNemar p          : {r['mcnemar_p']}   -> {r['skew']}")
        if r["bootstrap_degenerate_resamples"]:
            print(f"    [!] {r['bootstrap_degenerate_resamples']} of {N_BOOT} resamples "
                  f"were degenerate — the CI is unstable at this prevalence.")
        if min(r["human_yes"], r["detector_yes"]) < 10:
            print(f"    [!] UNDERPOWERED: fewer than 10 positives on one side. "
                  f"Kappa is highly prevalence-sensitive here; report the CI, "
                  f"not the point estimate alone.")

    print("\n" + "-" * 78)
    print(f"SENSITIVITY SWEEP — unsupported threshold "
          f"(PRE-REGISTERED = {PREREG_THRESHOLD}; sweep is robustness only)")
    print("-" * 78)
    print(sweep.to_string(index=False))
    print("\n  The threshold was locked before annotation. Do NOT adopt the")
    print("  kappa-maximising row: that fits the metric to its own reference standard.")

    if META.exists():
        drift = json.loads(META.read_text())["context_drift"]
        tot = drift["stored_vs_rebuilt (total drift vs eval-time labels)"]
        print("\n" + "-" * 78)
        print("CARRIED FORWARD — context drift vs eval time (from sample build)")
        print("-" * 78)
        print(f"  unsupported label flips {tot['unsupported_label_disagreements']}/{tot['n']}"
              f" · EHR flips {tot['ehr_label_disagreements']}/{tot['n']}"
              f" · mean |Δ score| {tot['unsupported_mean_abs_score_delta']}")
        print("  Human and detector judged IDENTICAL context, so this does not enter")
        print("  the kappa; it bounds how far these labels sit from the paper's own.")

    print(f"\n[INFO] Wrote 4 CSVs to {ANNOT_DIR}")
    print(f"[INFO] {len(disagreements)} disagreement rows saved for qualitative review.")


if __name__ == "__main__":
    main()
