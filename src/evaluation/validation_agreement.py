"""
src/evaluation/validation_agreement.py — did the corrected scorer actually fix it?

Scores the round-2 validation sample (30 unseen rows + 3 attention checks) and
answers one question: on data neither the fix nor the annotator has seen, does
the corrected negation-contradiction scorer stop producing false positives
without losing genuine detections?

WHY NOT COHEN'S KAPPA
=====================
The sample is deliberately enriched for rows the ORIGINAL detector flagged, so
its prevalence is not the eval set's, and if the fix works the expected outcome
is near-zero positives on both sides — where kappa is unstable or undefined.
The primary statistic is therefore a PAIRED McNEMAR test of original vs
corrected, both scored against the human labels on the same rows. That is the
direct test of "did the fix reduce errors", and pairing removes between-row
variance. Kappa is reported as a footnote only.

ATTENTION CHECKS are excluded from every statistic and reported separately.
They are genuine constructed contradictions; if the annotator missed them, a
uniform "no" response set cannot be trusted and the result is void.

Usage:
    python -m src.evaluation.validation_agreement
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

DIR = Path("experiments/results/annotation_v2")
FILLED = DIR / "validation_filled.csv"
SAMPLE = DIR / "validation_sample.csv"
META = DIR / "validation_metadata.json"


def confusion(h: np.ndarray, d: np.ndarray) -> dict:
    return {
        "tp": int(((h == "yes") & (d == "yes")).sum()),
        "tn": int(((h == "no") & (d == "no")).sum()),
        "fp": int(((h == "no") & (d == "yes")).sum()),
        "fn": int(((h == "yes") & (d == "no")).sum()),
        "accuracy": round(float((h == d).mean()), 4),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not FILLED.exists():
        raise SystemExit(f"[FATAL] {FILLED} not found. Export from annotate_v2.html first.")

    f = pd.read_csv(FILLED, keep_default_na=False)
    s = pd.read_csv(SAMPLE)
    df = f.merge(s[["annot_id", "stratum", "is_attention_check",
                    "ehr_contradiction_v1", "ehr_contradiction_v2"]], on="annot_id")

    # ── gates ────────────────────────────────────────────────────────────────
    blank = df[df.human_contradiction == ""]
    if len(blank):
        raise SystemExit(f"[FATAL] {len(blank)} row(s) unannotated: {', '.join(blank.annot_id)}")
    if not set(df.human_contradiction) <= {"yes", "no"}:
        raise SystemExit(f"[FATAL] bad labels: {set(df.human_contradiction) - {'yes','no'}}")
    print(f"[INFO] {len(df)} rows, all annotated and valid.")

    # ── attention checks (excluded from stats) ───────────────────────────────
    checks = df[df.is_attention_check]
    caught = int((checks.human_contradiction == "yes").sum())
    det_caught = int((checks.det_v2 == "yes").sum())
    print("\n" + "=" * 78)
    print("ATTENTION CHECKS (constructed genuine contradictions, excluded from stats)")
    print("=" * 78)
    print(f"  human caught    : {caught}/{len(checks)}")
    print(f"  corrected caught: {det_caught}/{len(checks)}")
    checks_ok = caught == len(checks)
    if not checks_ok:
        print("  [!] ANNOTATOR MISSED A CONSTRUCTED CONTRADICTION. A uniform 'no'")
        print("      response set cannot be distinguished from inattention. Treat the")
        print("      result below as VOID and re-annotate.")

    # ── real rows ────────────────────────────────────────────────────────────
    real = df[~df.is_attention_check]
    h = real.human_contradiction.to_numpy()
    v1 = real.det_v1.to_numpy()
    v2 = real.det_v2.to_numpy()

    c1, c2 = confusion(h, v1), confusion(h, v2)

    # Paired McNemar on per-row correctness.
    ok1, ok2 = (v1 == h), (v2 == h)
    b = int((ok1 & ~ok2).sum())        # original right, corrected wrong
    c = int((~ok1 & ok2).sum())        # original wrong, corrected right
    p_mcnemar = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) else float("nan")

    def _kappa(x, y):
        if len(set(x)) == 1 and len(set(y)) == 1:
            return 1.0 if x[0] == y[0] else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(cohen_kappa_score(x, y, labels=["no", "yes"]))

    print("\n" + "=" * 78)
    print(f"ORIGINAL vs CORRECTED, against human labels (n = {len(real)} unseen rows)")
    print("=" * 78)
    for name, cc, dd in [("original ", c1, v1), ("corrected", c2, v2)]:
        print(f"  {name}: TP {cc['tp']}  TN {cc['tn']}  FP {cc['fp']}  FN {cc['fn']}"
              f"   accuracy {cc['accuracy']:.4f}   kappa {_kappa(h, dd):.4f}")
    print(f"\n  false positives: {c1['fp']} -> {c2['fp']}"
          f"   ({c1['fp'] - c2['fp']} eliminated)")
    print(f"  McNemar (paired): original-only-correct {b}, corrected-only-correct {c},"
          f" p = {p_mcnemar:.6f}")

    # ── stratum A: the pre-registered success criterion ──────────────────────
    a = real[real.stratum == "A_originally_flagged"]
    a_human_yes = int((a.human_contradiction == "yes").sum())
    a_v2_yes = int((a.det_v2 == "yes").sum())
    a_v1_yes = int((a.det_v1 == "yes").sum())

    # Exact binomial CI on the corrected FP rate over stratum A.
    a_fp = int(((a.human_contradiction == "no") & (a.det_v2 == "yes")).sum())
    ci = binomtest(a_fp, len(a), 0.5).proportion_ci(confidence_level=0.95)

    print("\n" + "-" * 78)
    print(f"STRATUM A — rows the ORIGINAL flagged (n = {len(a)})")
    print("-" * 78)
    print(f"  original flags      : {a_v1_yes}/{len(a)}")
    print(f"  human says genuine  : {a_human_yes}/{len(a)}")
    print(f"  corrected flags     : {a_v2_yes}/{len(a)}")
    print(f"  corrected FP rate   : {a_fp}/{len(a)} = {a_fp/len(a):.4f}"
          f"   95% CI [{ci.low:.4f}, {ci.high:.4f}]")

    passed = (a_human_yes <= 1) and (a_v2_yes <= 1) and checks_ok
    print("\n" + "=" * 78)
    print("PRE-REGISTERED CRITERION (locked 2026-08-16, before annotation)")
    print("=" * 78)
    print(f"  <=1 stratum-A row genuinely a contradiction : {a_human_yes} -> "
          f"{'OK' if a_human_yes <= 1 else 'FAIL'}")
    print(f"  corrected flags <=1 stratum-A row          : {a_v2_yes} -> "
          f"{'OK' if a_v2_yes <= 1 else 'FAIL'}")
    print(f"  annotator caught all attention checks      : {caught}/{len(checks)} -> "
          f"{'OK' if checks_ok else 'FAIL'}")
    print(f"\n  RESULT: {'FIX VALIDATED' if passed else 'NOT VALIDATED'}")

    # ── outputs ──────────────────────────────────────────────────────────────
    summary = pd.DataFrame([
        {"detector": "original", **c1, "kappa": round(_kappa(h, v1), 4)},
        {"detector": "corrected", **c2, "kappa": round(_kappa(h, v2), 4)},
    ])
    summary.to_csv(DIR / "validation_summary.csv", index=False)

    dis = real[(real.det_v1 != real.human_contradiction) |
               (real.det_v2 != real.human_contradiction)]
    dis[["annot_id", "q_idx", "system", "mode_used", "question_type", "stratum",
         "human_contradiction", "det_v1", "det_v2", "human_note"]].to_csv(
        DIR / "validation_disagreements.csv", index=False)

    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_real": int(len(real)), "n_attention_checks": int(len(checks)),
        "attention_checks_caught_by_human": caught,
        "original": c1, "corrected": c2,
        "false_positives_eliminated": c1["fp"] - c2["fp"],
        "mcnemar": {"original_only_correct": b, "corrected_only_correct": c, "p": p_mcnemar},
        "stratum_A": {"n": int(len(a)), "original_flags": a_v1_yes,
                      "human_genuine": a_human_yes, "corrected_flags": a_v2_yes,
                      "corrected_fp_rate": round(a_fp / len(a), 4),
                      "ci95": [round(ci.low, 4), round(ci.high, 4)]},
        "prereg_passed": bool(passed),
    }
    (DIR / "validation_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[INFO] Wrote validation_summary.csv, validation_disagreements.csv, "
          f"validation_results.json to {DIR}")


if __name__ == "__main__":
    main()
