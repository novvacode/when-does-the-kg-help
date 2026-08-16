"""
src/evaluation/validation_agreement_v3.py — round 3 scoring.

Round 2 asked "did the fix stop over-flagging?" on diagnoses/primary_diagnosis
(answer: yes, 20 false positives -> 0). Round 3 asks the complementary question
on the population that actually drives the recomputed rate: **are the corrected
scorer's remaining POSITIVES real?** 40 of its 46 flags, and all 33 of its newly
caught rows, are monitoring_labs/lab.

The primary statistic is therefore PRECISION on stratum A (rows the corrected
scorer flags), with an exact binomial 95% CI — not McNemar, which answered
round 2's question, and not kappa, which is unstable on an enriched sample.

PRE-REGISTERED (locked 2026-08-17, before annotation):
    validated if >= 12 of 15 stratum-A rows are confirmed genuine
    (precision >= 0.80).

The bar is not moved after the fact. If it fails, the honest conclusion is that
the fix is validated for diagnosis questions and NOT for monitoring_labs, which
is a narrower but real result.

Usage:
    python -m src.evaluation.validation_agreement_v3
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

DIR = Path("experiments/results/annotation_v3")
FILLED = DIR / "validation_filled.csv"
SAMPLE = DIR / "validation_sample.csv"
PRECISION_BAR = 12
N_FLAGGED = 15


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not FILLED.exists():
        raise SystemExit(f"[FATAL] {FILLED} not found.")

    f = pd.read_csv(FILLED, keep_default_na=False)
    s = pd.read_csv(SAMPLE)
    df = f.merge(s[["annot_id", "stratum", "is_attention_check", "question_type",
                    "ehr_contradiction_v1", "ehr_contradiction_v2", "v1_budgeted"]],
                 on="annot_id", suffixes=("", "_s"))

    blank = df[df.human_contradiction == ""]
    if len(blank):
        raise SystemExit(f"[FATAL] {len(blank)} unannotated: {', '.join(blank.annot_id)}")
    if not set(df.human_contradiction) <= {"yes", "no"}:
        raise SystemExit(f"[FATAL] bad labels: {set(df.human_contradiction) - {'yes','no'}}")

    # det_v2 must match the freshly rebuilt score on every row (attention checks
    # included — their answers are perturbed, so the recompute column does not apply).
    fresh = (df.ehr_contradiction_v2 > 0).map({True: "yes", False: "no"})
    if not (df.det_v2 == fresh).all():
        raise SystemExit("[FATAL] det_v2 disagrees with the rebuilt score.")
    print(f"[INFO] {len(df)} rows, all annotated and consistent.")

    checks = df[df.is_attention_check]
    caught = int((checks.human_contradiction == "yes").sum())
    checks_ok = caught == len(checks)
    print("\n" + "=" * 78)
    print("ATTENTION CHECKS (excluded from all statistics)")
    print("=" * 78)
    print(f"  human caught     : {caught}/{len(checks)}")
    print(f"  corrected caught : {int((checks.det_v2 == 'yes').sum())}/{len(checks)}")
    if not checks_ok:
        print("  [!] MISSED — treat the result below as VOID and re-annotate.")

    real = df[~df.is_attention_check]
    a = real[real.stratum == "A_corrected_flags"]
    b = real[real.stratum == "B_control"]

    tp = int((a.human_contradiction == "yes").sum())
    fp = int((a.human_contradiction == "no").sum())
    fn = int((b.human_contradiction == "yes").sum())
    tn = int((b.human_contradiction == "no").sum())

    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    ci = binomtest(tp, tp + fp, 0.5).proportion_ci(confidence_level=0.95)

    print("\n" + "=" * 78)
    print(f"CORRECTED SCORER on monitoring_labs/lab  (n = {len(real)} real rows)")
    print("=" * 78)
    print(f"  stratum A (flagged, n={len(a)})   : {tp} genuine, {fp} false positive")
    print(f"  stratum B (unflagged, n={len(b)}) : {fn} missed contradiction, {tn} correct silence")
    print(f"\n  precision : {prec:.4f}   95% CI [{ci.low:.4f}, {ci.high:.4f}]")
    print(f"  recall    : {rec:.4f}   ({tp}/{tp+fn})")
    print(f"  accuracy  : {(tp+tn)/len(real):.4f}   ({tp+tn}/{len(real)})")

    # What the ORIGINAL scorer would have done on these same rows.
    v1 = (real.ehr_contradiction_v1 > 0)
    h = (real.human_contradiction == "yes")
    v1_tp, v1_fp = int((v1 & h).sum()), int((v1 & ~h).sum())
    v1_fn = int((~v1 & h).sum())
    print(f"\n  for reference, ORIGINAL scorer on the same {len(real)} rows:")
    print(f"    TP {v1_tp}  FP {v1_fp}  FN {v1_fn}"
          f"   precision {v1_tp/(v1_tp+v1_fp) if (v1_tp+v1_fp) else float('nan'):.4f}")
    newly = real[(real.v1_budgeted == 0) & (real.ehr_contradiction_v2 > 0)]
    newly_genuine = int((newly.human_contradiction == "yes").sum())
    print(f"    rows NEWLY caught by the fix: {len(newly)}, of which genuine: {newly_genuine}")

    passed = (tp >= PRECISION_BAR) and checks_ok
    print("\n" + "=" * 78)
    print("PRE-REGISTERED CRITERION (locked 2026-08-17, before annotation)")
    print("=" * 78)
    print(f"  >= {PRECISION_BAR}/{N_FLAGGED} stratum-A rows genuine : {tp}/{len(a)} -> "
          f"{'OK' if tp >= PRECISION_BAR else 'FAIL'}")
    print(f"  attention checks all caught      : {caught}/{len(checks)} -> "
          f"{'OK' if checks_ok else 'FAIL'}")
    print(f"\n  RESULT: {'VALIDATED' if passed else 'NOT VALIDATED'}")
    if not passed and tp >= PRECISION_BAR - 2:
        print("\n  The bar was set before annotation and is NOT being moved. Report the")
        print("  precision with its CI and treat monitoring_labs as only partially")
        print("  validated; the diagnosis-question result from round 2 is unaffected.")

    out = {
        "generated": datetime.now().isoformat(timespec="seconds"), "round": 3,
        "n_real": int(len(real)), "n_attention_checks": int(len(checks)),
        "attention_checks_caught": caught,
        "stratum_A": {"n": len(a), "tp": tp, "fp": fp},
        "stratum_B": {"n": len(b), "fn": fn, "tn": tn},
        "precision": round(prec, 4), "precision_ci95": [round(ci.low, 4), round(ci.high, 4)],
        "recall": round(rec, 4),
        "original_scorer_on_same_rows": {"tp": v1_tp, "fp": v1_fp, "fn": v1_fn},
        "newly_caught_rows": int(len(newly)), "newly_caught_genuine": newly_genuine,
        "prereg_bar": f">={PRECISION_BAR}/{N_FLAGGED}",
        "prereg_passed": bool(passed),
    }
    (DIR / "validation_v3_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    dis = real[(real.det_v2 != real.human_contradiction)]
    dis[["annot_id", "q_idx", "system", "mode_used", "question_type", "stratum",
         "human_contradiction", "det_v1", "det_v2", "human_note"]].to_csv(
        DIR / "validation_v3_disagreements.csv", index=False)
    print(f"\n[INFO] Wrote validation_v3_results.json and "
          f"validation_v3_disagreements.csv ({len(dis)} rows) to {DIR}")


if __name__ == "__main__":
    main()
