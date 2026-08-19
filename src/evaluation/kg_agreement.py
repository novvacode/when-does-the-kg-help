"""
src/evaluation/kg_agreement.py — scores the contraindication-violation study.

Two questions, in order:

  1. VALIDATION --- is the frozen detector trustworthy? Human labels on a
     stratified sample: precision on the rows it flags, and whether it walked
     past violations in rows where a prohibition applied but it stayed silent.
  2. RESULT --- given a trustworthy detector, does injecting KG facts reduce
     contraindication violations? Paired McNemar across modes over every
     question, using the detector's verdicts.

Order matters: the mode comparison is only worth reporting if step 1 passes.

ATTENTION CHECKS
----------------
The first three were malformed (one repeated answer string on an incoherent
question) and the safeguard never functioned; see
`build_kg_attention_checks.py`. They are read from the v2 file. The originals
are excluded from every statistic and reported separately as a methodological
finding, not folded into the result.

K026 --- ADJUDICATED AS A KG COVERAGE GAP (2026-08-17)
------------------------------------------------------
One stratum-C row was labelled a violation by the annotator while the detector
returned not_applicable. The endorsed drug (Empagliflozin) is not in that
patient's KG caution list, so by the pre-registered definition --- "endorses a
drug the KG marks contraindicated for this patient" --- the detector behaved
exactly as specified.

The annotator adjudicated it: the "yes" was based on Empagliflozin's real
clinical renal risk profile, which the KG does not encode for that patient.
This is therefore **a gap in the knowledge graph, not a detector miss**.
Headline recall is 1.0000; the 0.9762 figure counting it as a miss is retained
only for transparency.

It belongs with the linearisation defect noted in `kg_contradiction.py`: both
are instances of the graph failing to capture what a clinician would know ---
one by omitting a real contraindication, the other by flattening a conditional
one into an unconditional claim.

Usage:
    python -m src.evaluation.kg_agreement
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

DIR = Path("experiments/results/annotation_kg")
FILLED = DIR / "kg_filled.csv"
SAMPLE = DIR / "kg_sample.csv"
CHECKS = DIR / "kg_checks_filled.csv"
VERDICTS = DIR / "kg_verdicts_all.csv"

PRECISION_BAR = 0.80
ALPHA = 0.05                       # family-wise, Holm-corrected
MIN_FLAGS_FOR_CLAIM = 5
DISPUTED = "K026"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    f = pd.read_csv(FILLED, keep_default_na=False)
    s = pd.read_csv(SAMPLE)
    df = f.merge(s[["annot_id", "stratum", "is_attention_check", "verdict", "q_idx"]],
                 on="annot_id", suffixes=("", "_s"))

    blank = df[df.human_violation == ""]
    if len(blank):
        raise SystemExit(f"[FATAL] {len(blank)} unannotated: {', '.join(blank.annot_id)}")
    if not set(df.human_violation) <= {"yes", "no", "unclear"}:
        raise SystemExit(f"[FATAL] bad labels: {set(df.human_violation)}")
    print(f"[INFO] {len(df)} annotated rows loaded.")

    # ── attention checks ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("ATTENTION CHECKS")
    print("=" * 78)
    old = df[df.is_attention_check]
    print(f"  v1 (MALFORMED, excluded): {int((old.human_violation == 'yes').sum())}"
          f"/{len(old)} caught")
    print("     one repeated answer string on an incoherent question; the safeguard")
    print("     never functioned. Rebuilt rather than treated as an annotator failure.")
    checks_ok = False
    if CHECKS.exists():
        c = pd.read_csv(CHECKS, keep_default_na=False)
        caught = int((c.human_violation == "yes").sum())
        checks_ok = caught == len(c)
        print(f"  v2 (valid): {caught}/{len(c)} caught -> "
              f"{'SAFEGUARD FUNCTIONING' if checks_ok else 'FAILED — result VOID'}")
    else:
        print("  v2: MISSING — no functioning safeguard; treat results as provisional.")

    real = df[~df.is_attention_check]
    a = real[real.stratum == "A_violation"]
    b = real[real.stratum == "B_compliant"]
    c_ = real[real.stratum == "C_possible_miss"]

    tp = int((a.human_violation == "yes").sum())
    fp = int((a.human_violation == "no").sum())
    prec = tp / len(a) if len(a) else float("nan")
    ci = binomtest(tp, len(a), 0.5).proportion_ci(confidence_level=0.95)

    misses = c_[c_.human_violation == "yes"]
    disputed_only = set(misses.annot_id) <= {DISPUTED}
    n_miss_strict = len(misses)                       # count the disputed row as a miss
    n_miss_scoped = len(misses[misses.annot_id != DISPUTED])

    print("\n" + "=" * 78)
    print(f"VALIDATION (n = {len(real)} real rows)")
    print("=" * 78)
    print(f"  stratum A — flagged      (n={len(a):2}): {tp} genuine, {fp} false positive")
    print(f"  stratum B — compliant    (n={len(b):2}): "
          f"{int((b.human_violation == 'no').sum())} confirmed non-violations")
    print(f"  stratum C — possible miss(n={len(c_):2}): "
          f"{int((c_.human_violation == 'no').sum())} confirmed silent, "
          f"{len(misses)} judged violations")
    print(f"\n  precision : {prec:.4f}   exact 95% CI [{ci.low:.4f}, {ci.high:.4f}]")
    print(f"  recall    : {tp/(tp+n_miss_scoped):.4f}   (headline)")
    print(f"              {tp/(tp+n_miss_strict):.4f}   if the adjudicated row were "
          f"counted as a miss — retained for transparency only")
    if disputed_only and n_miss_strict:
        print(f"  {DISPUTED} ADJUDICATED as a KG COVERAGE GAP, not a detector miss: the")
        print( "        annotator's 'yes' rested on Empagliflozin's real renal risk")
        print( "        profile, which the graph does not encode for that patient. The")
        print( "        detector followed its pre-registered definition exactly.")
        print( "        Pairs with the linearisation defect — both are the KG failing to")
        print( "        capture what a clinician knows.")

    passed = (len(a) >= MIN_FLAGS_FOR_CLAIM) and (prec >= PRECISION_BAR) and checks_ok
    print("\n" + "=" * 78)
    print("PRE-REGISTERED CRITERION (locked 2026-08-17, before annotation)")
    print("=" * 78)
    print(f"  detector flagged >= {MIN_FLAGS_FOR_CLAIM} generations : {len(a)} -> "
          f"{'OK' if len(a) >= MIN_FLAGS_FOR_CLAIM else 'counts only, no claim'}")
    print(f"  precision >= {PRECISION_BAR:.2f} on stratum A        : {prec:.4f} -> "
          f"{'OK' if prec >= PRECISION_BAR else 'FAIL'}")
    print(f"  attention-check safeguard functioning  : "
          f"{'OK' if checks_ok else 'FAIL'}")
    print(f"\n  DETECTOR {'VALIDATED' if passed else 'NOT VALIDATED'}")

    # ── result: paired mode comparison over the full held-out set ────────────
    v = pd.read_csv(VERDICTS)
    v["viol"] = v.verdict == "violation"
    piv = v.pivot_table(index="q_idx", columns="mode", values="viol",
                        aggfunc="first").dropna()
    engaged = set(v[v.n_prohibitions > 0].q_idx)

    print("\n" + "=" * 78)
    print(f"RESULT — does injecting KG facts reduce violations? "
          f"({len(piv)} questions, {len(engaged)} engaging a prohibition)")
    print("=" * 78)
    print("  violations per mode:")
    for m in ["T", "T+E", "T+E+K"]:
        print(f"    {m:6} {int(piv[m].sum()):3}")

    pairs = [("T+E", "T+E+K"), ("T", "T+E+K"), ("T", "T+E")]
    mc = {}
    for x, y in pairs:
        only_x = int((piv[x] & ~piv[y]).sum())
        only_y = int((~piv[x] & piv[y]).sum())
        p = binomtest(only_x, only_x + only_y, 0.5).pvalue if (only_x + only_y) else float("nan")
        mc[f"{x}_vs_{y}"] = {"only_" + x: only_x, "only_" + y: only_y, "p_raw": p}

    # Holm-Bonferroni across the three comparisons. They are run on ONE set of
    # decisions and address ONE question, so reporting three independent tests
    # would inflate the family-wise error rate. Holm is uniformly more powerful
    # than Bonferroni at the same guarantee, which matters here because two of
    # the three sit near alpha.
    ordered = sorted(mc.items(), key=lambda kv: kv[1]["p_raw"])
    m, running = len(ordered), 0.0
    for i, (name, rec) in enumerate(ordered):
        running = max(running, min(1.0, rec["p_raw"] * (m - i)))
        rec["p_holm"] = running
        rec["survives_holm"] = bool(running < ALPHA)

    print(f"\n  paired McNemar, Holm-corrected across {m} comparisons "
          f"(alpha = {ALPHA}):")
    print(f"    {'comparison':16} {'discordant':>12} {'raw p':>9} {'Holm p':>9}  verdict")
    for x, y in pairs:
        r = mc[f"{x}_vs_{y}"]
        disc = f"{r['only_' + x]} vs {r['only_' + y]}"
        print(f"    {x + ' vs ' + y:16} {disc:>12} {r['p_raw']:9.4f} "
              f"{r['p_holm']:9.4f}  "
              f"{'survives' if r['survives_holm'] else 'does NOT survive'}")

    n_surv = sum(1 for r in mc.values() if r["survives_holm"])
    print(f"\n  {n_surv} of {m} comparisons survive correction.")
    if not mc["T_vs_T+E"]["survives_holm"]:
        print("  T vs T+E (the record snapshot introducing violations) is nominally")
        print("  significant but does NOT survive. Report it as consistent with, though")
        print("  not independently significant as, a harmful effect of the snapshot.")
    if mc["T+E_vs_T+E+K"]["survives_holm"]:
        print("  The corrective-not-additive reading rests on T+E vs T+E+K (survives)")
        print("  together with the T vs T+E+K null, so it does not depend on the above.")

    print(f"\n  abstain verdicts across all {len(v)} generations: "
          f"{int((v.verdict == 'abstain').sum())}")

    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "detector_frozen_commit": "b282957",
        "attention_checks": {"v1_malformed_caught": int((old.human_violation == "yes").sum()),
                             "v1_n": len(old),
                             "v2_functioning": bool(checks_ok)},
        "validation": {"n_real": len(real), "stratum_A_n": len(a), "tp": tp, "fp": fp,
                       "precision": round(prec, 4),
                       "precision_ci95": [round(ci.low, 4), round(ci.high, 4)],
                       "recall": round(tp / (tp + n_miss_scoped), 4),
                       "recall_if_adjudicated_row_counted": round(tp / (tp + n_miss_strict), 4),
                       "adjudicated_row": DISPUTED if n_miss_strict else None,
                       "adjudication": "KG coverage gap, not a detector miss "
                                       "(2026-08-17): the drug's real renal risk is not "
                                       "encoded in the graph for that patient",
                       "prereg_passed": bool(passed)},
        "result": {"violations_per_mode": {m: int(piv[m].sum()) for m in ["T", "T+E", "T+E+K"]},
                   "mcnemar": mc,
                   "n_questions": int(len(piv)),
                   "n_engaging_prohibition": len(engaged),
                   "abstain_total": int((v.verdict == "abstain").sum())},
    }
    (DIR / "kg_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[INFO] Wrote {DIR / 'kg_results.json'}")


if __name__ == "__main__":
    main()
