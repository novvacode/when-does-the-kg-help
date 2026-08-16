"""
src/evaluation/build_validation_sample_v3.py — round 3: validate the corrected
scorer's positives on `monitoring_labs` / `lab`, the population round 2 missed.

WHY A THIRD ROUND
=================
Round 2 validated the FALSE-POSITIVE side on `diagnoses`/`primary_diagnosis`,
where the ICD-negation bug lived: 20 original false positives eliminated, 0
introduced (McNemar p = 2.0e-06). But the full recompute then showed that
**40 of the corrected scorer's 46 remaining flags — and all 33 of its NEWLY
caught rows — are `monitoring_labs`/`lab`**, a population no human has looked
at. The recomputed rate is therefore driven almost entirely by unvalidated
detections.

Round 2 asked "did we stop over-flagging?" (answer: yes). Round 3 asks the
complementary question: **are the flags that remain actually real?**

DESIGN (20 real rows + 3 attention checks)
------------------------------------------
  stratum A  15 rows  unseen monitoring_labs/lab rows the CORRECTED scorer
                      flags. These are the detections under test. Most are
                      newly caught by the fix (33 of the 40 candidates are).
  stratum B   5 rows  unseen monitoring_labs/lab rows it does NOT flag, to
                      check it is not missing obvious contradictions there.

Disjoint from BOTH prior samples (session 1's 75 rows and round 2's sample),
enforced on q_idx|system. All rows mode != T, so the n/a rule never applies.

PRE-REGISTERED, LOCKED BEFORE ANNOTATION
----------------------------------------
  · label rule    : corrected score > 0 -> yes (unchanged across all rounds)
  · human question: identical wording to rounds 1 and 2
  · primary stat  : PRECISION of the corrected scorer on stratum A
                    (TP / flagged), with an exact binomial 95% CI
  · criterion     : the corrected scorer's positives on this population are
                    VALIDATED if the human confirms >= 12 of 15 stratum-A rows
                    (precision >= 0.80). Below that, the 33 new detections are
                    not trustworthy and the recomputed rate must NOT enter the
                    paper for these question types.

This criterion can fail, and failing is an acceptable outcome: it would mean
the fix is validated for diagnosis questions only, which is still a real
result and narrower than the recompute implies.

ATTENTION CHECKS: 3 constructed genuine contradictions, excluded from every
statistic, same rationale as round 2 — they distinguish a real response set
from an inattentive one.

Usage:
    python -m src.evaluation.build_validation_sample_v3    # needs Neo4j running
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.build_annotation_sample import rebuild_contexts
from src.evaluation.build_validation_sample import build_html, make_attention_checks
from src.evaluation.fix_ehr_contradiction import (
    negation_contradiction_score, ehr_contradiction_score_v1,
)

RESULTS = Path("experiments/results/final_eval/per_question_results.csv")
RECOMP = Path("experiments/results/final_eval/negation_contradiction_per_question.csv")
PRIOR1 = Path("experiments/results/annotation/annotations_filled.csv")
PRIOR2 = Path("experiments/results/annotation_v2/validation_sample.csv")
OUT_DIR = Path("experiments/results/annotation_v3")
SAMPLE_CSV = OUT_DIR / "validation_sample.csv"
HTML_OUT = OUT_DIR / "annotate_v3.html"

N_FLAGGED = 15
N_CONTROL = 5
N_ATTENTION = 3
SEED = 42
TARGET_TYPES = ["monitoring_labs", "lab"]
PRECISION_BAR = 12          # of N_FLAGGED — pre-registered


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    res = pd.read_csv(RESULTS)
    rec = pd.read_csv(RECOMP)[["q_idx", "system", "v1_budgeted", "v2_budgeted"]]
    df = res.merge(rec, on=["q_idx", "system"], how="left")

    p1 = pd.read_csv(PRIOR1)
    p2 = pd.read_csv(PRIOR2)
    prior = set(p1.q_idx.astype(str) + "|" + p1.system) | set(p2.q_idx.astype(str) + "|" + p2.system)

    df["rid"] = df.q_idx.astype(str) + "|" + df.system
    pool = df[(~df.rid.isin(prior)) & (df.mode_used != "T")
              & (df.question_type.isin(TARGET_TYPES))].copy()
    print(f"[INFO] prior-annotated rows excluded: {len(prior)}")
    print(f"[INFO] unseen {TARGET_TYPES} scorable pool: {len(pool)}")

    flagged = pool[pool.v2_budgeted > 0]
    control = pool[pool.v2_budgeted == 0]
    print(f"[INFO] stratum A candidates (corrected flags): {len(flagged)}"
          f"  ({int((flagged.v1_budgeted == 0).sum())} newly caught by the fix)")
    print(f"[INFO] stratum B candidates (unflagged): {len(control)}")
    if len(flagged) < N_FLAGGED:
        raise SystemExit(f"[FATAL] only {len(flagged)} flagged candidates, need {N_FLAGGED}.")

    rng = np.random.default_rng(SEED)

    def take(sub: pd.DataFrame, n: int, label: str) -> pd.DataFrame:
        idx = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
        out = sub.iloc[np.sort(idx)].copy()
        out["stratum"] = label
        return out

    sample = pd.concat([take(flagged, N_FLAGGED, "A_corrected_flags"),
                        take(control, N_CONTROL, "B_control")], ignore_index=True)
    sample["is_attention_check"] = False

    # Attention-check candidates: unused rows, any dx type (a constructed
    # contradiction reads the same regardless of the question it hangs off).
    spare = df[(~df.rid.isin(prior)) & (~df.rid.isin(set(sample.rid)))
               & (df.mode_used != "T")
               & (df.question_type.isin(["diagnoses", "primary_diagnosis"]))]
    spare = spare.sample(frac=1.0, random_state=SEED).head(40)

    ctx = rebuild_contexts(pd.concat([sample, spare], ignore_index=True))

    checks = make_attention_checks(spare, ctx)
    if len(checks) < N_ATTENTION:
        raise SystemExit(f"[FATAL] only built {len(checks)}/{N_ATTENTION} attention checks.")
    sample = pd.concat([sample, checks], ignore_index=True)

    keys = list(zip(sample.hadm_id, sample.question, sample.mode_used))
    sample["context_unbudgeted"] = [ctx.get(k, ("", "", 0))[0] for k in keys]
    sample["context_budgeted"] = [ctx.get(k, ("", "", 0))[1] for k in keys]
    sample["n_kg_facts_rebuilt"] = [ctx.get(k, ("", "", 0))[2] for k in keys]

    tek = sample[sample.mode_used == "T+E+K"]
    lost = tek[(tek.n_kg_facts > 0) & (tek.n_kg_facts_rebuilt == 0)]
    if len(lost):
        raise SystemExit(f"[FATAL] {len(lost)} T+E+K rows lost KG facts. Nothing written.")

    sample["ehr_contradiction_v1"] = [ehr_contradiction_score_v1(a, c)
                                      for a, c in zip(sample.predicted_answer, sample.context_budgeted)]
    sample["ehr_contradiction_v2"] = [negation_contradiction_score(a, c)
                                      for a, c in zip(sample.predicted_answer, sample.context_budgeted)]
    sample["det_v1_label"] = np.where(sample.ehr_contradiction_v1 > 0, "yes", "no")
    sample["det_v2_label"] = np.where(sample.ehr_contradiction_v2 > 0, "yes", "no")

    # GATE: the rebuilt score must reproduce the recompute this sample was
    # drawn from. If it does not, the sample was selected on stale values.
    real = sample[~sample.is_attention_check]
    mismatch = int(((real.ehr_contradiction_v2 > 0) != (real.v2_budgeted > 0)).sum())
    if mismatch:
        raise SystemExit(f"[FATAL] {mismatch} rows disagree with the recompute they were "
                         f"selected from. Sample would be invalid. Nothing written.")
    print(f"[INFO] Selection gate OK — rebuilt scores match the recompute on all {len(real)} rows.")

    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "annot_id", [f"W{i:03d}" for i in range(1, len(sample) + 1)])
    sample["human_contradiction"] = ""
    sample["human_note"] = ""

    sample.to_csv(SAMPLE_CSV, index=False)
    HTML_OUT.write_text(build_html(sample), encoding="utf-8")

    a = sample[sample.stratum == "A_corrected_flags"]
    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "round": 3, "seed": SEED,
        "purpose": "validate the corrected scorer's POSITIVES on monitoring_labs/lab",
        "n_presented": int(len(sample)), "n_real": int(len(real)),
        "n_attention_checks": int(sample.is_attention_check.sum()),
        "strata": real.stratum.value_counts().to_dict(),
        "question_types": real.question_type.value_counts().to_dict(),
        "overlap_with_prior_rounds": 0,
        "stratum_A_newly_caught_by_fix": int((a.v1_budgeted == 0).sum()),
        "prereg": {
            "label_rule": "corrected score > 0 -> yes",
            "primary_stat": "precision of corrected scorer on stratum A, exact binomial 95% CI",
            "criterion": f">= {PRECISION_BAR}/{N_FLAGGED} stratum-A rows confirmed genuine "
                         f"(precision >= {PRECISION_BAR/N_FLAGGED:.2f})",
            "locked": "2026-08-17, before annotation",
        },
    }
    (OUT_DIR / "validation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("ROUND 3 SAMPLE BUILT")
    print("=" * 74)
    print(f"  presented rows   : {len(sample)}  ({len(real)} real + {N_ATTENTION} attention checks)")
    print(f"  strata           : {meta['strata']}")
    print(f"  question types   : {meta['question_types']}")
    print(f"  overlap w/ prior : 0 (enforced against both earlier samples)")
    print(f"  stratum A newly caught by the fix: {meta['stratum_A_newly_caught_by_fix']}/{len(a)}")
    print(f"\n  PRE-REGISTERED: validated if >= {PRECISION_BAR}/{N_FLAGGED} confirmed genuine")
    print(f"\n  {SAMPLE_CSV}\n  {HTML_OUT}")


if __name__ == "__main__":
    main()
