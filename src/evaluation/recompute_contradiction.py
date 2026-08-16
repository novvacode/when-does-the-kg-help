"""
src/evaluation/recompute_contradiction.py — replace the EHR-contradiction column
with the validated negation-contradiction rate, across all 2,100 eval rows.

No GPU and no regeneration: answers are stored in per_question_results.csv, and
context rebuilding reproduces eval time exactly (0 drift, verified 2026-08-16).
Only retrieval runs — roughly 900 unique (hadm_id, question, mode) contexts.

REPORTS TWO VARIANTS, because they are NOT interchangeable
----------------------------------------------------------
  budgeted    the context the model actually received, and the one the human
              validation was run against. RECOMMENDED as the paper's number.
  unbudgeted  how the ORIGINAL column was computed (run_evaluation.py scores
              ctx["prompt_context"] before generate() budgets it). Reported so
              the change is decomposable into "fix" vs "context basis".

THE n/a RULE STILL APPLIES
--------------------------
Mode T receives no EHR snapshot, so contradiction against it is undefined —
n/a, never 0.0000 (RESEARCH_LOG 2026-08-15). Rates are computed over scorable
(mode != T) rows only, and the n/a count is reported alongside. System T is
entirely n/a.

NAMING
------
This is the **negation-contradiction rate**: answer negates what the EHR
asserts (Type 1). Type 2 (answer asserts what the record refutes) requires
clinical reasoning and is NOT measured. Do not report this under the broader
"EHR-contradiction" name.

Usage:
    python -m src.evaluation.recompute_contradiction     # needs Neo4j running
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.build_annotation_sample import rebuild_contexts
from src.evaluation.fix_ehr_contradiction import (
    negation_contradiction_score, ehr_contradiction_score_v1,
)

RESULTS = Path("experiments/results/final_eval/per_question_results.csv")
OUT_DIR = Path("experiments/results/final_eval")


def main() -> None:
    df = pd.read_csv(RESULTS)
    print(f"[INFO] {len(df)} rows / {df.q_idx.nunique()} questions / {df.system.nunique()} systems")

    ctx = rebuild_contexts(df)
    keys = list(zip(df.hadm_id, df.question, df.mode_used))
    unb = [ctx.get(k, ("", "", 0))[0] for k in keys]
    bud = [ctx.get(k, ("", "", 0))[1] for k in keys]
    df["n_kg_facts_rebuilt"] = [ctx.get(k, ("", "", 0))[2] for k in keys]

    tek = df[df.mode_used == "T+E+K"]
    lost = tek[(tek.n_kg_facts > 0) & (tek.n_kg_facts_rebuilt == 0)]
    if len(lost):
        raise SystemExit(f"[FATAL] {len(lost)} T+E+K rows lost KG facts. Nothing written.")
    print(f"[INFO] KG integrity OK ({int((tek.n_kg_facts_rebuilt > 0).sum())}/{len(tek)} "
          f"T+E+K rows with facts).")

    ans = df.predicted_answer.fillna("").tolist()
    df["v1_unbudgeted"] = [ehr_contradiction_score_v1(a, c) for a, c in zip(ans, unb)]
    df["v1_budgeted"] = [ehr_contradiction_score_v1(a, c) for a, c in zip(ans, bud)]
    df["v2_unbudgeted"] = [negation_contradiction_score(a, c) for a, c in zip(ans, unb)]
    df["v2_budgeted"] = [negation_contradiction_score(a, c) for a, c in zip(ans, bud)]

    scorable = df[df.mode_used != "T"]

    # Reproduction check against the stored column (v1 on unbudgeted context).
    flips = int(((df.ehr_contradiction > 0) != (df.v1_unbudgeted > 0)).sum())
    print(f"[INFO] Stored-vs-recomputed v1 label flips: {flips}/{len(df)} "
          f"(0 expected — confirms the rebuild reproduces eval time)")

    # The corrected scorer is NOT a strict subset of the original; quantify.
    on_v1 = scorable.v1_budgeted > 0
    on_v2 = scorable.v2_budgeted > 0
    print(f"[INFO] scorable rows: {len(scorable)}  "
          f"| v1 flags {int(on_v1.sum())}  v2 flags {int(on_v2.sum())}")
    print(f"[INFO] suppressed by fix (v1 yes -> v2 no): {int((on_v1 & ~on_v2).sum())}")
    print(f"[INFO] NEWLY caught by fix (v1 no -> v2 yes): {int((~on_v1 & on_v2).sum())}")

    rows = []
    for sysname, g in df.groupby("system"):
        gs = g[g.mode_used != "T"]
        n_na = int((g.mode_used == "T").sum())
        rec = {"system": sysname, "n_rows": len(g), "n_na_mode_T": n_na,
               "n_scorable": len(gs),
               "orig_ehr_contradiction_reported": round(float(g.ehr_contradiction.mean()), 4)}
        if len(gs):
            rec.update({
                "negation_contradiction_rate_budgeted": round(float((gs.v2_budgeted > 0).mean()), 4),
                "negation_contradiction_rate_unbudgeted": round(float((gs.v2_unbudgeted > 0).mean()), 4),
                "v1_rate_budgeted_for_reference": round(float((gs.v1_budgeted > 0).mean()), 4),
                "n_flagged_budgeted": int((gs.v2_budgeted > 0).sum()),
            })
        else:
            rec.update({"negation_contradiction_rate_budgeted": np.nan,
                        "negation_contradiction_rate_unbudgeted": np.nan,
                        "v1_rate_budgeted_for_reference": np.nan,
                        "n_flagged_budgeted": 0})
        rows.append(rec)
    table = pd.DataFrame(rows).sort_values("system")

    table.to_csv(OUT_DIR / "negation_contradiction_by_system.csv", index=False)
    df[["q_idx", "hadm_id", "system", "mode_used", "question_type",
        "ehr_contradiction", "v1_unbudgeted", "v1_budgeted",
        "v2_unbudgeted", "v2_budgeted"]].to_csv(
        OUT_DIR / "negation_contradiction_per_question.csv", index=False)

    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "metric_name": "negation_contradiction_rate",
        "scope": "Type 1 only (answer negates what the EHR asserts). "
                 "Type 2 (answer asserts what the record refutes) NOT measured.",
        "validated": "30 unseen rows, 2026-08-16: FP 20->0, McNemar p=2.0e-06",
        "n_rows": int(len(df)), "n_scorable": int(len(scorable)),
        "stored_vs_recomputed_v1_flips": flips,
        "suppressed_by_fix": int((on_v1 & ~on_v2).sum()),
        "newly_caught_by_fix": int((~on_v1 & on_v2).sum()),
        "recommended_variant": "budgeted (what the model saw; what was validated)",
    }
    (OUT_DIR / "negation_contradiction_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n" + "=" * 96)
    print("NEGATION-CONTRADICTION RATE BY SYSTEM  (mode-T rows are n/a, excluded)")
    print("=" * 96)
    print(table[["system", "n_scorable", "n_na_mode_T", "orig_ehr_contradiction_reported",
                 "negation_contradiction_rate_budgeted",
                 "negation_contradiction_rate_unbudgeted"]].to_string(index=False))
    print("\n  'orig_ehr_contradiction_reported' is the OLD column, which averaged mode-T")
    print("  rows in as 0.0000 and is invalid on two counts (n/a rule + kappa -0.037).")
    print(f"\n[INFO] Wrote 3 files to {OUT_DIR}")


if __name__ == "__main__":
    main()
