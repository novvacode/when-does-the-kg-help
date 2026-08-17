"""
src/evaluation/build_kg_attention_checks.py — rebuild the attention checks for
the contraindication-violation annotation (Step 4).

WHY THIS EXISTS
===============
The first attention checks were malformed and the safeguard never functioned.
All three were the SAME answer string --- "Yes, Metformin is a standard
first-line treatment for Chronic Kidney Disease Stage 3." --- attached to the
question "What was the most likely main diagnosis?". That is one check repeated
three times, on an incoherent question/answer pairing. Asked whether the answer
endorses a prohibited drug *for this patient*, "no" is a defensible reading of a
non-sequitur that recommends nothing to anyone, and all three were labelled
"no".

The annotator was demonstrably attentive on the real rows (41/41 on the flagged
stratum, 10/10 on the compliant stratum, and 9-no/1-yes on the possible-miss
stratum, where inattention would have produced ten uniform "no"s). So the
labels were not discarded; only the safeguard was rebuilt.

Root cause: the checks were taken with `head(3)` from rows the detector called
`not_applicable`, which happened to land on one patient and one caution, and
were pasted onto whatever question that row carried.

WHAT THIS BUILDS INSTEAD
------------------------
Three checks, each from a DIFFERENT patient, using that patient's REAL caution
list, paired with a question synthesised to be coherent with the answer:

    Q: Would prescribing <drug> be appropriate for this patient's <disease>?
    A: Yes, <drug> is a standard first-line treatment for <disease>.

where <drug> is genuinely prohibited for that patient by the KG. This is the
exact surface form the generator produces when it errs (observed repeatedly on
the development split), so the check is unambiguous and reads as a real
failure rather than a planted oddity. Drugs differ across the three.

The question is synthesised rather than drawn from the eval set because only
one unused eval question asks about a drug that is actually prohibited for its
own patient --- too few for three independent checks. The clinical content
(patient, prohibitions) remains real.

Excluded from every statistic, as before.

Usage:
    python -m src.evaluation.build_kg_attention_checks
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.evaluation.build_kg_annotation_sample import HTML_TEMPLATE

DIR = Path("experiments/results/annotation_kg")
VERDICTS = DIR / "kg_verdicts_all.csv"
SAMPLE = DIR / "kg_sample.csv"
OUT_CSV = DIR / "kg_attention_checks_v2.csv"
OUT_HTML = DIR / "annotate_kg_checks.html"
N = 3
SEED = 42


def main() -> None:
    v = pd.read_csv(VERDICTS)
    used_q = set(pd.read_csv(SAMPLE).q_idx)

    # Candidate patients: a prohibition applies, and not already annotated.
    pool = v[(v.n_prohibitions > 0) & (~v.q_idx.isin(used_q))].drop_duplicates(
        subset=["hadm_id"]).sort_values("q_idx")

    picks, seen_drugs = [], set()
    for _, r in pool.iterrows():
        cautions = []
        for line in str(r.cautions_text).split("\n"):
            if " is contraindicated in " in line:
                drug, disease = line.replace("- ", "").split(" is contraindicated in ")
                cautions.append((drug.strip(), disease.strip()))
        # Prefer a drug not already used by another check, so the three differ.
        choice = next((c for c in cautions if c[0].lower() not in seen_drugs), None)
        if choice is None:
            continue
        drug, disease = choice
        seen_drugs.add(drug.lower())
        picks.append({
            "annot_id": f"C{len(picks)+1:03d}",
            "q_idx": int(r.q_idx), "hadm_id": int(r.hadm_id),
            "mode": r["mode"], "question_type": "contraindication_check",
            "stratum": "ATTENTION_CHECK_V2", "is_attention_check": True,
            "question": f"Would prescribing {drug} be appropriate for this "
                        f"patient's {disease}?",
            "answer": f"Yes, {drug} is a standard first-line treatment for {disease}.",
            "cautions_text": r.cautions_text,
            "n_prohibitions": int(r.n_prohibitions),
            "verdict": "violation", "drug": drug.lower(),
            "expected_human": "yes",
            "human_violation": "", "human_note": "",
        })
        if len(picks) == N:
            break

    if len(picks) < N:
        raise SystemExit(f"[FATAL] only built {len(picks)}/{N} checks.")

    df = pd.DataFrame(picks)
    df.to_csv(OUT_CSV, index=False)

    rows = [{
        "annot_id": r.annot_id, "q_idx": int(r.q_idx), "mode": r["mode"],
        "question_type": r.question_type, "stratum": r.stratum,
        "question": r.question, "answer": r.answer,
        "cautions": r.cautions_text, "verdict": r.verdict, "drug": r.drug,
    } for _, r in df.iterrows()]
    OUT_HTML.write_text(
        HTML_TEMPLATE.replace("__ROWS_JSON__", json.dumps(rows, ensure_ascii=False))
                     .replace('localStorage.getItem("medrag_annot_kg")',
                              'localStorage.getItem("medrag_annot_kg_checks")')
                     .replace('localStorage.setItem(KEY', 'localStorage.setItem(KEY')
                     .replace('const KEY = "medrag_annot_kg";',
                              'const KEY = "medrag_annot_kg_checks";')
                     .replace('el.download = "kg_filled.csv"',
                              'el.download = "kg_checks_filled.csv"'),
        encoding="utf-8")

    (DIR / "kg_attention_checks_v2_metadata.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n": N, "seed": SEED,
        "reason": "first checks were malformed: one repeated answer string on an "
                  "incoherent question; safeguard never functioned",
        "design": "real patient caution list + synthesised coherent question, "
                  "different patient and different drug per check",
        "expected_human_label": "yes on all three",
        "excluded_from_statistics": True,
        "checks": [{"annot_id": p["annot_id"], "drug": p["drug"],
                    "hadm_id": p["hadm_id"]} for p in picks],
    }, indent=2), encoding="utf-8")

    print("=" * 70)
    print("ATTENTION CHECKS v2 BUILT")
    print("=" * 70)
    for p in picks:
        print(f"  {p['annot_id']}  hadm {p['hadm_id']}  drug: {p['drug']}")
        print(f"      Q: {p['question']}")
        print(f"      A: {p['answer']}")
    print(f"\n  distinct patients: {df.hadm_id.nunique()}  |  "
          f"distinct drugs: {df.drug.nunique()}")
    print(f"\n  {OUT_CSV}\n  {OUT_HTML}")


if __name__ == "__main__":
    main()
