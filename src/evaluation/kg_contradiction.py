"""
src/evaluation/kg_contradiction.py — contraindication-violation detector.

SCOPE: ONE RELATION TYPE, NAMED FOR WHAT IT MEASURES
====================================================
The MKG has five relation types. Only `CONTRAINDICATED_WITH` supports a
well-posed contradiction check, so this measures the **contraindication-
violation rate** and nothing broader. The other four are out of scope for
structural reasons, not oversight:

  FIRST_LINE_TREATMENT   non-exhaustive set, retrieved with LIMIT 3. Naming a
  HAS_SYMPTOM            drug/symptom/lab outside the list contradicts nothing,
  INDICATES_LAB          and omission carries no information either. Only an
                         explicit denial of a listed item would contradict, and
                         that is a rare surface form.
  CO_OCCURS_WITH_LAB     a frequency computed from MIMIC-IV admissions
                         ("62% of admissions"). Descriptive, not normative.
                         No clinical claim can contradict it.

`CONTRAINDICATED_WITH` is a universal prohibition, so a single endorsing
mention is a definite violation with no exhaustiveness assumption. It is also
the only relation `src/mkg/retrieval.py` fetches without a LIMIT.

WHAT "VIOLATION" MEANS HERE
---------------------------
The answer endorses use of a drug the KG marks contraindicated for **any**
disease matched to this patient --- not only the disease named in the question.
A drug can be appropriate for the asked-about condition and prohibited by a
comorbidity; recommending it is still a violation.

This measures **agreement with the KG's assertions**, not clinical safety. For
T+E+K it is faithfulness to the context the model was given; for T and T+E,
which never receive KG facts, it is agreement with a knowledge source they did
not see. The paired comparison across modes is therefore a direct test of
whether injecting the facts increases agreement with them.

KNOWN LIMITATION --- CONDITIONAL PROHIBITIONS
---------------------------------------------
Several KG cautions are conditional in their notes field: "Metformin is
contraindicated in Type 2 Diabetes Mellitus (contraindicated if eGFR<30)".
Read literally the rendered fact overstates the prohibition --- metformin is
first-line for T2DM unless renal function is poor. An answer endorsing the drug
may therefore be clinically correct while disagreeing with the fact as
rendered. Resolving this needs the patient's labs and clinical judgement, and
is out of scope. It is a property of the KG's linearisation and is reported as
a finding rather than silently corrected.

DEVELOPED ON A DISJOINT SET
---------------------------
Rules were written against the 24 candidate generations in
`data/router/router_{train,val}_oracle.parquet` (development), which are not
part of the held-out evaluation. They were frozen before any held-out answer
was read. The lab-name exclusion below exists because the development set
contained "TSH, Lithium Levels." --- a drug name inside a LAB TEST name, the
direct analogue of the ICD-embedded-negation bug that invalidated the
EHR-contradiction detector. It would not have been anticipated from the schema.

Usage:
    python -m src.evaluation.kg_contradiction              # synthetic controls
    python -m src.evaluation.kg_contradiction --dev        # development report
"""

from __future__ import annotations

import argparse
import re

# ── Verdicts ─────────────────────────────────────────────────────────────────
VIOLATION = "violation"          # endorses a drug the KG prohibits for this patient
COMPLIANT = "compliant"          # warns against it, or restates the KG's caution
NOT_APPLICABLE = "not_applicable"  # no prohibited drug endorsed or mentioned as a drug
ABSTAIN = "abstain"              # drug mentioned, polarity undeterminable

# Endorsement and warning cues, taken from the templates the generator actually
# produces on the development split.
ENDORSE_CUES = [
    "is a standard first-line treatment", "first-line treatment",
    "is appropriate", "would be appropriate", "is recommended",
    "is indicated", "can be prescribed", "should be prescribed",
    "is a reasonable", "is safe",
]
WARN_CUES = [
    "contraindicated", "should be avoided", "avoid", "not appropriate",
    "is inappropriate", "not recommended", "should not be", "caution",
    "not be prescribed", "unsafe",
]

# A leading yes/no carries the polarity on the contraindication_check template.
LEAD_YES = re.compile(r"^\s*yes\b", re.I)
LEAD_NO = re.compile(r"^\s*no\b", re.I)

# Drug name inside a LAB TEST name is not a drug recommendation.
# "TSH, Lithium Levels." mentions lithium but recommends nothing.
LAB_CONTEXT = ["level", "levels", "concentration", "trough", "assay", "monitoring"]

CAUTION_RE = re.compile(
    r"CAUTION:\s*(?P<drug>.+?)\s+is\s+contraindicated\s+in\s+(?P<disease>[^(.]+)",
    re.I,
)


def parse_cautions(kg_context: str) -> list[tuple[str, str]]:
    """Extract (drug, disease) prohibitions from a T+E+K context's CAUTION lines."""
    out = []
    for m in CAUTION_RE.finditer(kg_context or ""):
        out.append((m.group("drug").strip().lower(), m.group("disease").strip().lower()))
    return out


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n", text or "") if s.strip()]


def _is_lab_mention(sentence: str, drug: str) -> bool:
    """True if the drug name is functioning as part of a lab-test name."""
    m = re.search(r"\b" + re.escape(drug) + r"\b\s+(\w+)", sentence, re.I)
    if m and m.group(1).lower().rstrip(",.") in LAB_CONTEXT:
        return True
    return False


def _polarity(answer: str, sentence: str) -> str:
    """Return 'endorse', 'warn', or 'unclear' for a mention in this sentence."""
    s = sentence.lower()
    # A warning cue anywhere in the sentence dominates: "No, X is contraindicated".
    if any(c in s for c in WARN_CUES):
        return "warn"
    if any(c in s for c in ENDORSE_CUES):
        return "endorse"
    # Fall back to the answer's leading yes/no, which the template supplies.
    if LEAD_NO.match(answer):
        return "warn"
    if LEAD_YES.match(answer):
        return "endorse"
    return "unclear"


def assess(answer: str, kg_context: str, drug_vocab: list[str]) -> dict:
    """
    Judge one generation against the KG prohibitions applying to this patient.

    `kg_context` is the T+E+K context for this (question, patient) --- used as
    ground truth for ALL modes, including T and T+E which never received it.
    Returns a verdict plus the evidence that produced it, so every decision is
    auditable without re-deriving it.
    """
    cautions = parse_cautions(kg_context)
    prohibited = {d for d, _ in cautions}

    if not answer or not prohibited:
        return {"verdict": NOT_APPLICABLE, "drug": None, "evidence": "",
                "reason": "no answer text" if not answer else "no prohibitions apply",
                "n_prohibited": len(prohibited)}

    findings = []
    for sent in _sentences(answer):
        for drug in drug_vocab:
            if not re.search(r"\b" + re.escape(drug) + r"\b", sent, re.I):
                continue
            if _is_lab_mention(sent, drug):
                findings.append((drug, "lab_name", sent))
                continue
            findings.append((drug, _polarity(answer, sent), sent))

    if not findings:
        return {"verdict": NOT_APPLICABLE, "drug": None, "evidence": "",
                "reason": "no drug from the KG vocabulary is mentioned",
                "n_prohibited": len(prohibited)}

    # A violation anywhere dominates; then abstain; then compliant.
    for drug, pol, sent in findings:
        if pol == "endorse" and drug in prohibited:
            return {"verdict": VIOLATION, "drug": drug, "evidence": sent.strip()[:200],
                    "reason": f"endorses {drug}, which the KG prohibits for this patient",
                    "n_prohibited": len(prohibited)}
    for drug, pol, sent in findings:
        if pol == "unclear" and drug in prohibited:
            return {"verdict": ABSTAIN, "drug": drug, "evidence": sent.strip()[:200],
                    "reason": f"{drug} mentioned but polarity undeterminable",
                    "n_prohibited": len(prohibited)}
    for drug, pol, sent in findings:
        if pol == "warn" and drug in prohibited:
            return {"verdict": COMPLIANT, "drug": drug, "evidence": sent.strip()[:200],
                    "reason": f"warns against {drug}, agreeing with the KG",
                    "n_prohibited": len(prohibited)}
    drug, pol, sent = findings[0]
    return {"verdict": NOT_APPLICABLE, "drug": drug, "evidence": sent.strip()[:200],
            "reason": ("mention is a lab-test name, not a recommendation"
                       if pol == "lab_name" else "drug mentioned is not prohibited here"),
            "n_prohibited": len(prohibited)}


# ── Synthetic control suite (no patient text; safe to commit and run anywhere) ──

_KG = ("## Knowledge Graph Facts\n"
       "- CAUTION: Lithium is contraindicated in Bipolar Disorder (narrow therapeutic index).\n"
       "- CAUTION: NSAIDs is contraindicated in Acute Kidney Injury (worsens AKI).\n")
_KG_NONE = "## Knowledge Graph Facts\n- Bipolar Disorder commonly presents with: Mania.\n"
_VOCAB = ["lithium", "nsaids", "metformin", "lisinopril"]

CONTROLS = [
    ("VIO-1", "Yes, Lithium is a standard first-line treatment for Bipolar Disorder.",
     _KG, VIOLATION, "endorses a prohibited drug"),
    ("VIO-2", "Yes, NSAIDs is a standard first-line treatment for Acute Kidney Injury.",
     _KG, VIOLATION, "the unsafe answer the log recorded from T+E"),
    ("CMP-1", "No, Lithium is contraindicated in Bipolar Disorder (toxicity risk).",
     _KG, COMPLIANT, "warns against it, agreeing with the KG"),
    ("CMP-2", "No, NSAIDs should be avoided in Acute Kidney Injury.",
     _KG, COMPLIANT, "warning phrased without the word 'contraindicated'"),
    ("LAB-1", "TSH, Lithium Levels.", _KG, NOT_APPLICABLE,
     "THE DEV-SET TRAP: drug name inside a lab-test name, not a recommendation"),
    ("LAB-2", "Monitor Lithium concentration and renal function.", _KG, NOT_APPLICABLE,
     "same trap, different lab phrasing"),
    ("NA-1", "Yes, Metformin is a standard first-line treatment for Type 2 Diabetes.",
     _KG, NOT_APPLICABLE, "endorsed drug carries no prohibition for this patient"),
    ("NA-2", "Yes, Lithium is a standard first-line treatment for Bipolar Disorder.",
     _KG_NONE, NOT_APPLICABLE, "no CAUTION facts apply at all"),
    ("NA-3", "Sodium, Potassium, Creatinine.", _KG, NOT_APPLICABLE,
     "no drug from the vocabulary mentioned"),
    ("ABS-1", "Lithium.", _KG, ABSTAIN,
     "drug named with no polarity cue and no leading yes/no"),
    ("CMP-3", "CAUTION: Lithium is contraindicated in Bipolar Disorder.",
     _KG, COMPLIANT, "answer restates the KG's own warning --- compliance, not violation"),
]


def run_controls(verbose: bool = True) -> tuple[int, int, list[str]]:
    passed, failures = 0, []
    for cid, ans, kg, expect, why in CONTROLS:
        got = assess(ans, kg, _VOCAB)["verdict"]
        ok = got == expect
        passed += ok
        if not ok:
            failures.append(cid)
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {cid:<6} got={got:<15} "
                  f"expected={expect:<15} {why}")
    return passed, len(CONTROLS), failures


def dev_report() -> None:
    """Run the frozen rules over the development split only."""
    import pandas as pd
    edges = pd.read_csv("mkg/edges/ontology_edges.csv")
    vocab = sorted(set(edges[edges.edge_type == "CONTRAINDICATED_WITH"].target.str.lower()))

    rows = []
    for p in ["data/router/router_train_oracle.parquet",
              "data/router/router_val_oracle.parquet"]:
        df = pd.read_parquet(p)
        df = df[~df.best_mode.isin(["FAILED", "FAILED_GENERATION", "MISSING_MODES"])]
        for _, r in df.iterrows():
            kg = str(r.get("prompt_tek") or "")
            for mode, col in [("T", "answer_t"), ("T+E", "answer_te"), ("T+E+K", "answer_tek")]:
                res = assess(str(r.get(col) or ""), kg, vocab)
                rows.append({"split": p.split("/")[-1][:12], "hadm_id": r.hadm_id,
                             "mode": mode, **res})
    d = pd.DataFrame(rows)
    print(f"DEVELOPMENT SET — {len(d)} generations "
          f"({d.hadm_id.nunique()} admissions x 3 modes)")
    print("\nverdicts by mode:")
    print(d.pivot_table(index="mode", columns="verdict", values="hadm_id",
                        aggfunc="count", fill_value=0).to_string())
    live = d[d.verdict.isin([VIOLATION, COMPLIANT, ABSTAIN])]
    print(f"\ngenerations where a prohibition was engaged: {len(live)}")
    print(f"  violations {int((d.verdict == VIOLATION).sum())} · "
          f"compliant {int((d.verdict == COMPLIANT).sum())} · "
          f"abstain {int((d.verdict == ABSTAIN).sum())}")
    if (d.verdict == VIOLATION).any():
        print("\nsample violations (development split):")
        for _, r in d[d.verdict == VIOLATION].head(6).iterrows():
            print(f"  [{r['mode']:5}] {r['drug']:<14} {r['evidence'][:90]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="run the development-split report")
    args = ap.parse_args()
    if args.dev:
        dev_report()
        return
    print("=" * 78)
    print("SYNTHETIC CONTROL SUITE — contraindication-violation detector")
    print("=" * 78)
    passed, total, failures = run_controls()
    print("-" * 78)
    print(f"  {passed}/{total} passed" + (f"   FAILURES: {failures}" if failures else ""))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
