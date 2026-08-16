"""
src/evaluation/fix_ehr_contradiction.py — corrected negation-contradiction scorer.

WHY THIS EXISTS
===============
The 2026-08-16 human annotation study found `ehr_contradiction_score()` has
Cohen's kappa = -0.0370 against human judgement, with ZERO true positives:
0 TP / 41 TN / 7 FP / 1 FN on 49 rows. All 7 false positives were triggered by
negation embedded in the ICD description itself ("Spondylosis WITHOUT
myelopathy", "Type 2 diabetes WITHOUT complications", "not carried"). Across
the 2,025 rows never annotated, 172 of the detector's 179 positives (96%) fall
on `diagnoses`/`primary_diagnosis` rows — the two question types whose gold
answer is a verbatim diagnosis string. The failure is systematic.

THE DEFECT
----------
The detector's premise is sound: "the EHR asserts X, the answer denies X".
It only ever checked the second half. It asked whether the answer contains
"<negation> X" for any X on a diagnosis/lab line, and never asked whether that
negation is the RECORD'S OWN WORDING. An answer that correctly copies the
patient's diagnosis is therefore scored as contradicting it.

THE FIX
-------
Fire only when the EHR POSITIVELY ASSERTS the term. If every mention of the
term in the EHR is itself negated, there is nothing to contradict.

    EHR "Spondylosis without myelopathy" + answer "Spondylosis without
        myelopathy"  -> silent (EHR never asserts myelopathy)
    EHR "Spondylosis without myelopathy" + answer "No myelopathy"
                     -> silent (still nothing asserted to contradict)
    EHR "Type 2 diabetes mellitus"       + answer "No diabetes"
                     -> FLAGS (genuine contradiction, still caught)

Plus two mechanical tightenings: candidate terms are taken from the VALUE side
of "Diagnoses: ..." rather than the whole line, and matching is word-boundary
rather than substring, so "not " cannot match inside "cannot".

DELIBERATELY NOT DONE
---------------------
No stopword list built from the 7 observed triggers ("myelopathy", "lesion",
"carried"). That would fit the fix to the very rows that exposed the bug,
which is the circularity this whole exercise exists to avoid. The rule is
general.

SCOPE — READ THIS BEFORE REPORTING THE METRIC
---------------------------------------------
Two distinct contradiction types exist:

  Type 1  answer NEGATES what the EHR ASSERTS.   Lexically detectable.
          This is what this scorer measures, correctly.
  Type 2  answer ASSERTS what the EHR REFUTES.   Requires clinical reasoning.
          NOT detectable by any string heuristic, and NOT measured here.

The one human-flagged contradiction the old detector missed (row A065,
"Decreased urine output, Anemia, Fatigue.") is Type 2 — it contains no
negation at all. This scorer does not catch it and cannot.

So the repaired metric is CORRECT but NARROWER than the name
"EHR-contradiction" implies. It must be reported as
**negation-contradiction rate**, with Type 2 stated as out of scope. Do not
let it quietly reclaim the broader name.

Usage:
    python -m src.evaluation.fix_ehr_contradiction     # runs the control suite
"""

from __future__ import annotations

import re

# Same five cues as the original scorer. Deliberately not extended ("denied",
# "negative", "ruled out"), so the only behavioural change is the assertion
# requirement, not a wider net.
NEG_CUES = {"no", "not", "denies", "without", "absent"}

# Negation scope stops at clause boundaries, so "No fever. Diabetes present."
# does not read as negating diabetes.
BOUNDARY = {",", ".", ";", ":", "(", ")", "/", "-"}

# How many tokens back to look for a cue, within the clause. Covers
# "without residual myelopathy" without spanning unrelated clauses.
SCOPE_WINDOW = 4

MIN_TERM_LEN = 5          # same threshold as the original (len(term) > 5)
PENALTY_PER_TERM = 0.2    # same increment as the original
MAX_PENALTY = 1.0

_TOKEN_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _negated_in_scope(tokens: list[str], idx: int) -> bool:
    """True if a negation cue governs tokens[idx] within its clause."""
    j, steps = idx - 1, 0
    while j >= 0 and steps < SCOPE_WINDOW:
        t = tokens[j]
        if t in BOUNDARY:
            break
        if t in NEG_CUES:
            return True
        j -= 1
        steps += 1
    return False


def _answer_negates(answer_tokens: list[str], term: str) -> bool:
    """Answer contains '<cue> term' adjacently — same semantics as the original."""
    for i, t in enumerate(answer_tokens):
        if t == term and i > 0 and answer_tokens[i - 1] in NEG_CUES:
            return True
    return False


def _ehr_asserts(ehr_tokens: list[str], term: str) -> bool:
    """True if at least one EHR mention of the term is NOT negated. THE FIX."""
    for i, t in enumerate(ehr_tokens):
        if t == term and not _negated_in_scope(ehr_tokens, i):
            return True
    return False


def negation_contradiction_score(answer: str, ehr_context: str) -> float:
    """
    Corrected scorer. Fires only when the answer negates a term the EHR
    positively asserts. Returns 0.0-1.0; labels use score > 0.

    Unlike the original this counts each distinct term once rather than once
    per occurrence, so the continuous value differs. Only the > 0 binarization
    is used for labelling, which is unaffected.
    """
    if not ehr_context or not answer:
        return 0.0

    ans_tokens = _tokens(answer)
    ehr_tokens = _tokens(ehr_context)

    penalty, counted = 0.0, set()
    for line in ehr_context.lower().split("\n"):
        if "diagnos" not in line and "lab" not in line:
            continue
        # Value side only: "Diagnoses: X, Y" -> "X, Y". Keeps the field label
        # itself out of the candidate set.
        values = line.split(":", 1)[1] if ":" in line else line
        for term in set(_tokens(values)):
            if len(term) <= MIN_TERM_LEN or not term.isalnum() or term in counted:
                continue
            if not _answer_negates(ans_tokens, term):
                continue
            if not _ehr_asserts(ehr_tokens, term):
                continue                      # EHR never asserts it -> not a contradiction
            counted.add(term)
            penalty = min(penalty + PENALTY_PER_TERM, MAX_PENALTY)
    return penalty


def ehr_contradiction_score_v1(answer: str, ehr_context: str) -> float:
    """The ORIGINAL scorer, copied verbatim from run_evaluation.py for comparison."""
    if not ehr_context or not answer:
        return 0.0
    ans_lower = answer.lower()
    ehr_lower = ehr_context.lower()
    penalty = 0.0
    negations = ["no ", "not ", "denies ", "without ", "absent "]
    for line in ehr_lower.split("\n"):
        if "diagnos" in line or "lab" in line:
            for term in line.split():
                if len(term) > 5:
                    for neg in negations:
                        if neg + term in ans_lower:
                            penalty = min(penalty + 0.2, 1.0)
    return penalty


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic control suite
# ══════════════════════════════════════════════════════════════════════════════
# All text here is SYNTHETIC — no MIMIC-IV content — so this suite is safe to
# commit and can run anywhere. It tests the two directions human annotation
# cannot: that genuine contradictions still fire, and that the specific bug
# stays fixed. These are sensitivity checks; they say NOTHING about how often
# contradictions occur in real data.

_EHR_ASSERTS_DM = "## EHR Snapshot\nDiagnoses: Type 2 diabetes mellitus, Essential hypertension\nLabs: Glucose 180"
# NOTE ON FIXTURE ORDERING: the negated term must NOT be followed by a comma.
# The original scorer takes terms via line.split(), so a trailing comma makes
# the term "myelopathy," and it then searches for "without myelopathy," — which
# an answer ending in a period never matches. The original would then be silent
# for a punctuation reason rather than the reason under test, and the
# regression case would prove nothing. Putting the term last reproduces the
# real observed failure.
_EHR_NEG_MYELO = "## EHR Snapshot\nDiagnoses: Cervical disc disorder, Spondylosis without myelopathy\nLabs: Creatinine 1.1"
_EHR_NEG_COMPL = "## EHR Snapshot\nDiagnoses: Type 2 diabetes without complications\nLabs: Glucose 150"
_EHR_ASSERTS_PNA = "## EHR Snapshot\nDiagnoses: Pneumonia, organism unspecified\nLabs: WBC 14.2"

CONTROLS: list[tuple[str, str, str, bool, str]] = [
    # (id, answer, ehr_context, should_fire, rationale)
    ("POS-1", "No diabetes.", _EHR_ASSERTS_DM, True,
     "EHR asserts diabetes; answer denies it — genuine Type-1 contradiction"),
    ("POS-2", "The patient is without hypertension.", _EHR_ASSERTS_DM, True,
     "EHR asserts hypertension; answer negates it"),
    ("POS-3", "No pneumonia.", _EHR_ASSERTS_PNA, True,
     "EHR asserts pneumonia; answer denies it"),

    ("REG-1", "Spondylosis without myelopathy.", _EHR_NEG_MYELO, False,
     "THE BUG: answer copies the ICD name verbatim — must not fire"),
    ("REG-2", "No myelopathy.", _EHR_NEG_MYELO, False,
     "harder case: different wording, but EHR still never asserts myelopathy"),
    ("REG-3", "Type 2 diabetes without complications.", _EHR_NEG_COMPL, False,
     "THE BUG: 'without complications' is the record's own phrasing"),

    ("NEG-1", "Type 2 diabetes mellitus.", _EHR_ASSERTS_DM, False,
     "answer agrees with the EHR, no negation present"),
    ("NEG-2", "Glucose is elevated at 180.", _EHR_ASSERTS_DM, False,
     "no negation of any asserted term"),
    ("NEG-3", "", _EHR_ASSERTS_DM, False, "empty answer"),
    ("NEG-4", "No diabetes.", "", False, "empty context"),

    # Documented boundary, not a defect: answer-side matching is adjacency-only,
    # inherited unchanged from the original. Widening it would be a separate
    # change that this study is not designed to validate.
    ("BOUND-1", "The patient does not have diabetes.", _EHR_ASSERTS_DM, False,
     "KNOWN MISS: 'not have diabetes' is not adjacent — documented limitation"),
]


def run_controls(verbose: bool = True) -> tuple[int, int, list[str]]:
    """Run the synthetic suite. Returns (passed, total, failures)."""
    passed, failures = 0, []
    for cid, answer, ehr, should_fire, why in CONTROLS:
        score = negation_contradiction_score(answer, ehr)
        fired = score > 0
        ok = fired == should_fire
        v1 = ehr_contradiction_score_v1(answer, ehr)
        if ok:
            passed += 1
        else:
            failures.append(cid)
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {cid:<8} "
                  f"corrected={'FIRE' if fired else 'silent':<6} "
                  f"expected={'FIRE' if should_fire else 'silent':<6} "
                  f"(original={'FIRE' if v1 > 0 else 'silent'})   {why}")
    return passed, len(CONTROLS), failures


def main() -> None:
    print("=" * 78)
    print("SYNTHETIC CONTROL SUITE — corrected negation-contradiction scorer")
    print("=" * 78)
    passed, total, failures = run_controls()
    print("-" * 78)
    print(f"  {passed}/{total} passed" + (f"   FAILURES: {failures}" if failures else ""))

    # The regression cases are the ones the original gets wrong; show the delta
    # explicitly so the fix's effect is visible rather than asserted.
    print("\n  Cases where original and corrected DISAGREE:")
    for cid, answer, ehr, should_fire, why in CONTROLS:
        v1 = ehr_contradiction_score_v1(answer, ehr) > 0
        v2 = negation_contradiction_score(answer, ehr) > 0
        if v1 != v2:
            print(f"    {cid:<8} original={'FIRE' if v1 else 'silent':<6} "
                  f"corrected={'FIRE' if v2 else 'silent':<6} -> "
                  f"{'FIXED' if v2 == should_fire else 'REGRESSION'}")
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
