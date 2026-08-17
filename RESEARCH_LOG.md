# Research Log — Adaptive RAG Router for EHR-Grounded Clinical QA

This log records the research history of the project: investigations, hypotheses,
discovered bugs, experiments, code changes, rollbacks, and conclusions. It is
maintained continuously as work progresses so the full history is reconstructable
without needing to read commit-by-commit diffs.

---

## 2026-08-17 — KG-contradiction check (Step 4): the KG *repairs* harm the EHR
## snapshot causes. First direct evidence that graph injection improves safety.

Closes the Limitations gap "we do not implement a KG-contradiction measure".
Detector: `src/evaluation/kg_contradiction.py`, frozen in commit `b282957`
**before any held-out answer was read**.

### Scope: one relation type of five, named for what it measures

Only `CONTRAINDICATED_WITH` supports a well-posed check, so this is the
**contraindication-violation rate**, not a "KG-contradiction rate". The others
are out of scope structurally, not by oversight:

| relation | why not |
|---|---|
| `FIRST_LINE_TREATMENT`, `HAS_SYMPTOM`, `INDICATES_LAB` | non-exhaustive lists retrieved with `LIMIT 3`; naming an item outside the list contradicts nothing, and omission carries no information |
| `CO_OCCURS_WITH_LAB` | a MIMIC-IV frequency ("62% of admissions") — descriptive, not normative; no clinical claim can contradict it |

`CONTRAINDICATED_WITH` is a universal prohibition, so one endorsing mention is
a definite violation. It is also the only relation `retrieval.py` fetches
without a `LIMIT`.

**Violation** = the answer endorses a drug the KG marks contraindicated for
**any** disease matched to this patient, not only the disease in the question.
The T+E+K context supplies ground truth for **all three modes**, so T and T+E
are judged against facts they never received — which is what makes the paired
comparison a test of whether injecting the facts changes behaviour.

### Development on a disjoint set caught a failure the schema would not predict

Rules were written against the 24 candidate generations in the router splits,
never the eval set. The dev set immediately produced
**`"TSH, Lithium Levels."`** — a prohibited drug named inside a **lab test
name**, endorsing nothing. That is the direct analogue of the ICD-embedded-
negation bug that invalidated the EHR-contradiction detector, in a form no
amount of schema reading would have surfaced. It is now an explicit exclusion
with two control cases. Synthetic suite: **11/11**.

### Validation: precision 1.0000

61 real rows + 3 attention checks, stratified over the detector's own verdicts.

| stratum | n | result |
|---|---|---|
| A — flagged violation | 41 | **41 genuine, 0 false positive** |
| B — flagged compliant | 10 | 10 confirmed non-violations |
| C — prohibition applies, detector silent | 10 | 9 confirmed silent, 1 disputed |

**Precision 1.0000**, exact 95% CI [0.9140, 1.0000] — clears the pre-registered
≥0.80 bar (41 flags ≥ the 5 required for a claim). **Zero abstains across all
900 generations.** Recall 1.0000 scoping out the single disputed row, 0.9762
counting it.

`K026` is **adjudicated as a KG coverage gap, not a detector miss**
(2026-08-17, project owner). The annotator judged an endorsement of
Empagliflozin a violation on the basis of that drug's real clinical renal risk
profile; the graph does not encode that contraindication for the patient
concerned, so the detector applied its pre-registered definition exactly.
**Headline recall stays 1.0000**; 0.9762 is retained only for transparency.

This is a concrete instance of the graph being incomplete relative to what a
clinician knows, and belongs with the linearisation defect below: one omits a
real contraindication, the other flattens a conditional one into an
unconditional claim. **Both are the KG failing to capture what a clinician
would bring to the same decision**, and both are worth reporting as properties
of the knowledge resource rather than of the routing system.

### THE RESULT — the KG's benefit is corrective, not additive

Paired McNemar over 300 questions, 212 engaging a prohibition:

| mode | violations |
|---|---|
| T | 12 |
| **T+E** | **18** |
| **T+E+K** | **11** |

| comparison | only first | only second | *p* |
|---|---|---|---|
| **T+E vs T+E+K** | **7** | **0** | **0.0156** |
| **T vs T+E** | 0 | **6** | **0.0312** |
| T vs T+E+K | 1 | 0 | 1.0000 |

Two findings, and the second is the one that matters:

1. **Injecting KG facts eliminates violations and introduces none** — 7 to 0
   against T+E, p = 0.0156. This is the first direct evidence in the project
   that the KG improves a safety outcome.
2. **The EHR snapshot alone makes safety WORSE** — T+E violates on 6 questions
   T does not, and none the other way (p = 0.0312). This independently
   reproduces the 2026-08-11 observation that the snapshot leads the model to
   affirm a drug it sees in the patient's medication list, now with a
   statistical test rather than a single example.
3. **T+E+K is NOT significantly better than plain T** (1 vs 0, p = 1.0). So the
   KG does not add safety over text-only retrieval — **it undoes the damage the
   EHR snapshot causes**. State it that way. "KG injection reduces
   contraindication violations" is true only relative to T+E, and reporting it
   without that qualifier would overclaim.

This sharpens H2 rather than contradicting it: the KG's value here is repairing
a specific failure mode introduced by structured EHR context, not general
improvement.

### Methodological finding: attention checks must be independently constructed

The first three attention checks **failed 0/3 — and it was a construction bug,
not inattention.** All three were the *same* answer string
("Yes, Metformin is a standard first-line treatment for CKD Stage 3") pasted
onto the question "What was the most likely main diagnosis?" — one check
repeated three times on an incoherent pairing, produced by `head(3)` over rows
that happened to share a patient and a caution. Answering "no" to a
non-sequitur that recommends nothing to anyone is defensible.

The annotator was demonstrably attentive on the real rows (41/41, 10/10,
9-no/1-yes on the stratum where a rubber-stamp would have produced ten uniform
"no"s), so the labels were kept and only the safeguard was rebuilt: three
different patients, three different drugs, real caution lists, questions
synthesised to be coherent with their answers. **Rebuilt checks: 3/3 caught.**

Rule to carry forward: **an attention check must be independently constructed
and internally coherent.** A malformed check does not measure attention, and
applying a pre-registered void rule to it would have discarded good work on the
strength of a bug. Verify what a check *contains*, not just that it exists —
the second time this project has been saved by that.

### Caveats

1. **One relation type of five.** Not a general KG-faithfulness measure.
2. **Conditional prohibitions.** Several cautions are conditional in their notes
   ("contraindicated if eGFR<30"), but `get_subgraph_facts()` flattens them into
   unconditional statements. Read literally the rendered fact can be wrong —
   metformin *is* first-line for T2DM unless renal function is poor — so an
   endorsement may be clinically correct while disagreeing with the fact as
   given. **This is a defect in the KG's linearisation and is worth reporting
   independently of this validation.**
3. This measures **agreement with the KG's assertions, not clinical safety**.
4. Violations are rare (11–18 per mode over 300 questions); the McNemar tests
   rest on 6–7 discordant pairs.
5. Single annotator, not inter-annotator agreement.

Artifacts: `experiments/results/annotation_kg/` — `kg_verdicts_all.csv` (all 900
generations) · `kg_sample.csv` · `kg_filled.csv` · `kg_checks_filled.csv` ·
`kg_results.json` · `kg_metadata.json` · `annotate_kg.html`.

---

## 2026-08-17 — Round 3: corrected scorer FAILS its pre-registered bar on
## `monitoring_labs`, and the original metric was under-detecting all along

23 rows (15 flagged + 5 control + 3 attention checks), `monitoring_labs`/`lab`
only, disjoint from both prior annotation rounds. Attention checks **3/3**
caught, so the labels are trustworthy.

### The pre-registered criterion FAILS. The bar is not being moved.

| | value |
|---|---|
| stratum A (flagged, n=15) | **11 genuine, 4 false positive** |
| **precision** | **0.7333**, 95% CI [0.4490, 0.9221] |
| recall | 0.9167 (11/12) |
| pre-registered bar | ≥ 12/15 (precision ≥ 0.80) |
| **result** | **NOT VALIDATED** — short by one row |

Locked 2026-08-17 before annotation; it fails by a single row and stays failed.
Round 2's diagnosis-question result (FP 20 → 0) is unaffected.

### But the comparison against the original inverts the expected story

Same 20 rows, original scorer: **TP 3 · FP 0 · FN 9** — precision 1.000,
**recall 0.25**. Corrected: TP 11 · FP 4 · FN 1 — precision 0.733,
**recall 0.917**.

On this population the original was not over-flagging at all. It was **missing
9 of 12 genuine contradictions**. The fix trades precision (1.00 → 0.73) for a
large recall gain (0.25 → 0.92). Of the 12 rows newly caught by the fix, **8
are genuine**.

**This changes what the original metric's near-zero rates meant.** It
over-flagged on `diagnoses`/`primary_diagnosis` (ICD-embedded negation, κ =
−0.037, 0 TP) *and* under-detected on `monitoring_labs`. The paper's reported
EHR-contradiction rates were not merely noisy — **the true contradiction rate
is higher than reported**, not lower. `monitoring_labs` is where this generator
actually contradicts the EHR: the human confirmed 12 genuine contradictions in
20 rows of that type, versus **0 in 30** diagnosis rows in round 2.

### All 4 false positives are ONE artifact

Every one is the same 4-word template answer — *"No abnormal labs available"* —
triggering on `no abnormal`, because "abnormal" (>5 chars) sits on an EHR lab
line and is positively asserted there. So precision 0.733 reflects a single
repeated answer template appearing 4 times, not four independent failure modes.

**The fix is obvious and must NOT be written yet.** "abnormal" is a generic
clinical qualifier, not a finding, so the rule would be to require the negated
term to be a clinical entity rather than a qualifier. But that fix would be
derived from these 4 observed rows, and validating it on them would be exactly
the circularity rounds 2 and 3 exist to avoid. If implemented it needs a
**fourth** fresh sample, disjoint from all three.

### Consequences for the paper

1. The `negation-contradiction rate` is validated for
   `diagnoses`/`primary_diagnosis` (round 2) and **only partially** for
   `monitoring_labs` (precision 0.73, CI [0.45, 0.92], n=15 — wide).
2. Any reported rate must carry that split. Do not present a single validated
   number across all question types.
3. The stronger, better-supported claim is the **qualitative** one: the
   original detector fails in *both* directions, and contradiction is
   concentrated in `monitoring_labs` — a question type the conference paper's
   metric was structurally blind to.
4. One stratum-B row (W005, a `lab` question) is a genuine contradiction both
   scorers miss — a reminder that recall is not 1.0 even after the fix.

Artifacts: `experiments/results/annotation_v3/` — `validation_sample.csv` ·
`validation_filled.csv` · `validation_v3_results.json` ·
`validation_v3_disagreements.csv` · `validation_metadata.json` ·
`annotate_v3.html`.

---

## 2026-08-16 (later still) — Corrected metric recomputed over all 2,100 rows.
## Two findings that change how it must be reported.

`python -m src.evaluation.recompute_contradiction`, 900 contexts rebuilt,
retrieval only, no GPU.

**Reproduction check: 0/2100 label flips** between the stored column and the
original scorer recomputed on rebuilt contexts. The rebuild reproduces eval
time exactly, so every number below is comparable to the paper's.

Effect of the fix on 1,462 scorable (mode ≠ T) rows:

| | flags |
|---|---|
| original scorer | 186 |
| **corrected scorer** | **46** |
| suppressed (v1 yes → v2 no) | 173 |
| **newly caught** (v1 no → v2 yes) | **33** |

Confirms the corrected scorer is not a subset: it removes 173 artifacts and
adds 33 detections the original's punctuation bug had hidden.

### FINDING 1 — the per-system table is CONFOUNDED. Do not report it as printed.

The raw output looks like a large Router win:

| system | n scorable | n/a (mode T) | rate (budgeted) |
|---|---|---|---|
| Router | 226 | 74 | 0.0044 |
| T+E+K | 300 | 0 | 0.0467 |
| T+E | 300 | 0 | 0.0600 |

**This is a selection artifact, not a result.** Rates are computed over
different question subsets: any system that routes to T has those rows dropped
as n/a, and the dropped rows are exactly where contradictions live.

| under | flags on the 74 questions Router sends to T | flags on the 226 it keeps | Fisher |
|---|---|---|---|
| T+E | **16 of 18** (rate 0.2162) | 2 (rate 0.0088) | p = 4.9e-09 |
| T+E+K | **13 of 14** (rate 0.1757) | 1 (rate 0.0044) | p = 6.0e-08 |

~90% of all contradictions sit in the 25% of questions the Router routes to T,
where the metric is **undefined rather than zero**. The Router does not avoid
contradictions; it routes them into unmeasurability.

Paired comparison on the 226 questions where Router is scorable — the honest
version:

| system | flagged | rate |
|---|---|---|
| Router | 1 | 0.0044 |
| **T+E+K** | **1** | **0.0044** |
| T+E | 2 | 0.0088 |
| Oracle | 1 | 0.0046 |
| StaticQType | 1 | 0.0049 |

**Router and T+E+K are tied.** The paper's original "EHR-contradiction is
effectively tied" conclusion survives — but for an entirely different reason,
on a metric that now works. Report the paired numbers; the unpaired table is
not interpretable.

**Generalised rule to carry forward:** the mode-T `n/a` rule creates a
SELECTION EFFECT for any system that chooses when to enter mode T. This is the
same defect family as reporting T's contradiction rate as 0.0000 (2026-08-15),
one level up: there the undefined value was mistaken for zero, here the
undefined *rows* silently change the denominator. Any per-system rate over a
mode-conditional metric must be computed on a matched question set.

Sanity check passed: Router rows are byte-identical to their base-mode system
on the same question (174 T+E rows, 52 T+E+K rows, exact match), so the
pipeline is consistent and the effect is genuinely distributional.

### FINDING 2 — 87% of the remaining flags are on question types never validated

| question type | v2 flags | validated by the human study? |
|---|---|---|
| monitoring_labs | 36 | **no** |
| lab | 4 | **no** |
| primary_diagnosis | 6 | yes |

**40 of 46 flags — and all 33 newly-caught rows — are `monitoring_labs`/`lab`.**
The 2026-08-16 validation covered `diagnoses`/`primary_diagnosis`, because that
is where the bug lived. It established that the fix eliminates false positives
there (20 → 0). It says **nothing** about whether the 33 new `monitoring_labs`
detections are genuine.

So the corrected metric's positive rate is driven almost entirely by an
unvalidated population. **Do not put the recomputed rate in the paper without a
third annotation round targeting `monitoring_labs`** (same design: fresh rows,
disjoint from both prior samples, blind, pre-registered). The validated claim
today is narrower than the recomputed table implies: *the fix removes the ICD
artifact on diagnosis questions*, not *the corrected rate is trustworthy
everywhere*.

Artifacts: `negation_contradiction_by_system.csv` ·
`negation_contradiction_per_question.csv` ·
`negation_contradiction_metadata.json`.

---

## 2026-08-16 (later) — EHR-contradiction detector FIXED and validated on unseen data

Repairs the defect found earlier the same day. The corrected scorer lives in
`src/evaluation/fix_ehr_contradiction.py`; `run_evaluation.py` is deliberately
**unchanged** so the paper's existing numbers stay reproducible from committed
code until the metric is formally replaced.

### The fix

The original asked "does the answer contain `<negation> X` for some X on a
diagnosis/lab line?" and never checked whether that negation was **the
record's own wording**. The corrected scorer fires only when the EHR
**positively asserts** the term — if every EHR mention of X is itself negated,
there is nothing to contradict. Plus two mechanical tightenings: candidate
terms come from the value side of `Diagnoses: ...`, and matching is
word-boundary rather than substring.

Deliberately **not** done: no stopword list built from the 7 observed triggers.
That would fit the fix to the rows that exposed the bug.

### A second, independent bug found while building the control suite

The original attaches punctuation to terms (`line.split()` yields
`"pneumonia,"`), so for an EHR listing "Pneumonia, organism unspecified" it
searched for the literal `"no pneumonia,"` and went **silent** on an answer
saying "No pneumonia." — a genuine contradiction. Same class as the
2026-08-13 KG matcher punctuation bug. Consequence: **the corrected scorer is
NOT a strict subset of the original.** It removes artifact positives *and*
adds genuine ones. Do not assume the recomputed rate can only fall.

### Validation — deliberately NOT on the rows that found the bug

The 75 annotated rows exposed the defect, so reusing them would be circular.
A disjoint sample was drawn from the 2,025 rows never annotated:

| stratum | n | drawn from |
|---|---|---|
| A — originally flagged | 20 | unseen `diagnoses`/`primary_diagnosis` the original flags |
| B — control | 10 | unseen rows the original does not flag, same question types |
| attention checks | 3 | real contexts, answer perturbed into a genuine contradiction |

Overlap with session 1: **0**, enforced on `q_idx|system`. All rows mode ≠ T.

| detector | TP | TN | **FP** | FN | accuracy |
|---|---|---|---|---|---|
| original | 0 | 10 | **20** | 0 | 0.3333 |
| **corrected** | 0 | **30** | **0** | 0 | **1.0000** |

**All 20 false positives eliminated, none introduced, none missed.** Paired
McNemar: original-only-correct 0, corrected-only-correct 20, **p = 2.0e-06**.
Stratum-A corrected FP rate 0/20, 95% CI [0.0000, 0.1684].

κ is *not* the headline here: the sample is enriched for originally-flagged
rows, so its prevalence is not the eval set's and κ is unstable at near-zero
positives. The paired McNemar is the correct test.

**The attention checks are what make this falsifiable.** If the fix works the
expected result is 30 consecutive "no" answers — indistinguishable from an
annotator who stopped reading. The annotator caught **3/3** constructed
contradictions (and the corrected scorer flagged 3/3), so the uniform "no"
reflects the data. Without them this result would prove nothing. They are
excluded from every statistic.

Synthetic control suite: **11/11 pass**, including three regression cases where
the original fires and the corrected is silent, and one where the original is
silent and the corrected correctly fires.

### SCOPE — the metric must be renamed

Two contradiction types exist:

- **Type 1** — answer negates what the EHR asserts. Lexically detectable.
  This is what the corrected scorer measures, correctly.
- **Type 2** — answer asserts what the EHR refutes. Requires clinical
  reasoning. **Not detectable by any string heuristic and not measured.**

The one human-flagged contradiction the old detector missed (row A065,
*"Decreased urine output, Anemia, Fatigue."*) is Type 2 — it contains no
negation at all.

So the repaired metric is **correct but narrower than the name implies**. It
must be reported as **negation-contradiction rate**, with Type 2 explicitly
out of scope. Do not let it quietly reclaim the broader "EHR-contradiction"
label — that would claim coverage the method does not have. (Project owner's
explicit decision, 2026-08-16.)

### Still to do

Recompute the corrected metric across all 2,100 rows and replace the paper
column. This needs **no GPU and no regeneration** — answers are stored and
context rebuild reproduces eval time exactly (0 drift, proven 2026-08-16) —
roughly 900 unique contexts, retrieval only. Report both budgeted (what the
model saw; what was validated) and unbudgeted (how the original column was
computed) variants, since they are not interchangeable.

Artifacts (gitignored): `experiments/results/annotation_v2/` —
`validation_sample.csv` · `validation_filled.csv` · `validation_summary.csv` ·
`validation_disagreements.csv` · `validation_results.json` ·
`validation_metadata.json` · `annotate_v2.html`.

---

## 2026-08-16 — Human annotation study: `unsupported_rate` validates, `ehr_contradiction` does NOT

First human evaluation in this project. 75 rows from the 300-question held-out
set, stratified across all 7 systems (seed 42), annotated by the project owner
blind to system identity and to the detector's labels (revealed only after
both answers were given on a row). Tooling:
`src/evaluation/build_annotation_sample.py` → `annotate.html` →
`src/evaluation/annotation_agreement.py`.

The unsupported threshold was **pre-registered at ≥ 0.5 on 2026-08-13, before
any annotation**, specifically so it could not be tuned against the human
labels. It was not changed. The 0.1–0.9 sweep below is robustness only.

### Result 1 — `unsupported_rate` is validated

| | n | human yes | detector yes | observed agr. | **Cohen's κ** | 95% CI |
|---|---|---|---|---|---|---|
| unsupported (≥0.5) | 75 | 29 (38.7%) | 30 (40.0%) | 0.8800 | **0.7486** | [0.5816, 0.8901] |

TP 25 · TN 41 · FP 5 · FN 4. McNemar p = 1.0 — **no directional skew**; the
detector errs symmetrically. κ = 0.75 is "substantial" and clears the project's
own pre-registered ≥ 0.6 bar.

Sweep: κ peaks at threshold 0.4 (0.7779) against 0.7486 at the pre-registered
0.5 — adjacent and close, so the locked choice was near-optimal by luck rather
than by tuning. **Do not retrofit 0.4.** Fitting the cut to the reference
standard it is being judged against would invalidate the exercise.

**Consequence:** H1 criterion 3 (the grounding failure) rests on a metric that
now has human validation. That result stands on firmer ground than before, not
weaker.

### Result 2 — `ehr_contradiction` is NOT measuring contradiction

| | n | human yes | detector yes | observed agr. | **Cohen's κ** | 95% CI |
|---|---|---|---|---|---|---|
| ehr_contradiction | 49 | 1 (2.0%) | 7 (14.3%) | 0.8367 | **−0.0370** | [−0.1011, 0.0000] |

TP **0** · TN 41 · FP 7 · FN 1. McNemar p = 0.070, false-positive skew.
κ is *negative* — agreement below chance. The 0.8367 observed agreement is
entirely the shared "no" majority and means nothing at this prevalence.
(26 mode-T rows excluded as n/a per the standing rule, not counted as
agreement.)

**Zero true positives. Every one of the detector's 7 positives was rejected by
the human, and the mechanism is unambiguous** — each was triggered by a
negation embedded in the ICD description itself:

| row | question type | trigger |
|---|---|---|
| A002 | diagnoses | `without lesion` |
| A009 | primary_diagnosis | `without myelopathy` |
| A034 | primary_diagnosis | `without lesion` |
| A046 | diagnoses | `not carried` |
| A062 | diagnoses | `not carried` |
| A069 | primary_diagnosis | `without hematuria` |
| A072 | primary_diagnosis | `without complications` |

`ehr_contradiction_score()` scans context lines containing "diagnos"/"lab",
takes any token longer than 5 characters, and flags the answer if it contains
`"no "/"not "/"without " + term`. ICD-10 descriptions routinely embed exactly
that phrasing ("Spondylosis **without myelopathy**", "Type 2 diabetes
**without complications**"). So an answer that *correctly copies the EHR's
diagnosis string* is flagged as contradicting the EHR. The detector fires on
ICD naming convention, and all 7 positives are on `diagnoses` /
`primary_diagnosis` rows — the two question types whose gold answer is a
verbatim diagnosis string.

**Consequence for the paper:** the EHR-contradiction column (T+E 0.0247,
T+E+K 0.0247, Router 0.0233) does not measure hallucination and must not be
reported as if it does. The associated H1 claim — "EHR-contradiction is
effectively tied (0.0233 vs 0.0247, p = 0.157)" — is a comparison between two
quantities that are both artifacts. Either withdraw the column or report it
explicitly as an unvalidated heuristic with κ ≈ 0 and 0 true positives.
This is now the third measurement defect found in this metric, after the
mode-T `n/a` problem (2026-08-15).

### Carried forward: context drift

Contexts are rebuilt because `run_evaluation.py` does not persist
`prompt_context`. Retrieval reproduction was **exact** (0 label flips, mean
|Δ| 0.000 vs eval time). Remaining drift is the deliberate choice to show the
annotator — and score the detector on — the *budgeted* context the model
actually received: 3/75 unsupported flips, 0 EHR flips, mean |Δ| 0.0531, all 3
on mode T. Human and detector judged identical evidence, so this does not
enter κ; it bounds how far these labels sit from the paper's own numbers.

### Honest caveats

1. **This is human-vs-detector agreement, NOT inter-annotator agreement.**
   One annotator. The design doc's "Cohen's κ ≥ 0.6 between two human
   annotators" is a different study and is still not done. Do not present this
   as satisfying it.
2. The annotator is the project owner — blind to system identity and detector
   labels, but not to the project's hypotheses.
3. **The EHR-contradiction κ is severely underpowered**: 1 human positive in
   49 rows. The negative κ is directionally trustworthy (TP = 0 with 7 FP and
   a fully explained mechanism), but the point estimate is unstable and the CI
   must be reported with it.
4. 75 rows over 7 systems is ~10–11 per system; this validates the metrics
   overall, not per system.

Artifacts (gitignored): `annotation_agreement.csv` ·
`annotation_confusion_matrix.csv` · `annotation_threshold_sweep.csv` ·
`annotation_disagreements.csv` (17 rows) · `sample_75.csv` ·
`sample_metadata.json` · `annotate.html`.

### Tooling defect found and fixed during this run

The first sample build produced **0/19 T+E+K rows with KG facts** (12 had them
at eval time) because the KG preflight only checked that `src.mkg.retrieval`
*imports* — which opens no connection. Neo4j auth was failing, `Retriever`
swallowed each failure, and contexts came back silently KG-less while the
script reported "0 failed rebuilds". Aggregated over 75 rows this looked like a
mild 6/75 drift; on the 19 affected rows unsupported nearly doubled
(0.215 → 0.402). Fixed with (a) a real preflight that executes a Cypher query,
(b) a hard integrity check that any T+E+K row with eval-time KG facts must
rebuild with them, and (c) per-mode drift reporting so a single-mode fault
cannot hide in an aggregate. The corrupt build is quarantined at
`experiments/results/annotation/_INVALID_no_kg_2026-08-13/`.

---

## 2026-08-13 — SHAP attribution on the deployed router (journal extension)

First work of the JBI journal extension. Adds feature-attribution evidence for
the routing behaviour that the feature ablation (Table VI) established only at
the performance level. **Nothing was retrained**; `models/router/` was opened
read-only, and the generator/QLoRA adapter were not touched.

`shap` was not installed (this is why `shap_summary.png` never existed despite
the dormant hook at `train_router.py:186`, and why the 2026-08-14 entry could
only conjecture about per-case attribution). Installed **shap 0.51.0**; numpy
2.4.6 / torch 2.5.1+cu121 / xgboost 3.2.0 unchanged afterwards.

New script: `src/evaluation/router_shap.py` (`python -m src.evaluation.router_shap`).

### Both tiers reproduced the deployed decision function before any attribution

Attributions over a feature vector the router never actually saw would be
worthless, so each tier is gated first:

| tier | set | gate | result |
|---|---|---|---|
| A | router_val, n=100 (the rows Table VI uses) | re-predicted probs vs `router_predictions.csv` | max abs delta **2.97e-08**, mode agreement **100%** |
| B | final held-out eval, n=300 (deployed decisions) | re-predicted mode vs `mode_used` | **300/300 = 100%**, 0 disagreements |

Tier B's 5 structural features were reconstructed via `PatientSnapshot` exactly
as `run_evaluation.py::get_struct_features` does; the perfect agreement confirms
the reconstruction is exact. SHAP additivity (base + Σφ vs raw margin) holds to
6.2e-07 (A) and 8.3e-07 (B).

### Block attribution CORROBORATES Table VI

Mean |SHAP| summed within block, pooled over classes:

| tier | embedding block (384) | patient block (5) | patient share |
|---|---|---|---|
| A (n=100) | 2.1008 | 0.2247 | **9.66%** |
| B (n=300) | 2.1169 | 0.2048 | **8.82%** |

Per class (tier A): T 85.6/14.4, T+E 90.0/10.0, T+E+K **95.3/4.7**. The router's
KG decision is the *least* patient-informed of the three — consistent with the
H2 mechanism (KG retrieval keys off the diagnosis list, so patient state cannot
inform whether the KG will help).

### But the per-FEATURE view is NOT the 98.4/1.6 story gain importance told

The block comparison sums 384 features against 5, which flatters the larger
block. Per feature, pooled:

| tier | avg embedding dim | avg patient feature | ratio |
|---|---|---|---|
| A | 0.005471 | 0.044941 | **8.21x** |
| B | 0.005513 | 0.040964 | **7.43x** |

Rank among all 389 features (tier B): **n_labs #5**, n_diag #14,
sparsity_score #17, sparsity_bucket #50 — all nonzero on 300/300 rows.
**n_meds is exactly 0.0 on every row and ranks #371**, independently confirming
the dead-feature caveat recorded on 2026-08-14 (caveat #4).

> ### ⚠ DO NOT CONFLATE THESE TWO CLAIMS
>
> Low **performance contribution** (Table VI: removing the patient features
> costs +0.0000 accuracy / −0.0181 macro-F1) and low **attribution** (SHAP:
> the patient block is ~9% of total |SHAP| mass) are *different measurements*
> and they do **not** agree here. Per feature, the patient features are 7–8x
> the average BGE dim; `n_labs` ranks **#5 of 389** and is nonzero on
> **300/300** held-out rows.
>
> - ✅ Correct: **"question-driven, with performance-neutral use of patient
>   state"** — the router consults patient features; consulting them does not
>   improve routing.
> - ❌ Wrong: "the router ignores patient state" / "patient features are
>   unused" / "attribution confirms the ablation." The attribution result is
>   *compatible* with the ablation, not a restatement of it.
>
> The collapse is easy to make and it overstates a negative result the paper
> already reports honestly at the right strength. This sits alongside the two
> standing corrections in SESSION_STATE.md §9 (the router is question-driven,
> never "patient-adaptive"; mode T's EHR-contradiction is `n/a`, never 0.0000).

**This does not contradict Table VI, and must not be reported as if it did.**
Table VI measures *performance* contribution (removing the patient features
costs +0.0000 accuracy / −0.0181 macro-F1); SHAP measures *attribution
magnitude*. A feature can carry attribution and still be redundant — its
information also recoverable from the question — or net-harmful. Both are true
here: the router demonstrably consults patient state, and consulting it does
not improve routing. The honest statement is **"question-driven with
performance-neutral use of patient state"**, not "patient features are
ignored". The 2026-08-14 reconciliation (structural features decisive on a
minority of borderline cases, negligible in aggregate) is now directly
evidenced rather than conjectured.

The n_labs mechanism reproduces per-decision without hand-picking. Tier B row
289, a `lab` question on an admission with no recorded labs: `n_labs` is the
**#1** attribution over all 389 features (φ = +0.808 toward T, scaled value
−1.45, the low end). This is the 2026-08-14 audit's hand-found mechanism,
recovered automatically.

### Faithfulness (deletion test) — SHAP's rankings are causal for this model

Ablate the top-k features SHAP credits toward the predicted class (replace with
the router_train mean), check P(predicted class) falls. Random-k control at the
same budget, 5 repeats, seed 42.

| tier | directional agreement | mean flip rate | AOPC top-k | AOPC random-k | ratio |
|---|---|---|---|---|---|
| A | **0.9133** | 0.300 | 0.1987 | 0.0071 | **27.9x** |
| B | **0.8989** | 0.2545 | 0.1950 | 0.0076 | **25.8x** |

Directional agreement reaches **1.0000 at k≥10** on tier B (0.99 at k≥10 on A);
it is weakest at k=1 (0.61 B / 0.70 A), i.e. the single top-ranked feature alone
is not always decisive, but small groups reliably are. Random-k barely moves the
model (flip rate 0.011–0.013 vs 0.25–0.30). Both curves plateau by k≈20, meaning
routing rests on a small set of features rather than diffuse mass across 389.

**Caveat to carry into the write-up:** ablating a dense BGE dimension to its
training mean is an off-manifold perturbation, so this measures the model's
local sensitivity to the features SHAP credits, not a real-world intervention.
That is the standard limitation of deletion-based faithfulness and should be
stated, not glossed.

### Artifacts (all under the gitignored `experiments/results/final_eval/`)

`shap_group_attribution.csv` · `shap_feature_ranking.csv` ·
`shap_per_decision_sample.csv` · `shap_faithfulness.csv` ·
`shap_faithfulness_summary.csv` · `shap_faithfulness_per_row.csv` ·
`shap_metadata.json` (versions, model sha256, seed, gate results) ·
`figures/shap_beeswarm_tier{A,B}_{T,TE,TEK}.png` ·
`figures/shap_faithfulness_curve_tier{A,B}.png`

---

## 2026-08-07 — QLoRA training crash (bug + mitigation, no root cause confirmed yet)

### Incident

First `python -m src.model.train_qlora` run after the audit fixes: pre-flight,
model load, and data prep (retrieval-augmented, mode distribution
`{T: 346, T+E: 339, T+E+K: 315}`, 0% retrieval fallback) all completed
correctly. Training started and logged 4 steps with healthy, decreasing loss
(2.674 → 1.411). Step time was abnormally slow (~544s/step vs. the ~1–2 min/
step implied by the README's "~4–6 hours for 226 steps" estimate — projects
to ~30+ hours at that rate). Crashed at step 20/226, during the *first
evaluation pass* (9/100 eval examples in), with:

```
[ERROR] Training failed: CUDA error: an illegal instruction was encountered
```

No checkpoint existed on disk afterward (`save_steps=20` never triggered
because eval, which runs first at a coincident step, crashed before
completing) — a naive re-run would restart from step 0, losing ~3 hours.

### Immediate bug found and fixed

`_sample_modes()` (added as part of the 2026-08-06 fix) used `np.array` /
`np.random.RandomState` but the file never imported `numpy`. This is a
separate, unrelated `NameError` that surfaced on the *first* run attempt
(before any GPU work happened) and was fixed first
(`import numpy as np` added to `src/model/train_qlora.py`).

### Working hypothesis for the crash (not yet confirmed — no GPU access to verify directly)

Retrieval-augmented training prompts (up to 5 note passages + EHR snapshot +
KG facts for T+E+K) are structurally much longer than the flattened-
snapshot-only text `max_length=1024` / `eval_steps=20` /
`per_device_eval_batch_size=1` were originally tuned for. Longer sequences
plausibly pushed VRAM usage on the 6 GB card into Windows' shared-GPU-memory
fallback (silent, catastrophic slowdown, no OOM error) during sustained
multi-hour load, eventually destabilizing into the illegal-instruction
fault — most likely during the sustained-load eval pass specifically, since
that's exactly where it crashed. This is a hypothesis, not a confirmed root
cause: no direct GPU/driver/VRAM telemetry was available at diagnosis time
(only `nvidia-smi` immediately after the crashed process had already exited,
showing 0 MiB used / 36°C idle — uninformative about conditions *during* the
crash).

### Mitigations applied (reduce risk + cost of recurrence; do not fully explain the crash)

- Added a token-length diagnostic (p50/p90/max over a 200-example sample) to
  `load_and_prep_data()`, printed before training starts, to make the
  "prompts got much longer" hypothesis checkable on the next run.
- `max_length`: 1024 → 768 (`src/model/train_qlora.py` SFTConfig).
- `EVAL_FRACTION` (fraction of `ehrqa_finetune.parquet` held out for eval):
  0.10 → 0.04, i.e. eval set ~100 → ~40 examples.
- `eval_steps` / `save_steps`: 20 → 40 (halves how often the eval code path
  — where the crash occurred — runs; kept equal since
  `load_best_model_at_end=True` requires `save_steps` to be a multiple of
  `eval_steps`).

### Escalation — full BSOD on retry

The retry (with the mitigations above) ran ~2–3 hours before the *Windows
system itself* crashed: `KMODE_EXCEPTION_NOT_HANDLED (0x1E)`, driver
`nvlddmkm.sys`. This is a kernel-mode bugcheck, not a catchable Python/CUDA
exception — none of `train_qlora.py`'s exception handling (or checkpointing)
can react to it, since the whole OS crashed.

Root-cause analysis (ranked by probability, evidence-based, no GPU access to
verify directly):

1. **`optim="paged_adamw_8bit"` (highest probability).** Paged optimizers use
   CUDA Unified Memory (UVM) demand-paging to spill optimizer state between
   VRAM and system RAM. This mechanism is documented as materially less
   stable on Windows WDDM than on Linux, and is a known source of driver
   hangs/crashes under sustained pressure. Almost certainly unnecessary here:
   LoRA has only ~38M trainable params, so 8-bit optimizer state is tens of
   MB — far too small to need paging. Likely a defensive leftover from
   before the retrieval-augmentation fix, never reassessed against the new
   (longer, more variable-length) prompts.
2. **VRAM oversubscription from longer/variable-length retrieval-augmented
   prompts**, compounding #1 — consistent with the previously-observed
   abnormal step time (silent paging/thrashing rather than a clean OOM).
   Sustained near-VRAM-limit operation over hours raises cumulative risk of
   a spike triggering WDDM's TDR (Timeout Detection & Recovery); a failed
   TDR recovery can escalate to exactly this bugcheck.
3. **Thermal/power throttling instability (laptop hardware)** — plausible
   given ~100% GPU utilization sustained for hours (far outside typical
   gaming load patterns laptop cooling is validated against), but
   unconfirmable without temperature/clock telemetry from during the crash
   (none exists — the crash prevented logging).
4. **Driver channel (Game Ready vs. Studio) / driver-CUDA version mismatch**
   — not yet checked; Studio drivers are generally recommended for sustained
   compute workloads.
5. **Memory fragmentation** from the wide allocation-size variance across
   T/T+E/T+E+K prompt lengths — contributing/compounding factor, not a
   standalone cause.
6. Gradient checkpointing config, dataloader settings — assessed as low
   probability; not changed.
7. Underlying hardware defect — cannot be ruled out remotely; only relevant
   if the fixes below don't resolve it.

### Fixes applied (2026-08-07, second round)

- `optim`: `"paged_adamw_8bit"` → `"adamw_8bit"` (removes the Windows-UVM-
  paging failure mode entirely; addresses the top-ranked cause).
- `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`,
  set at the top of `train_qlora.py` before `torch` is imported (must
  precede CUDA context init to take effect) — reduces allocator
  fragmentation from variable-length sequences.
- `eval_steps`/`save_steps`: 40 → 25 (more frequent checkpoints are
  affordable again now that the more likely root cause is addressed; caps
  how much progress any future crash, of any cause, can destroy).

None of these changes touch `Retriever`, `TOP_K`, retrieval semantics,
oracle-label generation, or evaluation — the T/T+E/T+E+K alignment fix from
the 2026-08-06 audit is unaffected.

### Checkpoint integrity verification (before the round-2-mitigated retry)

Checked `models/medgemma-4b-qlora/` for a resumable checkpoint from the
round-1-mitigated run (the one that ended in the BSOD). Found
`checkpoint-40` (step 40 coincided with `eval_steps=save_steps=40` from
round 1), but it was **incomplete**:

| File | Complete reference (`checkpoint-220`, old run) | `checkpoint-40` |
|---|---|---|
| `adapter_model.safetensors` | 77,116,432 bytes | 77,116,432 bytes (complete) |
| `optimizer.pt` | 61,104,836 bytes | 286,720 bytes (truncated ~0.5%) |
| `scheduler.pt` | present | missing |
| `rng_state.pth` | present | missing |
| `trainer_state.json` | present | missing |

Conclusion: the BSOD hit mid-checkpoint-write. Model weights finished
serializing first and are intact; optimizer/scheduler/RNG/trainer-state
writes were in progress or not yet started when the OS crashed.
`trainer_state.json` (needed to restore `global_step`/epoch/best-metric
tracking) never got written at all. This checkpoint cannot be safely
resumed from — `resume_from_checkpoint` requires it. Deleted
(`rm -rf models/medgemma-4b-qlora/checkpoint-40`); no salvage value and its
presence could interact unpredictably with the next run's checkpoint-
rotation logic.

**Answer to "will the next run start from checkpoint or step 0": step 0.**
No valid checkpoint exists — the only one on disk was incomplete and has
been removed.

### Fault-tolerance hardening applied (2026-08-07, third round)

- `get_last_checkpoint()` rewritten: now validates every checkpoint
  (`trainer_state.json`, `optimizer.pt`, `scheduler.pt`, `rng_state*.pth`,
  `adapter_model.safetensors`, `adapter_config.json` all present) before
  trusting it, iterating newest-to-oldest and skipping (with a loud
  `[WARN]`) any incomplete one instead of either crashing on it or silently
  treating it as valid. Falls back to `None` (step 0) only if every
  checkpoint found is incomplete.
- `save_steps`/`eval_steps`: 25 → 15 (bounds worst-case lost progress from
  any future crash, of any cause, to ~15 steps).
- `save_total_limit`: 2 → 3 (so a crash corrupting the newest checkpoint —
  exactly what happened this time — still leaves two older *validated*
  checkpoints in reserve before falling back to step 0, rather than one).
- `main()` now prints an explicit, unambiguous banner stating whether the
  run is resuming (with checkpoint path) or starting from step 0, so this
  never has to be inferred from log scrollback again.

### Training completed successfully (2026-08-08)

240/240 steps, 2 epochs, ~27h48m runtime (slow, but no crash — the
fault-tolerance/stability fixes above held). `train_loss` 2.89 → 0.27;
`eval_loss` 1.62 → 0.052, plateauing from ~epoch 0.5 onward (train_loss kept
falling toward ~0.03-0.09 while eval_loss stayed flat at ~0.052-0.056 —
consistent with the LoRA adapter mostly memorizing the 960-example train
split after the first half-epoch; `load_best_model_at_end=True` already
selects the checkpoint least affected by this, so no action needed, but
worth keeping in mind when interpreting downstream oracle/router/eval
results — this adapter may generalize somewhat less than the flat eval_loss
alone suggests).

A second, unrelated bug then crashed `save_training_metadata()`:
`context_mode_distribution` (from `pd.Series(...).value_counts()`) contains
numpy `int64` values, which `json.dump()` cannot serialize. This happened
**after** `trainer.model.save_pretrained()`, `tokenizer.save_pretrained()`,
and `trainer.save_state()` had already completed — verified directly on
disk: `checkpoint-210`/`-225`/`-240` are all complete (every required file
present, `adapter_model.safetensors` matches the known-good reference size
exactly), and the top-level `models/medgemma-4b-qlora/` has valid weights,
tokenizer, and `trainer_state.json`. Only `training_metadata.json` was
truncated (241 bytes, invalid JSON). **No model/weight corruption; no
retraining needed.**

Also noted: the top-level `adapter_model.safetensors` is exactly 2x
`checkpoint-*`'s size (154,112,104 vs. 77,116,432 bytes) — consistent with
fp32 vs. bf16 storage, not corruption. `trainer_state.json` confirms
`best_model_checkpoint: checkpoint-210` (best_metric edges out checkpoint-240
by a statistically immaterial amount in the 5th decimal place) — since
`load_best_model_at_end=True`, the top-level save holds checkpoint-210's
weights, reloaded and re-serialized (apparently in fp32) by
`save_pretrained()`.

**Fixes applied:**
- `save_training_metadata()` now uses a `_NumpyJSONEncoder` (converts any
  `np.integer`/`np.floating` to native Python `int`/`float`) so this can't
  recur on a future fine-tuning run.
- `training_metadata.json` for this completed run reconstructed directly
  from the console log + `trainer_state.json` (not re-derived/estimated) —
  `library_versions` left explicitly marked as unverified rather than
  guessed, since the investigating session has no access to the `ehr-rag`
  conda environment.

**Status: fine-tuning phase complete.** `models/medgemma-4b-qlora/` is ready
to use as-is for the next pipeline steps (router dataset build → oracle
labels → router training → final eval).

---

## 2026-08-09 — Stale `data/lakehouse/qa/` directory silently used by 3 pipeline scripts

### Trigger

`python -m src.router.build_router_dataset` completed without error but
reported `Sparsity join: 0/200 rows matched (0.0%)` and
`Sparsity join: 0/100 rows matched (0.0%)` for router_train/router_val —
100% of rows ended up with `sparsity_bucket="unknown"` and `n_meds`
defaulted to 0 for every row.

### Investigation

Compared `hadm_id` values, dtypes, and file mtimes across every candidate
source, using `pyarrow` installed ad hoc into the investigating session
(diagnostic only — not part of the `ehr-rag` env):

| File | mtime | hadm_id nunique | `contraindication_check` rows |
|---|---|---|---|
| `data/qa/ehrqa_router_train.parquet` | 2026-08-06 18:26 (fresh, this session's Step 3 rerun) | 28 | 32 |
| `data/lakehouse/qa/ehrqa_router_train.parquet` | **2026-07-05 23:50** (stale — predates this entire audit) | 34 | **0** |
| `data/router/router_train_examples.parquet` (actual `build_router_dataset.py` output) | — | 34, hadm_id sample matches the **stale** file exactly | — |
| `data/lakehouse/sparsity.parquet` | 2026-08-06 18:28 (computed from `data/qa/`, per `lakehouse/sparsity.py`'s own `QA_DIR = Path("data/qa")`) | 220 total across all splits, disjoint hadm_id population from the stale file | — |

Root cause confirmed: `data/qa/` (written by `generate_synthetic_ehrqa.py`,
the actual/current QA generator) and `data/lakehouse/qa/` (a leftover
snapshot from **before** the 2026-08-06 audit even started — old enough
that `contraindication_check` didn't exist as a question type yet) both
existed on disk simultaneously. `build_router_dataset.py`'s
`get_input_path()` checked `data/lakehouse/qa/` *first* and silently used
it whenever present, ignoring the fresh `data/qa/` data entirely. Since
`sparsity.parquet` was correctly computed from `data/qa/`, the two hadm_id
populations barely overlapped — hence the 0% join. This was a systematic
directory-selection bug, not a dtype/format mismatch (both files had
identical `int64` hadm_id dtype).

### Escalation — this also silently affected the completed QLoRA training run

`train_qlora.py`'s `DATA_PATH` and `run_evaluation.py`'s `EVAL_QA_FILE` were
**both** hardcoded to `data/lakehouse/qa/...` with no fallback at all (worse
than `build_router_dataset.py`'s wrong-priority fallback — these had no
chance of finding the right file). Cross-checked against the 2026-08-08
training run's own console log: it reported "Unique admissions: 167" —
which matches `data/lakehouse/qa/ehrqa_finetune.parquet`'s hadm_id count
exactly (the fresh file has 137). **Confirmed: the completed 27h48m QLoRA
fine-tuning run was trained on the stale finetune split and saw zero
`contraindication_check` examples.**

This is not a data-leakage or split-validity problem — both the stale and
fresh finetune files independently draw only from the `finetune_train`
patient split in `splits/patient_splits.json` (verified: `get_admissions_
for_patients()` filters by subject_id via that split regardless of which
QA-generation run produced the file), just a different sample of admissions
within it, since admission ordering during generation uses DuckDB's
unseeded `ORDER BY random()`. So the trained adapter is not scientifically
invalid — but it does not include the finding-#5 fix (the
`contraindication_check` question type meant to give T+E+K a question type
answerable only via genuine KG reasoning), which was the entire point of
that change. **Whether to retrain is left to the project owner** — it is a
~27+ hour decision on already-scarce, crash-prone local compute, not
something to redo unilaterally.

### Fixes applied

- `build_router_dataset.py`: removed the `data/lakehouse/qa/` fallback
  entirely; `INPUT_TRAIN`/`INPUT_VAL` now point only at `data/qa/` (the one
  real canonical source, matching `generate_synthetic_ehrqa.py`'s own
  `OUT_DIR`). `get_input_path()` simplified to a single-path version.
- `train_qlora.py::DATA_PATH`, `run_evaluation.py::EVAL_QA_FILE`,
  `test_medgemma.py::EVAL_DATA`, and the default-hadm_id lookup in
  `retriever.py::main()` all repointed from `data/lakehouse/qa/` to
  `data/qa/`.
- `data/lakehouse/qa/` renamed to
  `data/lakehouse/qa_STALE_PRE_AUDIT_2026-07-05/` (archived, not deleted —
  preserved in case anyone needs to confirm what an old run used; no code
  references it anymore, confirmed via repo-wide grep after the rename).
- README.md's MedGemma Fine-Tuning section path reference corrected.

### Verification (2026-08-09, after project owner re-ran the fixed script)

`python -m src.router.build_router_dataset` re-run confirmed fixed:
`Sparsity join: 200/200 (100.0%)` and `100/100 (100.0%)`; bucket
distribution train `{low:113, medium:68, high:19}`, val
`{low:63, medium:23, high:14}`; `n_meds` warning gone.

Independently re-verified at the data level (not just the console log) by
reading `data/router/router_{train,val}_examples.parquet` directly:
`n_meds` has real non-zero values (mostly 8 = `MAX_MEDS` cap, some lower for
sparser patients — not a constant), `sparsity_bucket` unique values are
exactly `['high','low','medium']` (lowercase, zero `'unknown'` rows —
confirms this fix and the earlier `feature_pipeline.py` bucket-casing fix
are working together correctly), row/mode counts match the console output
exactly (600 = 200×3, 300 = 100×3, `T`/`T+E`/`T+E+K` each exactly 200/100).

**Both blocking issues (sparsity join, n_meds propagation) are resolved.
Safe to proceed to `python -m src.router.oracle_labels` and
`python -m src.router.train_router --tune`.**

---

## 2026-08-12 (later) — QA redesign implemented: patient-dependent routing + disease-level split

Project owner chose the full redesign (over reporting the negative result)
and requested all three additional baselines. Implemented; no GPU work run.

### Change 1 — Disease-level split kills the KG-memorisation confound

`splits/patient_splits.json` cannot protect KG questions, because disease→drug
facts are patient-independent: "Metformin is contraindicated in CKD-3" learned
from a fine-tune patient transfers intact to an eval patient. Measured effect
of this on 2026-08-11: mode T, with neither EHR nor KG, answered 62.5% of
contraindication questions exactly right.

Added a **family-aware, deterministic 50/50 split of the 25 seed diseases**
(`_split_seed_diseases()`). KG questions in the fine-tune split may only use
`FINETUNE_DISEASES`; router/eval splits may only use `HELDOUT_DISEASES`. The
model still learns the guideline-answer *format* (so it is not penalised on
ROUGE/BERTScore for phrasing) but never sees the *facts* it is tested on.

Diseases are split as **families**, never individually — an initial naive
split put CKD Stage 3/4 in fine-tune and Stage 5 in held-out, which would
have leaked, since the stages share contraindications (Metformin, NSAIDs,
Nitrofurantoin). `DISEASE_FAMILIES` groups the renal spectrum (CKD 3/4/5 +
AKI), thrombo-embolic/anticoagulation, hepatic, upper-GI, cardiac-metabolic,
infection, and psychiatric clusters. An assertion fails the build if any
family straddles the split. Resulting split: 13 fine-tune / 12 held-out,
overlap 0.

### Change 2 — Patient-dependent question types (the actual fix for routing)

The 2026-08-12 audit established that no router could work because the
optimal mode was a function of question TYPE, not PATIENT. Two new types
create genuine per-patient variation for the SAME question:

**`monitoring_labs`** — "Which laboratory tests should be monitored for this
patient's {disease}?" Gold = the guideline panel (KG `INDICATES_LAB`).
  * patient whose EHR already contains those labs → T+E can answer
  * patient with sparse EHR → the answer exists only in the KG → T+E+K needed

The gold answer is the same either way, so the *best mode* — not the answer —
varies by patient. This is a direct operationalisation of H2, and it is the
mechanism the previous design lacked entirely. Each row records
`ehr_covers_answer`, a ground-truth flag for whether the EHR alone sufficed,
so H2 can be analysed directly instead of inferred.

**`expected_symptoms`** — same structure over `HAS_SYMPTOM` edges.

### Change 3 — Question mix rebalanced per split

`QUESTION_MIX` (extractive, KG-dependent) per admission:
finetune (6, 2) · router_train (3, 4) · router_val (3, 4) · eval (3, 4).
Previously KG-dependent questions were only 16% of router splits, which is
why T+E+K had 10 train / **2 val** labels and per-class F1 was meaningless.

### Change 4 — Three evaluation baselines added

`SYSTEMS` is now 7. `StaticQType` (question_type → majority mode, fit on
router_train oracle labels ONLY) and `Oracle` (per-question best achievable
mode, derived post-hoc from the same generations — trains nothing, leaks
nothing) join the existing five. **Always-T+E was requested but is already
present**: the existing `T+E` system IS that policy, so a duplicate row was
deliberately not added rather than doubling compute for identical numbers.

Also fixed: the figure palette was hardcoded to 5 colours and would have
crashed with an IndexError on 7 systems.

### Post-regeneration verification caught two further bugs (both fixed)

**Bug A — `ehr_covers_answer` was 0 for 100% of monitoring_labs rows**
(77/77, 27/27, 17/17, 36/36), meaning the patient-dependence mechanism —
the entire point of the redesign — was not firing at all. Cause:
`patient_lab_names` was built from `get_abnormal_labs(limit=3)`, i.e. the
top *three abnormal* labs, while admissions average ~30 distinct labs.
Requiring the whole guideline panel to fall inside a 3-lab sample made the
flag unsatisfiable. Fixed by adding `get_all_lab_names()` (full DISTINCT
lab set per admission) and scoring coverage as a fraction with an 0.8
threshold, plus storing the continuous `ehr_lab_coverage` so H2 can be
analysed on a graded variable rather than a hard cut.

**Bug B — `d_note` was measured from the wrong reference point.** The SQL
computed `admittime − last_note`; since notes are written *during* the
stay this was almost always negative. Measured over 266 admissions: 111
were exactly 999 (no notes), 150 were negative, and **zero** fell in
between — the third sparsity term had silently degenerated into a
has-notes/no-notes binary rather than the note-staleness measure the design
doc specifies. This matters because `sparsity_bucket` is the independent
variable for H2.

Fixed to measure from `dischtime`, clamped at 0. A first attempt
over-corrected: `COALESCE(GREATEST(x, 0), 999)` collapsed everything to 0
(high bucket fell 20.3% → 4.5%) because DuckDB's `GREATEST(NULL, 0)`
returns 0, destroying the 999 sentinel. Corrected to an explicit
`CASE WHEN MAX(chartdate) IS NULL THEN 999 ELSE GREATEST(...) END`.

Honest caveat to carry into the write-up: even correctly computed, d_note
is ~0 for every admission that has notes (MIMIC-IV discharge summaries are
authored at discharge by definition), so in practice this term still
functions as a note-presence indicator. That is a property of the data, not
a bug, and should be stated rather than presented as a staleness measure.

### Verification results (all cheap, CPU-only)

| Check | Result |
|---|---|
| `ehr_covers_answer` varies | router_train 15/8 · router_val 7/7 · eval 24/14 |
| `ehr_lab_coverage` range | 0.00 – 1.00, mean 0.57–0.72 |
| sparsity buckets | low 140 / medium 81 / high 44 (16.6%), no threshold warning |
| negative d_note | 0 |
| KG disease overlap finetune ∩ eval | **0** |
| KG-dependent share of router/eval splits | ~38–47% (was 16%) |

The patient-dependent routing signal now exists in the data: for the same
`monitoring_labs` question, roughly half of patients have the guideline
panel already in their EHR (T+E sufficient) and half do not (T+E+K
required). That variation is what every previous router iteration lacked.

### Not yet run

Nothing GPU-heavy executed. QA + sparsity regeneration are complete and
verified; the retrain cycle is the next step.

---

## 2026-08-16 — Post-hoc rigour pass: bootstrap CIs, ablation, latency split, metric fix

Four additions requested to strengthen the results. One of them **overturns
a mechanistic claim made on 2026-08-14** and is recorded as a correction.

### 1. Bootstrap 95% CIs (10,000 paired resamples over the 300 questions)

Resampling is paired — every system answers the same questions — so
differences retain their pairing.

| metric | Router | T+E+K |
|---|---|---|
| BERTScore-F1 | 0.9685 [0.9601, 0.9762] | 0.9708 [0.9630, 0.9782] |
| BLEU | 0.7488 [0.7049, 0.7920] | 0.7481 [0.7043, 0.7912] |
| ROUGE-L | 0.8632 [0.8275, 0.8961] | 0.8630 [0.8280, 0.8948] |

Paired differences (Router − T+E+K):

| metric | diff | 95% CI | p | verdict |
|---|---|---|---|---|
| BLEU | +0.0007 | [−0.0009, +0.0022] | 0.304 | n.s. |
| ROUGE-L | +0.0002 | [−0.0068, +0.0065] | 0.391 | n.s. |
| BERTScore-F1 | −0.0024 | [−0.0057, +0.0003] | 0.526 | n.s. |
| EHR-contradiction | −0.0013 | [−0.0033, +0.0000] | 0.157 | n.s. |
| unsupported_rate | +0.0722 | [+0.0517, +0.0949] | 2.2e-10 | **significant** |
| total latency (ms) | **−3522.8** | [−3705.7, −3330.0] | 2.0e-42 | **significant** |

Quality-parity CIs all straddle zero — H1 criterion 1 now rests on interval
estimates, not point estimates.

### 2. Latency decomposition — sharpens the H1 efficiency claim

| system | retrieval ms | generation ms | total ms | retrieval % |
|---|---|---|---|---|
| T | 39.6 | 2170.0 | 2209.6 | 1.8% |
| T+E | 97.4 | 3243.9 | 3341.3 | 2.9% |
| **T+E+K** | **4342.8** | 3184.9 | 7527.7 | **57.7%** |
| **Router** | **822.3** | 3182.6 | 4004.9 | 20.5% |
| StaticQType | 609.2 | 3180.8 | 3790.0 | 16.1% |

**Generation cost is essentially identical across T+E, T+E+K and Router
(~3183 ms).** The router's entire 3523 ms saving comes from retrieval —
specifically from skipping the Neo4j round-trip, which is 57.7% of T+E+K's
total. The efficiency claim is therefore about *KG retrieval avoidance*, not
about shorter prompts or faster decoding, and should be stated that way.

### 3. CORRECTION — the "patient-adaptive routing" claim is NOT supported

Feature ablation, same hyperparameters, only the feature set varies:

| variant | n_features | accuracy | macro-F1 | F1 T | F1 T+E | F1 T+E+K |
|---|---|---|---|---|---|---|
| majority baseline | 0 | 0.5900 | 0.2474 | – | – | – |
| **question_only** (384 BGE) | 384 | **0.9200** | **0.8873** | 0.8000 | 0.9672 | **0.8947** |
| **patient_only** (5 structural) | 5 | **0.3600** | 0.3385 | 0.3051 | 0.5000 | 0.2105 |
| full (deployed) | 389 | 0.9200 | 0.8692 | 0.8182 | 1.0000 | 0.7895 |

Contribution of patient features (full − question_only): accuracy
**+0.0000**, macro-F1 **−0.0181**, T+E+K F1 **−0.1052**. Patient features
alone are barely above chance (0.36 accuracy).

**This contradicts the 2026-08-14 conclusion that the router performs
patient-adaptive routing, and that conclusion is withdrawn.** The earlier
evidence was real but insufficient: the router *does* vary its prediction
across patients for identical question text (structural features differ), and
`n_labs` *does* separate T from T+E on lab questions — but the ablation shows
that variation is net-neutral-to-harmful, and question embeddings alone reach
the same accuracy with a *higher* macro-F1. Correlational evidence for a
mechanism is not evidence that the mechanism carries the performance; the
ablation is the correct test and it was not run until now.

Reconciling the earlier lookup comparison: the router still beats the
question-TEXT lookup (0.92 vs 0.82), but the ablation shows the gain comes
from **BGE embeddings generalising across semantically similar questions**
(a lookup is exact-match and fails on the 11% unseen questions), not from
patient state.

**Implications.** H1's parity-at-lower-latency result is unaffected — it does
not depend on why the router routes well. But the framing must change: this
is a *learned question-routing policy*, not a patient-adaptive one. Combined
with the already-contradicted H2, the honest overall finding is that on this
benchmark **retrieval mode is predictable from the question, and patient
state adds no measurable routing signal.** That is a legitimate and useful
negative result, but it is a materially weaker claim than the one implied by
the project's original framing.

Artifacts: `router_ablation.csv`, `bootstrap_ci.csv`,
`paired_differences_router_vs_tek.csv`, `latency_decomposition.csv`,
`unsupported_length_confound.csv`. The deployed router in `models/router/`
was not modified by the ablation.

### 4. unsupported_rate is substantially a context-length proxy

Pearson correlation between prompt length and unsupported_rate: pooled
**r = −0.517, r² = 0.267, p = 7.7e-144** (per-system r from −0.27 to −0.57).
About a quarter of the metric's variance is explained by context length
alone, confirming it penalises any system that retrieves less.

**Length-controlled replacement — and it does NOT rescue H1 criterion 3.**

Each answer is scored against its real context and against a length-matched
decoy context from a different question; `grounding_excess = decoy − real`
cancels the shared length bias. Higher = better grounded.

| system | unsupported (orig) | unsupported (recomputed) | vs decoy | **grounding_excess** |
|---|---|---|---|---|
| T | 0.5813 | 0.6982 | 0.7134 | **0.0152** |
| T+E | 0.3244 | 0.3566 | 0.8405 | 0.4839 |
| **T+E+K** | 0.1832 | 0.1872 | 0.8143 | **0.6271** |
| **Router** | 0.2554 | 0.2748 | 0.8127 | **0.5379** |
| Random | 0.3493 | 0.4038 | 0.7533 | 0.3495 |
| StaticQType | 0.2696 | 0.3022 | 0.8063 | 0.5041 |
| Oracle | 0.2552 | 0.2802 | 0.8097 | 0.5295 |

Router vs T+E+K: **0.5379 vs 0.6271, diff −0.0891, p = 1.2e-05.** The router
is still significantly less grounded after the length bias is removed.

**The "metric artifact" explanation offered on 2026-08-15 is therefore
withdrawn.** The length confound is real (original r = −0.51 → controlled
r = +0.33, r² 0.26 → 0.11), but correcting it does not change the verdict:
H1 criterion 3 is a genuine failure. The router trades grounding for
latency, because it routes ~25% of questions to a mode with materially
weaker grounding. This must be reported as a real limitation, not as a
measurement problem.

**New finding the corrected metric exposes: mode T is essentially
ungrounded.** Its grounding_excess is **0.0152**, i.e. an answer produced
under mode T is barely better explained by its own retrieved passages than
by an unrelated context of the same length. The original metric hid this by
reporting T at 0.5813 — a number that looks merely "worse" rather than
"unsupported by its own evidence". This is a stronger and more interpretable
hallucination result than the original metric could produce, and it is the
version worth reporting.

Caveat: the controlled metric is not perfectly length-free (r = +0.33,
r² = 0.11 residual). Some of that is legitimate — more context genuinely can
ground better — but it is not a fully orthogonalised measure and should be
described as length-*controlled*, not length-*independent*.

---

## 2026-08-15 — FINAL HELD-OUT EVALUATION COMPLETE (300/300, 7 systems)

Completed after one resume from the 90-question checkpoint. 2100 rows
(300 × 7), **0 empty answers**, all figures and CSVs written.

| System | BLEU | ROUGE-L | BERTScore-F1 | EHR-Contra | Unsupported | Latency (ms) | KG facts |
|---|---|---|---|---|---|---|---|
| T | 0.1588 | 0.2676 | 0.7937 | 0.0000* | 0.5813 | 2210 | 0.00 |
| T+E | 0.6859 | 0.7965 | 0.9553 | 0.0247 | 0.3244 | 3341 | 0.00 |
| T+E+K | 0.7481 | 0.8630 | 0.9708 | 0.0247 | 0.1832 | 7528 | 8.88 |
| **Router** | **0.7488** | **0.8632** | **0.9685** | **0.0233** | 0.2554 | **4005** | 2.04 |
| Random | 0.5131 | 0.6272 | 0.9002 | 0.0193 | 0.3493 | 4298 | 3.27 |
| StaticQType | 0.7338 | 0.8392 | 0.9634 | 0.0240 | 0.2696 | 3790 | 1.34 |
| Oracle | 0.7497 | 0.8727 | 0.9708 | 0.0233 | 0.2552 | 3962 | 1.66 |

Router mode usage: T+E 174, T 74, **T+E+K 52 (17.3%)**.

### H1 verdict: 2 of 3 pre-registered criteria met — PARTIALLY SUPPORTED

Tested with paired Wilcoxon tests on the same 300 questions:

1. **Quality parity — PASS.** Router vs T+E+K BERTScore 0.9685 vs 0.9708
   (Δ −0.24pp, **p = 0.53**); BLEU 0.7488 vs 0.7481 (p = 0.30); ROUGE-L
   0.8632 vs 0.8630 (p = 0.39). Statistically indistinguishable on all
   three, which is exactly the pre-registered bar ("within 1–2pp, not
   significantly worse").
2. **Latency — PASS, exceeds target.** 4005 ms vs 7528 ms = **−46.8%**
   (target was −25–40%), achieved by invoking the KG on only 17.3% of
   questions.
3. **Hallucination — FAIL.** EHR-contradiction is effectively tied
   (0.0233 vs 0.0247, p = 0.157), but unsupported-rate is
   **0.2554 vs 0.1832, Δ +0.0722, p = 2.2e-10** — the router is
   significantly worse.

**Why criterion 3 fails, and an important caveat about the metric.**
`unsupported_score()` is `|answer_words − context_words| / |answer_words|`,
so it is *mechanically* anti-correlated with context length: any system that
sometimes retrieves less context scores worse by construction. The router
routes 24.7% of questions to T (unsupported 0.5813) and inherits that. This
is largely a **metric artifact rather than a demonstrated safety
regression** — but it is a real failure against the pre-registered
criterion and must be reported as such, not explained away.

### Metric defect found: EHR-contradiction is UNDEFINED for mode T

T scores **exactly 0.0000** while every other system sits at 0.019–0.025.
Cause: `ehr_contradiction_score()` only scans context lines containing
"diagnos" or "lab", and mode T has no EHR snapshot at all, so the detector
structurally cannot fire. The correct reading is **"not measurable for T"**,
not "T never contradicts the EHR". Reporting 0.0000 would make the weakest
system look perfectly safe. The targets doc anticipated exactly this
("any system's hallucination rate = 0% → detector likely not firing").
Must be marked N/A in the paper.

### H2 verdict: CONTRADICTED (again), consistent with the known mechanism

T+E+K advantage over T+E by sparsity bucket:

| bucket | T+E | T+E+K | gap |
|---|---|---|---|
| high | 0.9835 | 0.9892 | **+0.57pp** |
| medium | 0.9607 | 0.9727 | +1.20pp |
| low | 0.9470 | 0.9665 | **+1.95pp** |

Pre-registered prediction was ≈0 in LOW growing to 2–5pp in HIGH. Observed
is the **exact inverse**, reproducing the 2026-08-11 finding with the full
held-out set. Mechanism (already established): KG retrieval keys off the
patient's diagnosis list, so high EHR sparsity means fewer diagnoses to
match and *less* KG signal — the KG cannot fill a gap when the gap includes
its own index key. This is a coherent negative result, not noise.

### Results ABOVE pre-registered ranges — cause identified, disclose don't hide

BLEU 0.69–0.75 against a 0.05–0.20 target; ROUGE-L and BERTScore likewise
above range for T+E / T+E+K / Router. The targets doc flags BLEU > 0.5 as
"possible reference answer leaking into the prompt". **It is not train/test
leakage** — it is the extractive-answer confound documented since the
2026-08-06 audit (finding #5): for the ~61% of eval questions that are
extractive templates, the gold answer is a verbatim EHR field that appears
in the T+E/T+E+K context, so copying it scores near-ceiling. Mode T, which
lacks the EHR snapshot, scores 0.1588 — inside the target range — which is
consistent with this explanation and inconsistent with genuine leakage.

Latencies also exceed targets (T+E+K 7528 ms vs 1800–3500). Dominated by the
Neo4j round-trip (~4.3 s measured during router-dataset build), not
generation.

### Genuinely strong results

- **Router ≈ Oracle upper bound**: BLEU 0.7488 vs 0.7497, ROUGE-L 0.8632 vs
  0.8727, BERTScore 0.9685 vs 0.9708. The router captures essentially all
  available routing headroom on this eval set.
- **Router > StaticQType on every quality metric** (BERTScore 0.9685 vs
  0.9634, BLEU 0.7488 vs 0.7338), confirming on held-out data the
  router-beats-lookup result found on router_val.
- **Router >> Random** (BERTScore 0.9685 vs 0.9002) and at lower latency
  (4005 vs 4298 ms).
- Clean quality ordering T < T+E < T+E+K, and unsupported-rate falls
  monotonically with more context (0.5813 → 0.3244 → 0.1832).

---

## 2026-08-14 (evening) — Final evaluation crashed; two robustness gaps fixed

### Incident

`run_evaluation.py` died ~6 questions into the 300-question held-out run:

    RuntimeError: CUDA error: an illegal memory access was encountered
      ... transformers/cache_utils.py:222 in lazy_initialization
          self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)

Same fault family as the 2026-08-09 oracle crash (illegal memory access at
inference), raised inside Gemma3's sliding-window KV-cache initialisation.
Root cause remains unconfirmed — consistent with the ongoing hardware/driver
instability rather than any code defect.

### Two robustness gaps this exposed (both real bugs, both fixed)

1. **`run_evaluation.py` had NO checkpointing.** It accumulated all rows in
   memory and wrote only at the end (single `to_csv`). `oracle_labels.py`
   has had a `CheckpointManager` since the beginning; the evaluation script
   never got one. Consequence: **every completed question was lost**, and a
   crash at question 299 of 300 would have destroyed a ~2.5h run.
   `experiments/results/final_eval/` was confirmed empty afterwards.

2. **`run_evaluation.generate()` had NO CUDA fault recovery.**
   `oracle_labels.AnswerGenerator.generate()` retries once after
   `empty_cache()`; the eval path let a single transient fault propagate and
   kill the process.

### Fixes

- Checkpoint every `CHECKPOINT_EVERY = 10` questions to
  `experiments/results/final_eval/_eval_checkpoint.parquet`, written
  **write-to-temp-then-`os.replace`** so a crash mid-write cannot corrupt it
  (this is exactly how the 2026-08-07 QLoRA checkpoint was corrupted).
  On restart, completed `q_idx` values are skipped.
- Checkpoint is deleted only after `per_question_results.csv` is
  successfully written, so a later clean run is never silently resumed from
  a stale one.
- `generate()` now retries once on CUDA/OOM errors after clearing the cache,
  and on a second failure records an empty answer and continues rather than
  aborting the run.

Ordering verified: checkpoints contain only raw generation rows (saved
before BERTScore and before the post-hoc Oracle system is appended), so a
resumed run cannot inherit stale scores or duplicate Oracle rows.

### CORRECTION (second crash, same session): the retry design was wrong

The retry added above **cannot work, and the log proves it**. After the
first illegal memory access: retry 1 failed identically, retry 2 failed
identically, then FAISS query-embedding failed for all three modes, and
finally even `.to(model.device)` on a tokenizer output raised. An illegal
memory access **poisons the CUDA context for the entire process** — it is
not a transient, retryable condition. Retrying only produces empty answers
that would silently contaminate results with fabricated failures.

Replaced with the correct design:
- `UnrecoverableCudaError` raised on illegal-memory-access / device-side
  assert / generic CUDA error.
- The main loop catches it, **flushes the checkpoint, logs how to resume,
  and exits (SystemExit 2)** rather than limping on with a dead context.
- `CHECKPOINT_EVERY` lowered 10 → 5, since a fault now forces a process
  restart and the flush interval directly bounds lost work.

**The checkpointing itself worked**: the second crash left a valid
checkpoint with **90/300 questions complete and 0 empty answers**, so ~40
minutes of work survived and the run resumes from question 91.

### Likely structural cause identified

`oracle_labels.py` completed all 300 questions cleanly at 3.58 GB peak.
`run_evaluation.py` crashes early — and the one structural difference is
that oracle reads pre-built `prompt_context` from parquet and loads **no
encoder**, whereas run_evaluation calls the live `Retriever`, which keeps a
**BGE encoder resident on CUDA alongside 4-bit MedGemma**.

Added `MEDRAG_EMBED_DEVICE=cpu` to force the query encoder onto CPU. Not
enabled by default: CPU and GPU embeddings differ by ~1e-6, which can flip a
near-tied FAISS neighbour, so devices must not be mixed *within* one
evaluation run. Using it therefore requires discarding the existing
90-question checkpoint and starting clean.

### Residual risk (unchanged)

This does not prevent the CUDA fault — it makes the run survivable. Worst
case a crash now costs ≤10 questions, and re-running the same command
resumes. Three GPU consumers coexist during evaluation (4-bit MedGemma,
BGE embedder on cuda in `Retriever`, and BERTScore at the end); if faults
recur frequently, moving the BGE embedder to CPU is the next lever, at some
retrieval-speed cost.

---

## 2026-08-14 (later) — Independent read-only audit of the router result

Full adversarial re-verification of Accuracy 0.9200 / Macro-F1 0.8692 /
Balanced Accuracy 0.8720. Nothing was retrained, modified or regenerated.

### Verified genuine

- **All headline metrics recompute exactly** from `router_predictions.csv`
  (0.9200 / 0.8692 / 0.8720), matching `router_metadata.json`.
- **Provenance clean**: all 8 `models/router/` artifacts written within 0.6s
  of each other at 09:59:53, postdating both oracle parquets. No stale
  artifact from a prior run.
- **No leakage**: hadm_id overlap 0, subject_id overlap 0, duplicate val
  rows 0, KG diseases in both router splits ⊆ HELDOUT set.
- **Cost advantage is real and not KG-avoidance**: router cost 5.18 vs
  qTYPE 5.86, qTEXT 6.44, majority 9.15, random 14.56, oracle floor 3.80 —
  while predicting T+E+K **20 times against 18 actual**, i.e. slightly
  over-using KG, not dodging it. 15 TP / 5 FP / 3 FN.
- **Beats every baseline on identical rows**: Random 0.33, Majority 0.59,
  qTYPE 0.82, qTEXT 0.82, Router 0.92. Head-to-head on the 16 disagreements
  with qTEXT: router 11 correct, lookup 1.

### MECHANISM CONFIRMED — n_labs drives patient-dependent routing

The apparent contradiction (structural features only 1.6% of aggregate
gain, yet the router beats a question-identity lookup) is now resolved with
direct evidence rather than conjecture. On `lab` questions (n=21):

| gold mode | n_labs mean | min | max |
|---|---|---|---|
| **T** | **0.0** | 0 | 0 |
| **T+E** | **39.1** | 10 | 84 |

`n_labs` separates the two classes perfectly (0 vs ≥10), and the router
scores **21/21** on these rows. A single threshold split on one feature is
decisive locally while contributing almost no aggregate gain — which is
exactly why importance understates it. This is genuine patient-dependent
routing: when a patient has no labs recorded, the EHR snapshot cannot help,
and the router correctly falls back to T.

### Survives disproof attempts

| test | result |
|---|---|
| Exclude 29 memorisable (Q,A)-overlap rows | router 0.9437 vs lookup 0.8451 |
| Val questions NEVER seen in train (n=11) | router **0.6364** vs lookup 0.0000 |
| Repeated questions with varying gold | router 22/23 = 95.7% |

### Caveats that must be reported with the headline number

1. **0.92 is carried by an easy majority subset.** Extractive questions
   (n=63, 63% of val) are classified **perfectly (1.0000)** because their
   optimal mode is deterministic (T+E composite ≈1.0 vs T ≈0.5). On the
   genuinely hard KG-dependent subset (n=37) accuracy is **0.7838**.
   Excluding the perfect T+E class entirely: accuracy 0.8049 / Macro-F1
   0.8038. The paper should report the stratified breakdown, not the
   headline alone.
2. **Generalisation to unseen question text is much weaker**: 0.9551 on
   seen questions vs **0.6364** on the 11 unseen ones. Still far above the
   lookup's 0.0000 there, but the headline benefits from question overlap.
3. **Patient-adaptivity is demonstrated cleanly on lab questions, weakly on
   KG questions** — of 3 repeated questions with varying gold, 2 are
   answered 100% correctly and 1 (a `monitoring_labs` pair) 50%.
4. **`n_meds` has importance exactly 0.0** — a dead feature in this run.
5. Only 37 of 100 val rows exercise the KG decision at all.

### Note: `ehr_covers_answer` / `ehr_lab_coverage` are NOT router features

`RouterConfig.ehr_feature_cols` is `[n_labs, n_diag, n_meds,
sparsity_score, sparsity_bucket]`; the engineered coverage signals are
absent. On reflection this is **correct, not an oversight**:
`ehr_covers_answer` is computed by comparing the patient's labs against the
KG panel that constitutes the gold answer, so feeding it to the router
would leak answer-derived information into the routing decision. It is
properly used only as an analysis variable for H2. No change made.

### Verdict

No bug or leakage found that invalidates the result. GO for the final
held-out evaluation, with caveats 1–3 carried into the write-up.

---

## 2026-08-14 — Oracle + router on redesigned data: H1 mechanism CONFIRMED

### Oracle labels — the class-support problem is solved

| metric | pre-redesign | now |
|---|---|---|
| **T+E+K support (train/val)** | 10 / 2 | **38 / 18** |
| label distribution (train) | T+E 78.5% / T 16.5% / T+E+K 5% | **T+E 59% / T 22% / T+E+K 19%** |
| tie-decided label rate | 0.88 | **0.785** |
| composite T / T+E / T+E+K | 0.51 / 0.959 / 0.967 | **0.51 / 0.850 / 0.901** |
| hallucination T / T+E / T+E+K | 0.285 / 0.05 / 0.05 | **0.295 / 0.12 / 0.095** |
| context echo (all modes) | 0.0 | 0.0 |

Two results matter beyond the support fix. The T+E → T+E+K composite gap
widened from +0.008 to **+0.051** (6x), i.e. the KG now adds measurable
answer quality. And hallucination shows a full gradient for the first time
(0.295 → 0.12 → 0.095); previously T+E and T+E+K were tied at 0.05, making
the KG look irrelevant to safety, which is the paper's actual value claim.

Runtime also fell to 14 min (val) / 29 min (train) from 30/52, and peak VRAM
to 3.6 GB from 6.0 GB, both consequences of the shorter budgeted contexts.

### Router — beats every static baseline

Reported: Accuracy 0.9200 (majority 0.5900), Macro-F1 **0.8692** (majority
0.2474), Balanced Accuracy 0.8720. Both Macro-F1 and Balanced Accuracy sit
*above* the pre-registered target ranges, so — per the targets doc's own
instruction — they were audited as a suspected bug rather than accepted.

Unlike the 2026-08-12 router, the metrics survive audit:

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| T | 0.86 | 0.78 | 0.82 | 23 |
| T+E | 1.00 | 1.00 | 1.00 | 59 |
| T+E+K | 0.75 | 0.83 | **0.79** | 18 |

No degenerate class (T+E+K F1 was 0.29 on n=2 last time; now 0.79 on n=18),
precision and recall are balanced, and Balanced Accuracy (0.872) no longer
exceeds Accuracy (0.920) — the signature of the earlier artifact is gone.
Leakage re-verified: train/val hadm_id and subject_id overlap both 0.

**Baseline ladder (val, n=100):**

| policy | Accuracy | Macro-F1 |
|---|---|---|
| always-majority (T+E) | 0.5900 | 0.2474 |
| question_TYPE → majority mode | 0.8200 | 0.7024 |
| question_TEXT → majority mode (strong) | 0.8200 | 0.7588 |
| **XGBoost router** | **0.9200** | **0.8692** |

On 2026-08-12 the question_TYPE lookup *beat* the router (0.97 vs 0.87).
It no longer does. The router now beats even the stronger question_TEXT
lookup by +10 accuracy points. Head-to-head on the 16 rows where router and
text-lookup disagree: **router correct 11, lookup correct 1 (+10 net)**.

### Proof the router is patient-adaptive, not a question lookup

Question identity is fully captured by the question_TEXT baseline, so
beating it requires signal beyond the question. Direct confirmation — the
router assigns **different modes to identical question text**, tracking the
per-patient gold labels:

    n=11  "What is the most abnormal lab value..."      preds=[T, T+E]    gold=[T, T+E]    correct
    n=10  "Which lab abnormality is most concerning..." preds=[T, T+E]    gold=[T, T+E]    correct
    n=3   "Which laboratory tests should be monitored"  preds=[T, T+E+K]  gold=[T]         wrong

This is the property every earlier iteration lacked and is the mechanism H1
depends on.

### Honest caveats

- **Aggregate feature importance still shows structural features at only
  1.6%** (BGE question dims 98.4%), which appears to contradict the
  patient-adaptivity finding. Most likely reconciliation: XGBoost gain
  importance is an aggregate, and the structural features appear to matter
  decisively on the ~16% of borderline cases while contributing little
  total gain. Stated as the probable explanation, not a verified one — a
  per-case attribution (e.g. SHAP, currently unavailable: the library is
  not installed) would be needed to confirm it.
- Only **3 of 16** repeated questions receive varying predictions, so
  patient-adaptive behaviour is real but confined to a minority of cases.
- Tie-decided label rate remains 0.785, reflecting the extractive-question
  ceiling (those are still 63% of the mix).
- `audit_router.py` printed **stale hardcoded router metrics** in its
  section 5 comparison (0.8700/0.7076 from the previous run). The lookup
  figures it computed were correct; the router figures it printed were not.
  Corrected numbers are used throughout this entry.

### Status

All pipeline stages now validated. Ready for the final held-out evaluation
with the seven-system table.

---

## 2026-08-13 — Retrain on redesigned data COMPLETE and validated

### Run outcome — dramatically healthier than any previous run

Early stopping at **step 180/240** (epoch 1.5); best = `checkpoint-135`
(eval_loss 0.3824), correctly restored by `load_best_model_at_end`.

**Wall clock 2h45m, down from 13h17m.** `eval_runtime` stayed flat at
56-59s for the entire run, versus the previous run's degradation from 87s to
350s. That strongly suggests the Windows shared-memory spillover which
plagued earlier runs did not occur this time — most plausibly because
budgeted prompts are shorter on the redesigned data (p50 total 682 tokens).
No CUDA fault, no BSOD. Model mtime (02:19) postdates data mtime (22:51),
confirming no stale-artifact reuse.

eval_loss 0.3824 is *higher* than the previous run's 0.2548 — expected and
desirable: the redesigned task is genuinely harder (KG facts can no longer
be memorised, and patient-dependent questions are ~37% of the mix).

### Validation results (54 generations, router_train data, eval untouched)

**1. Context echo: 0/54 = 0.0%** (broken model: 52-63%). Fully resolved.

**2. KG-memorisation broken by the disease split — the key result:**

| mode | exact-correct on held-out-disease KG questions |
|---|---|
| T (no EHR, no KG) | 2/12 = **16.7%**  ← previous model: **62.5%** |
| T+E (EHR, no KG) | 2/12 = 16.7% |
| T+E+K (has KG) | 4/12 = **33.3%** |

T's ability to answer KG questions without KG access collapsed from 62.5% to
16.7%, confirming the disease-family split removed the memorisation
shortcut. KG lift is now **+16.7pp** measured against a non-inflated
baseline.

**3. Extractive questions — clean mode separation:**
T 16.7% vs T+E 100% vs T+E+K 100%. Exactly the expected pattern (EHR is
necessary and sufficient for extractive questions).

### Honest caveats

- **n=12 KG questions** in this spot-check. Directionally clear but not a
  precise effect size; the oracle run over 200/100 questions is the real
  measurement.
- **The three modes still frequently produce identical answers on KG
  questions.** Inspected samples show all three emitting the same output
  even where the KG contains the correct panel (e.g. gold "Creatinine, BUN,
  Potassium" vs all-modes "Bicarbonate, Calcium, Creatinine, Potassium,
  Sodium"). So the model does not always *use* retrieved KG facts even when
  they are present. T+E+K wins on 1/3 of KG questions rather than most.
- **Residual, unavoidable knowledge:** MedGemma is a medical base model and
  knows common contraindications (e.g. NSAIDs in AKI) from *pretraining*,
  independent of our fine-tuning. The disease split cannot remove that, and
  it will keep the T baseline above zero. This is a limitation to disclose,
  not a bug — and it again biases *against* the KG hypothesis.
- Exact-match is harsh here (semantically-correct answers differing only in
  a parenthetical reason score 0). The oracle's composite metric
  (0.60·BERTScore + 0.25·ROUGE-L + 0.15·EM) will credit near-misses, so
  oracle labels should be more discriminating than these exact-match rates.

### Status

Generator validated. `data/router/router_*_examples.parquet` are STALE
(built 2026-08-08/09 from the pre-redesign QA) and must be rebuilt before
oracle labelling.

---

## 2026-08-13 (later) — Two KG-retrieval bugs found during router-dataset rebuild

Rebuild ran clean (sparsity join 200/200 and 100/100, buckets healthy), but
inspection of the output found **18/74 (24.3%) of KG-dependent questions
retrieved ZERO KG facts**. Since these are precisely the questions where
T+E+K is supposed to win, this silently understated the KG effect.

**Bug 1 — the retriever ignored the question.**
`Retriever._retrieve_kg(question, hadm_id)` accepts `question` but never
used it, matching only on the patient's ICD diagnosis strings. A diagnosis
rendered as "Hypertensive chronic kidney disease with stage 1 through stage
4..." does not clear `find_matching_diseases()`'s 0.6 token-overlap bar
against the node "Essential Hypertension" — even though the question
literally names that disease. Verified: the question text contains the
target disease name in **100%** of the failing rows, and all 18 are
recoverable from it. Fixed by appending the question as an additional
match candidate (diagnoses still evaluated first, so patient context keeps
priority).

**Bug 2 — the matcher never stripped punctuation.**
Term extraction used `text.lower().replace(",", " ").split()`, leaving
punctuation attached. A question ending "...Essential Hypertension?" yields
the token `"hypertension?"`, which != `"hypertension"`, so the overlap score
was 1/2 = 0.5 — under the 0.6 bar. "Bipolar Disorder" only worked by luck,
because `disorder` is a STOPWORD, leaving a single-term match. Replaced with
regex tokenisation (`[a-z0-9]+`), which removes the whole class of
trailing-`?`/`.`/`)`/apostrophe misses. This also improves plain diagnosis
matching, independent of the question fix.

Both fixes verified together: 4/4 previously-failing KG questions now match,
and 3/3 extractive control questions are unchanged (no spurious matches
introduced).

### Consequence for the trained model — decision required

The model was fine-tuned with the buggy retriever, so ~24% of its
KG-dependent training contexts contained no KG facts. Quantified: KG types
are 15.8% of the finetune split and T+E+K is ~1/3 of modes, so roughly
**1.2% of the 960 training examples** were affected. The model demonstrably
can use KG facts when present (33.3% vs 16.7% exact-correct), having learned
from the 76% that did contain them.

Options recorded (owner's call, no unilateral action):
1. **Rebuild router dataset only** (~10 min) and proceed. Defensible: the
   generator is unchanged and simply receives KG facts more reliably now.
   Leaves a small train/inference distributional difference to disclose.
2. **Rebuild + retrain** (~2h55m total). Strict alignment; removes the
   caveat entirely. Cheap now that training is 2h45m rather than 13h.

Either way `build_router_dataset` MUST be re-run, because the stored
`prompt_context` values were produced by the buggy retriever.

### Post-fix rebuild verification

| metric | before fixes | after fixes |
|---|---|---|
| KG-type questions with facts (train) | 56/74 = 75.7% | **74/74 = 100%** |
| KG-type questions with facts (val) | 75.7% | **37/37 = 100%** |
| mean KG facts on KG questions | 6.9 | **10.3 / 10.7** |
| reported prompt length T / T+E / T+E+K | 1924 / 2103 / 2271 | **451 / 723 / 893** |
| mode isolation (T has EHR or KG) | 0% | 0% (unchanged) |
| mode isolation (T+E has KG) | 0% | 0% (unchanged) |

Note the KG context change is larger than the coverage number alone
suggests: mean facts per KG question rose from 6.9 to ~10.5, so T+E+K
prompts now carry materially more knowledge content, not merely more
reliable presence. This strengthens the case for retraining rather than
proceeding on the existing adapter.

### Retrain on KG-fixed retrieval — COMPLETE and validated (2026-08-13)

Ran the full 240 steps / 2 epochs in 4h14m; early stopping did not fire
because eval_loss kept improving until step 210 and then failed only twice
before the epoch cap (patience=3). Best = `checkpoint-210`, eval_loss
**0.37875**, correctly restored. One eval blip at step 195 (78s vs the
steady 59s) with no sustained degradation and no CUDA fault. Timestamps
verified: model 08:32 postdates both the QA data (22:51) and the KG
retrieval fix (03:00), so no stale-artifact reuse.

Validation (54 generations, router_train data; held-out eval untouched):

| metric | pre-KG-fix model | KG-fixed model |
|---|---|---|
| context echo rate | 0/54 = 0.0% | **0/54 = 0.0%** |
| T exact-correct (KG questions) | 16.7% | 16.7% |
| T+E exact-correct | 16.7% | 16.7% |
| **T+E+K exact-correct** | 33.3% | **41.7%** |
| **KG lift (T+E+K − T)** | +16.7pp | **+25.0pp** |
| extractive T / T+E / T+E+K | 16.7 / 100 / 100% | 16.7 / 100 / 100% |

The KG retrieval fixes raised measured KG benefit from +16.7pp to +25.0pp,
as expected from 76%→100% coverage and 6.9→10.5 mean facts.

A sample now shows the clean three-way separation the study is designed to
detect — and shows EHR alone actively misleading:

    GOLD : No, NSAIDs is contraindicated in Acute Kidney Injury (nephrotoxic).
    T    : No, NSAIDs is contraindicated in Acute Kidney Injury (worsens it).
    T+E  : Yes, NSAIDs is a standard first-line treatment for AKI.   <- WRONG/unsafe
    T+E+K: No, NSAIDs is contraindicated in Acute Kidney Injury (worsens AKI).

Unchanged caveat: on `monitoring_labs` the three modes still frequently
emit the same answer even with KG facts present, so the KG is used
inconsistently rather than always. n=12 KG questions here; the oracle run
over 200/100 is the real measurement.

### Also fixed: misleading prompt-length statistic

`build_router_dataset`'s summary reported `RetrievalResult.n_tokens_approx`
(an unbudgeted `len(text)//4` estimate) rather than the real tokenised
length of the budgeted context — overstating mode T's prompt as 1924 tokens
when the stored context is 442 (4.4x). The stored parquet column was always
correct; only the printed summary was wrong. Fixed, because this figure
feeds the paper's prompt-cost/efficiency analysis. Verified stored contexts
are within budget: T max 606, T+E max 962, T+E+K max 1299 (limit 1300).

---

## 2026-08-12 (final) — Pre-training GO/NO-GO audit #2

### CRITICAL bug caught: the run would have resumed the pre-redesign model

`models/medgemma-4b-qlora/` still contained `checkpoint-120/150/165` from the
2026-08-09 run. All three are structurally complete, so
`get_last_checkpoint()` would have validated `checkpoint-165` as healthy and
**resumed from it** — continuing a model fine-tuned on the OLD, pre-redesign
QA data, for 13 hours, silently.

Note the trap: `training_metadata.json` records
`dataset_path = data/qa/ehrqa_finetune.parquet`, which is the *same path* the
new run uses. Only the timestamps reveal the problem — the adapter was
trained 2026-08-09 18:48, while the QA data was regenerated 22:47. A
path-based staleness check would not have caught this; only comparing model
mtime against data mtime does.

Archived to `models/_ARCHIVE_pre_qa_redesign_2026-08-09/`. Output dir now
empty; `get_last_checkpoint()` confirmed to return None.

### (Q,A) pair overlap across splits — investigated, benign

6 split-pairs share identical (question, answer) tuples. Characterised:

* **finetune ∩ eval (9 pairs)** — extractive questions where two
  *different, patient-disjoint* admissions happen to share a diagnosis
  string ("Pneumonia, organism unspecified") or a generic answer ("No
  abnormal labs available"). This is coincidental label collision, not
  leakage: ICD labels necessarily repeat across patients, and the model
  must still read the patient's context to produce the string.
* **router/eval (18-21 pairs)** — KG questions, identical by construction:
  the guideline answer for a disease is patient-independent, so
  "What symptoms … Acute Kidney Injury?" → "Oliguria, Edema, Nausea."
  recurs for every AKI patient.

**Why this is benign rather than harmful**, verified empirically: for the
same repeated question text, the correct routing varies by patient. 3 of 4
repeated `monitoring_labs` questions have varying `ehr_covers_answer`, with
a within-question `ehr_lab_coverage` spread averaging 0.75 (router_train)
and 0.55 (eval), max 1.00. So a router that memorises question-text → mode
is actively *penalised*; it must use patient features. This is exactly the
property every previous iteration lacked, and the overlap is the mechanism
that tests it rather than a confound.

### Verification summary (all CPU-only, no training)

| Area | Result |
|---|---|
| hadm_id leakage, all 6 split pairs | 0 overlap |
| subject_id leakage, all 6 split pairs | 0 overlap |
| Disease-family integrity | all 7 families intact, 0 straddling |
| finetune ∩ router/eval KG diseases | 0 |
| KG share of router/eval splits | 37% / 37% / 39% (was 16%) |
| `ehr_covers_answer` both classes present | router_train 15/8 · router_val 7/7 · eval 24/14 |
| Within-question coverage spread | mean 0.55–0.75, max 1.00 |
| Answer embedded in own question | 0 rows, all splits |
| Schema / nulls | all required columns present, 0 nulls |
| Budget arithmetic | 600+350+350+96+128 = 1524 = MAX_SEQ_LENGTH |
| `save_steps % eval_steps` | 0 (required by load_best_model_at_end) |
| Loss masking on new data | **2.97% supervised** (was 100%) |
| Sequence overflow | 0/12; max total 1302 < 1524 |
| Retrieval fallback | 0.0% |
| KG facts for every held-out eval disease | 8/8 diseases, 3–11 facts each, none empty |
| FAISS index vs chunk metadata | 2,282,927 == 2,282,927 |

### GPU stability assessment (honest)

**Ruled out** — none of the crashes are attributable to the current training
config: the 2026-08-09 run completed 165 steps / 13h17m on exactly this
configuration (adamw_8bit, MAX_SEQ_LENGTH 1524, grad-checkpointing,
batch 1 × accum 8) without a fault.

**Mitigated** — paged optimiser removed (top suspect for the 2026-08-07
BSOD); sequence lengths bounded and asserted; checkpoints every 15 steps
with 3 retained and validity-checked on resume.

**Residual, NOT eliminated** — `expandable_segments` is confirmed
unsupported on this Windows build (warning observed again this session), so
allocator-fragmentation mitigation is inert; Windows shared-GPU-memory
spillover remains possible (it is the likely cause of the eval slowdown from
87s → 350s observed mid-run); and the 2026-08-09 `oracle_labels` crash
(illegal memory access, inference-time) has no confirmed root cause. The
laptop cannot be called crash-proof. What has changed is that a crash now
costs ≤15 steps rather than the whole run.

---

## 2026-08-12 — Router trained: headline metrics do not survive audit. Core design problem identified.

Reported metrics: Accuracy 0.8700 (majority 0.7900), Macro-F1 0.7076
(majority 0.2942), Balanced Accuracy 0.9451. Macro-F1 lands inside the
pre-registered 0.60-0.75 target, but Balanced Accuracy is *above* the
0.65-0.80 target — and the targets doc explicitly says above-range results
should be treated as suspected bugs. Audited accordingly; the audit
invalidates the headline reading.

### Finding 1 — Balanced Accuracy 0.9451 is an artifact of a 2-sample class

```
            Pred_T  Pred_T+E  Pred_T+E+K
True_T          19         0           0
True_T+E         3        66          10
True_T+E+K       0         0           2
```

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| T | 0.86 | 1.00 | 0.93 | 19 |
| T+E | 1.00 | 0.84 | 0.91 | 79 |
| **T+E+K** | **0.17** | **1.00** | **0.29** | **2** |

Balanced Accuracy is the mean of per-class *recall*, so the 2/2 T+E+K class
contributes a perfect 1.00 and drags the average to 0.945. Precision on that
class is 0.17: the router predicted T+E+K **12 times and was right twice**.
In deployment that is 10 unnecessary Neo4j round-trips — the exact cost H1
claims the router avoids. The honest per-class number is F1 = 0.29.

### Finding 2 — No data leakage (ruled out)

train/val hadm_id overlap = 0; patient-level split overlap = 0. The splits
are clean. The inflation is not leakage.

### Finding 3 — The router is a question-type lookup table, not an adaptive router

- Structural patient features carry **2.9%** of total feature importance;
  the 384 BGE question-embedding dims carry **97.1%**. `sparsity_bucket`
  ranks 30th at 0.0096.
- 88% of val rows have a question text that appeared verbatim in train
  (only 33 unique questions in train, 27 in val, 18 shared).
- **A trivial `question_type -> majority mode` lookup table, fit on train,
  scores val Accuracy 0.9700 vs. the XGBoost router's 0.8700.** The lookup
  table *beats* the trained router on accuracy (router wins on Macro-F1
  only via the degenerate 2-sample class).

The router is therefore an expensive, worse-performing approximation of a
seven-row lookup keyed on question type. It is not routing on patient state.

### Root cause: the QA design makes the optimal mode a function of question type, not patient state

Established chain:
1. Six of seven question types are extractive templates whose gold answer is
   a verbatim EHR field.
2. So T+E saturates (composite exactly 1.0000 on three types) and 93% of
   T+E vs T+E+K pairs are quality-identical.
3. So the best mode is essentially fixed per question type
   (primary_diagnosis -> always T+E, contraindication_check -> T or T+E+K),
   with no dependence on the individual patient.
4. So there is no patient-adaptive signal for a router to learn, and
   sparsity features are correctly ignored by XGBoost as uninformative.
5. So H2 (sparsity-dependent routing) has no mechanism to appear, consistent
   with the inverted/absent H2 signal already observed.

This is a **dataset-design** limitation, not an implementation bug. For a
learned router to be meaningful, the optimal mode must vary *across patients
for the same question* — which the current template QA never produces.

### Mandatory consequence for the paper

A static `question_type -> mode` policy is now a **required baseline** in
Phase 6. Any reviewer will ask whether the learned router beats a lookup
table; on current evidence it does not on accuracy. Reporting the router
without this baseline would not be defensible.

### Assets that remain scientifically sound

- Retrieval-augmented QLoRA fine-tuning, verified aligned train/inference.
- KG value on guideline questions: +18.8pp exact-correct
  (T/T+E 62.5% -> T+E+K 81.3%), a conservative lower bound given
  memorisation.
- Clean hallucination separation: T 28.5% vs T+E/T+E+K 5.0%.
- Finding that EHR context *hurts* on contraindication questions
  (T 0.8838 > T+E 0.8708).
- Mechanistic H2 result: KG retrieval is keyed off the diagnosis list, so
  high EHR sparsity *removes* the KG's entry point rather than increasing
  its value.

### Status: strategic decision required from project owner before further compute

---

## 2026-08-11 (later) — Full oracle labels complete (train+val); confound assessment revised

Both splits finished cleanly: val 100/100, train 200/200, **0 failures**
(resume-from-checkpoint worked after the CUDA crash). Echo rate 0.0 across
all modes on both splits. Hallucination T 0.285 vs T+E/T+E+K 0.05 — again
matching the pre-registered targets.

### REVISION to the 2026-08-11 "KG memorisation is fatal" assessment

Earlier entry called Confound 2 a threat to the central question. With
n=32 contraindication questions in the train split, the picture is more
favourable than the val-only view suggested:

| | exact-correct on contraindication_check (n=32) |
|---|---|
| T (no EHR, no KG) | 20/32 = 62.5% |
| T+E (EHR, no KG) | 20/32 = 62.5% |
| **T+E+K (EHR + KG)** | **26/32 = 81.3%** |

KG retrieval adds **+18.8 percentage points** over the memorisation floor.
Crucially, memorisation **inflates the no-KG baseline**, which biases the
measured KG effect *downward*. A positive KG result obtained despite
memorisation is therefore a **conservative lower bound**, not an artifact —
it does not manufacture an effect, it hides part of one. This weakens but
does not invalidate the T+E+K claim. It must still be disclosed, and the
clean version of the experiment (KG questions excluded from fine-tuning)
would be expected to show a *larger* gap, not a smaller one.

Also note T (0.8838) > T+E (0.8708) on this question type: adding the EHR
snapshot **actively hurts**, apparently by leading the model to affirm a
drug it sees in the patient's medication list. That is itself a reportable
finding about EHR-only grounding.

### The actual blocker is class support, not contamination

| class | train | val |
|---|---|---|
| T | 33 | 19 |
| T+E | 157 | 79 |
| **T+E+K** | **10** | **2** |

Two T+E+K examples in val. Per-class F1 for that class is statistically
meaningless, and the pre-registered Macro-F1 target of 0.60-0.75 is
unreachable at this support. 6 of the 10 T+E+K train labels come from
contraindication_check — the KG question type is doing its job, there is
simply far too little of it (32/200 = 16% of questions).

Root cause chain: 6 of 7 question types are extractive template questions
whose answers are verbatim EHR fields → T+E saturates (composite 1.0000 on
diagnoses/primary_diagnosis/next_step) → **93% of T+E vs T+E+K pairs are
quality-identical** → T+E+K can only win on the one KG-dependent type,
which is a small minority of the question mix.

### H2 signal is INVERTED — and there is a mechanistic explanation

| sparsity | T | T+E | T+E+K |
|---|---|---|---|
| high | 42.1% | 57.9% | **0.0%** |
| medium | 13.2% | 82.4% | 4.4% |
| low | 14.2% | 79.6% | 6.2% |

Pre-registered target was 50-70% T+E+K in the HIGH bucket; observed is 0%.
The direction is reversed, and the mechanism is clear: **KG retrieval is
keyed off the patient's diagnosis list** (`find_matching_diseases()` matches
diagnosis text against Disease nodes). High EHR sparsity means *few
diagnoses*, so there is nothing to match against and no KG facts are
retrieved. The KG cannot fill a gap when the gap includes the very field it
is indexed on.

This contradicts H2 as originally stated, but it is a legitimate,
publishable mechanistic finding — arguably more interesting than the
predicted result. It should be reported as such rather than engineered away.
(Note the high bucket is only n=19; treat cautiously.)

### Status

Oracle labels are trustworthy for the first time in this project: 0% echo,
88% tie rate now correctly *reported* rather than hidden, real quality
separation between modes. The remaining problems are experimental-design
(question mix), not implementation defects.

---

## 2026-08-11 — Oracle val results: fixes confirmed, but two new confounds found

### Crash (recoverable)

`router_val` completed 100/100 cleanly. `router_train` then died at question
2/200 with `CUDA error: an illegal memory access was encountered` — the third
GPU-level fault on this machine (cf. 2026-08-07 illegal instruction + BSOD).
Not a code defect; consistent with the ongoing hardware/driver instability.
`data/router/checkpoints/router_train_checkpoint.parquet` survived with 10
completed questions, and `CheckpointManager` will resume from there.

### Confirmed: the pre-training audit fixes worked

| Metric | Broken run (2026-08-09) | This run |
|---|---|---|
| context_echo_rate (T/T+E/T+E+K) | 52-63% | **0.0 / 0.0 / 0.0** |
| mean composite T | 0.5237 | 0.5591 |
| mean composite T+E | 0.5853 | **0.9591** |
| mean composite T+E+K | 0.5758 | **0.9673** |
| halluc_rate T | 0.0 (detector dead) | **0.26** |
| halluc_rate T+E / T+E+K | 0.0 | **0.02 / 0.02** |

The hallucination pattern (T 26% vs T+E/T+E+K 2%) closely matches the
pre-registered targets in `Target_Benchmarks_Success_Criteria.docx`
(T 12-22%, T+E 3-8%). The generator is now behaving correctly.

**KG value is demonstrable where it should be.** On `contraindication_check`
questions T+E+K beats T+E outright in 4/20 cases, and the failures are
clinically serious ones:

    Q: Would prescribing Lisinopril be appropriate for this patient's Acute Kidney Injury?
    GOLD : No, Lisinopril is contraindicated in Acute Kidney Injury (...)
    T+E  : Yes, Lisinopril is a standard first-line treatment for AKI.   <- WRONG, unsafe
    T+E+K: No, Lisinopril is contraindicated in Acute Kidney Injury.     <- correct

### Confound 1 (CONFIRMED) — ceiling effect makes KG benefit unmeasurable on 6/7 question types

| question_type | n | comp T | comp T+E | comp T+E+K | T+E+K − T+E |
|---|---|---|---|---|---|
| contraindication_check | 20 | 0.9525 | 0.9292 | 0.9704 | **+0.0412** |
| diagnoses | 14 | 0.5177 | 1.0000 | 1.0000 | 0.0000 |
| primary_diagnosis | 14 | 0.5326 | 1.0000 | 1.0000 | 0.0000 |
| lab | 13 | 0.2527 | 0.9603 | 0.9603 | 0.0000 |
| medication | 13 | 0.1743 | 0.8757 | 0.8757 | 0.0000 |
| next_step | 13 | 0.6770 | 0.9864 | 0.9864 | 0.0000 |
| summary | 13 | 0.6001 | 0.9716 | 0.9716 | 0.0000 |

T+E scores **exactly 1.0000** on diagnoses/primary_diagnosis and ≥0.95 on
85% of all val questions. T+E and T+E+K emit **byte-identical answers in
96/100 cases**. This is the original audit's finding #5 (template answers
are copies of EHR fields) showing up quantitatively: once the EHR snapshot
is in the prompt, the answer is already present verbatim and there is
literally no headroom for KG to contribute. T+E+K's 2% label share is
therefore mostly a *ceiling artifact*, not evidence that the KG is useless.

### Confound 2 (CONFIRMED, more serious) — fine-tuning leaked KG knowledge into the weights

On `contraindication_check`, **T scores 0.9525 — higher than T+E's 0.9292**,
despite T having neither EHR nor KG context. A model with no access to the
contraindication facts should not answer contraindication questions well.

Explanation: `contraindication_check` pairs were generated from
`mkg/edges/ontology_edges.csv` and ~180 of them went into the *fine-tuning*
split. Disease→drug contraindications are **patient-independent**, so the
patient-level split in `splits/patient_splits.json` provides no protection
whatsoever — the exact fact "Metformin is contraindicated in CKD-3" learned
from a finetune patient transfers directly to a router-val patient. The
model has memorised the KG into its weights and no longer needs to retrieve
it.

This is a genuine threat to the paper's central question. "When does the KG
help?" cannot be answered with a generator that already knows the KG. A
reviewer would identify this immediately.

### Confound 3 (CONFIRMED) — no H2 signal in the current labels

sparsity × best_mode crosstab shows no shift toward T+E+K as sparsity rises
(high: 11 T+E / 0 T+E+K; low: 48 T+E / 1 T+E+K), against a pre-registered
target of 50-70% T+E+K in the high bucket. Given Confounds 1 and 2 this is
expected and is not independent evidence about H2.

### Diagnostic bug fixed

`selection_is_tie()` compared **cost-adjusted** scores, so two modes with
byte-identical answers differed by exactly the cost delta (0.001 > 
TIE_EPSILON) and were counted as a real quality difference. It reported
`tie_decided_label_rate: 0.0` while 96% of T+E/T+E+K pairs were in fact
quality-identical. Now measured on raw composites.

### Status / decisions required

Fixes to the generator are confirmed working. The blockers are now
**experimental-design** issues, not bugs, and they change what the study can
claim. Options recorded for the project owner (no action taken unilaterally):

1. **Report honestly as-is** — "KG helps only on relational/guideline
   questions; on extractive EHR questions there is no headroom." The
   pre-registered targets doc explicitly allows this as a valid finding.
   Confound 2 must still be disclosed as a limitation.
2. **Remove `contraindication_check` from the fine-tuning split** and
   retrain (~13h), so KG facts are not memorised and T+E+K's advantage
   measures genuine retrieval benefit. This is the scientifically strongest
   option and directly addresses Confound 2.
3. **Add harder, non-extractive question types** whose answers are not
   verbatim EHR fields, to break the ceiling in Confound 1.

---

## 2026-08-11 — Retrained model VALIDATED: context echoing eliminated

### Training run outcome

Completed via **early stopping at step 165/240** (epoch 1.375) — the callback
added in the pre-training audit fired exactly as designed after 3 consecutive
non-improving evals past the best.

eval_loss: 0.8095 → 0.4606 → 0.3325 → 0.3242 → 0.3201 → 0.3281 → 0.2855 →
**0.2548 (best, step 120)** → 0.2591 → 0.2619 → 0.2573 → stop.
`load_best_model_at_end` correctly restored checkpoint-120's weights
(verified in `trainer_state.json`: `best_model_checkpoint = checkpoint-120`,
`best_metric = 0.2548`).

Contrast with the broken run, which collapsed to eval_loss 0.052 by epoch 0.5
and flatlined. The **higher** absolute loss here is the healthy signal: 0.052
was the model succeeding at the wrong task (context continuation); ~0.25 is
what genuine extractive clinical QA costs. `training_metadata.json` is valid
JSON this time (numpy encoder fix confirmed) and records dropout 0.10,
max_length 1524, optim adamw_8bit — all audit fixes applied and traceable.

Wall clock 13h17m, down from 27h48m, as predicted from the shorter sequences.

### Post-training gate: context-echo test

15 generations (5 questions x 3 modes) on router-train data (held-out eval
set deliberately untouched). Raw detector output was 2/15 = 13.3%, but
**both flagged cases were false positives** — e.g.:

    [T+E] Q: What condition was this admission mainly for?
          GOLD: Mitral valve disorders
          PRED: Mitral valve disorders   <- flagged "echo"

That is a perfect answer. **True echo rate: 0/15.**

### Bug found in the echo detector itself (fixed)

The detector added on 2026-08-10 used `answer[:60] in prompt_context`, which
is invalid for extractive QA: a correct answer to "what was the diagnosis?"
is by definition a span that appears in the context. Left unfixed it would
have reported a large phantom echo rate in the next oracle run and could
have triggered an unnecessary third retrain.

Rewrote `is_context_echo()` to require the genuine pathology's two
distinguishing properties: (1) the output is long (>=150 chars — real gold
answers are mean 24 / max 68 tokens), and (2) it reproduces the context's
*structure* (`## ` / `[Passage ` scaffolding) or its *opening* rather than
extracting a mid-context span.

Validated against ground truth:

| Test | Result |
|---|---|
| Archived degenerate-run outputs (known bad) | flags **70.4% / 64.3% / 64.3%** (T/T+E/T+E+K) — higher than the original 52-63% estimate, since the old detector's `[:2000]` prefix cap under-counted |
| 6 hand-built correct extractive answers (known good) | **0 false positives** |
| Synthetic true echo | correctly flagged |

### Research-relevant observation: the modes now genuinely differentiate

From the sample generations, on patient-specific questions:

- **T** produces confidently wrong answers ("Other diseases of anus and
  rectum" for a mitral-valve admission) — correct behaviour, since T has no
  EHR access and cannot know the diagnosis.
- **T+E / T+E+K** produce exactly correct answers.

This is precisely the signal the oracle needs and which the broken model
could not provide: previously all three modes emitted near-identical context
echoes, producing the ~89% tie rate that made the labels meaningless. T+E and
T+E+K still agree on these diagnosis/lab questions (expected — KG adds
nothing there); the `contraindication_check` question type is where T+E+K
should separate, and that is now testable.

### Status

Generator validated. Cleared to regenerate oracle labels.

---

## 2026-08-10 — Pre-training audit: TRUE root cause of context echoing found

Full-pipeline audit before committing another ~27h GPU run. **The 2026-08-09
"overfitting" hypothesis was WRONG and is retracted.** The real causes are
mechanical and were proven with the actual tokenizer on the actual data.

### Finding A (CONFIRMED, primary) — loss was computed over the context

`load_and_prep_data()` emitted a single concatenated `{"text": ...}` field
and `SFTConfig` used `dataset_text_field="text"` with `completion_only_loss`
left at its default of `None`. Verified against the installed TRL 1.6.0 API
(`completion_only_loss` and `assistant_only_loss` both exist; neither was
set). Result: cross-entropy ran over **every** token, prompt included.

Measured with the real MedGemma tokenizer: gold answers are mean 24 / p99 52
/ max 68 tokens; assembled contexts are mean 2085 (T), 2396 (T+E), 2558
(T+E+K). So **>99% of the gradient signal was literally "reproduce the
retrieved clinical passages."** Validated directly: for a representative
example, 473/476 tokens were supervised under the old config = 99.37% of
loss on context reproduction. The model echoed context because that is
precisely and almost exclusively what it was trained to do.

### Finding B (CONFIRMED, primary) — truncation deleted the question and answer

`max_length=768` combined with HF/TRL's default `truncation_mode="keep_start"`
(confirmed by introspection). Measured: **88.3% of assembled contexts exceed
768 tokens** (86.4% exceed 1024; 68.1% exceed 2048). Because the sequence is
ordered context → question → answer, keep_start truncation discarded the
question AND the gold answer outright in ~88% of training examples. Those
examples contained *only* context tokens — and, per Finding A, full loss was
applied to them. Note this bug was introduced by the 2026-08-06 retrieval-
augmentation change itself (the original `max_length=1024` was already too
small once prompts became retrieval-augmented); lowering it to 768 during
the 2026-08-07 VRAM work made it worse but did not create it.

### Finding C (CONFIRMED) — three different truncation regimes across the pipeline

training truncated at 768 · `run_evaluation.py` truncated at 2048 (so the
question was cut at *inference* too, for 68% of contexts) ·
`oracle_labels.py` applied no truncation at all. The oracle therefore scored
modes under conditions matching neither training nor final evaluation.

### Finding D (CONFIRMED) — generation budget mismatch

`oracle_labels.py` generated 128 new tokens; `run_evaluation.py` generated
256. The mode selected as "best" by the oracle was chosen under a different
generation budget than the answers reported in the results table.

### Finding E (CONFIRMED) — naive truncation would have destroyed the experiment

Important design trap avoided: `RetrievalResult.prompt_context` orders
sections passages → EHR → KG. Any tail-truncation fix (the obvious one)
strips EHR and KG *first* — deleting exactly what distinguishes T+E and
T+E+K from T, silently collapsing the three modes toward identical prompts
and manufacturing "the KG doesn't help" as an artifact. Per-section
budgeting was implemented specifically to prevent this.

### Finding F (CONFIRMED, critical) — the next run would have resumed the broken model

`models/medgemma-4b-qlora/` still contained `checkpoint-210/225/240` from
the degenerate run. `get_last_checkpoint()` validates and resumes from the
newest complete checkpoint — so starting training would have **resumed the
context-echoing model** rather than training fresh, silently carrying the
failure forward into another 27h run.

### Finding G (CONFIRMED) — oracle tie-breaking bias (previously deferred, now fixed)

`OracleSelector.select()` used `max(dict, key=...)`, which resolves ties to
whichever key iterates first — always `"T"`. In the 2026-08-09 run, 102/114
(train) and 55/65 (val) of T's "wins" were exact ties. Now an explicit,
documented cheapest-mode-wins-within-TIE_EPSILON policy (which is the
correct policy for H1's cost argument), plus a per-row `label_was_tie` flag
and a `tie_decided_label_rate` in the report so this is never invisible again.

### Fixes applied

| # | Fix | Files |
|---|---|---|
| A | prompt/completion dataset schema + `completion_only_loss=True`; dataset reduced to exactly `{prompt, completion}` so TRL's masked path engages | `train_qlora.py` |
| B | New per-section token budgeting module; hard assertion aborts training if any sequence would overflow | `context_budget.py` (new), `train_qlora.py` |
| C | One shared `MAX_SEQ_LENGTH=1524` across training / router-dataset build / evaluation | `context_budget.py`, `train_qlora.py`, `build_router_dataset.py`, `run_evaluation.py` |
| D | One shared `MAX_NEW_TOKENS=128` for oracle and evaluation | `context_budget.py`, `oracle_labels.py`, `run_evaluation.py` |
| E | Fixed per-section budgets (passages 600 / EHR 350 / KG 350) keep the passage block **identical across modes**, so T/T+E/T+E+K differ only by the additive EHR/KG content | `context_budget.py` |
| F | Degenerate model + contaminated router + bad oracle labels + n=1 eval outputs archived to `models/_ARCHIVE_degenerate_run_2026-08-08/` | filesystem |
| G | Explicit tie policy + `label_was_tie` / `tie_decided_label_rate` / `context_echo_rate_per_mode` diagnostics | `oracle_labels.py` |

### Cheap validation performed (CPU only, no training)

1. **Budgeting**: 120 contexts across all 3 modes — zero overflow, zero
   sections lost. Budgeted sizes: T mean 437/max 606, T+E mean 752/max 944,
   T+E+K mean 952/max 1300.
2. **Mode isolation**: T has no EHR/KG; T+E has EHR only; T+E+K has both;
   and the passage block is byte-identical across all three modes for the
   same question — confirming modes now differ *only* by added EHR/KG.
3. **Loss masking**: 473 prompt tokens masked to -100, 3 answer tokens
   supervised; decoded supervised span == the gold answer exactly.
4. **Real pipeline** (actual `load_and_prep_data`, 10 examples, real
   retriever + Neo4j): schema exactly `{prompt, completion}`, mode
   distribution balanced, retrieval fallback 0.0%, full-sequence max 1312 <
   1524, question verifiably present in the prompt, completion intact
   (sample was a `contraindication_check` item — the new KG-grounded type,
   confirming those flow through correctly).

### Expected secondary benefit

Sequences drop from ~2575 (truncated to 768) to ~500–1300 real tokens, and
only ~24 tokens per example now carry gradient. Combined with early
stopping, the next run should be substantially shorter than 27h and place
less sustained load on the GPU that BSOD'd twice.

### Residual risks (documented, not blocking)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` logs "not supported on
  this platform" on Windows — the fragmentation mitigation is inert here.
  Harmless, but it is not providing the protection intended on 2026-08-07.
- Hardware stability (BSOD root cause never definitively confirmed) is
  unchanged by any of this.
- Answers are short (mean 24 tokens), so supervised signal per example is
  small — expected for extractive clinical QA, but it means eval_loss will
  look low in absolute terms and should not be over-interpreted.
- Passage budget of 600 tokens means fewer passages reach the prompt than
  FAISS's top_k=5 retrieves; this is applied identically at train and
  inference, but should be stated in the paper as a 6GB-VRAM design
  constraint rather than left implicit.

---

## 2026-08-09 — Oracle labels degenerate: generated "answers" are context echoes

### Trigger

`python -m src.router.oracle_labels` ran successfully end to end (199/200
train, 100/100 val, no failures) but the label distribution looked wrong on
its face and inverted from the pre-fix run:

|  | T | T+E | T+E+K |
|---|---|---|---|
| Pre-fix (stale model+data) | train 3% / val 2% | train 80.5% / val 79% | train 16.5% / val 19% |
| Post-fix (this run) | train 57% / val 65% | train 39.5% / val 33% | train 3% / val 2% |

`mean_metrics_per_mode` from the run's own report JSON immediately
contradicted the win-rate story: T's mean composite score (train 0.528, val
0.524) is clearly *lower* than T+E's (0.587/0.585) and T+E+K's
(0.581/0.576) — yet T "wins" the majority of individual questions. That
contradiction was the signal something specific (not just "the numbers look
off") was wrong.

### Investigation

1. Per-question margin analysis: when `best_mode == "T"`, the median and
   25th-percentile margin over T+E (`composite_t - composite_te`) is
   **exactly 0.000** — 102/114 (train) and 55/65 (val) of T's "wins" are
   exact ties, not real quality advantages.
2. `OracleSelector.select()` uses `max(adjusted_scores, key=adjusted_scores.get)`
   over a dict built in the fixed order `["T", "T+E", "T+E+K"]`. Python's
   `max()` on ties returns the *first* maximal key in iteration order — so
   every exact tie silently resolves to `"T"`. This is a real bug (silent,
   undocumented tie-breaking bias), but it's a symptom, not the root cause —
   ties this frequent shouldn't be happening at all.
3. Why so many exact ties: checked `answer_t == answer_te` (identical
   generated text) directly — **54–57% of rows** have byte-identical
   generated answers between T and T+E (and ~50-59% between T and T+E+K).
4. Inspected the actual generated text. It is not a real answer to the
   question in a majority of cases — it's the model **echoing its own
   retrieved-passage context verbatim**, cut off mid-word at exactly
   `max_new_tokens=128`:
   ```
   Question: What condition was this admission mainly for?
   Reference: Mitral valve disorders
   answer_t: '## Retrieved Clinical Passages\n[Passage 1 | DS]\npatient is
   very confused about why she is here... ATRIAL FIBRILLATION ON COUM'
   ```
   Quantified across all 3 modes: 63.3% (T), 53.8% (T+E), 52.3% (T+E+K) of
   answers are prefix-matches of their own input prompt context — the model
   is continuing/predicting its input rather than performing QA, roughly
   half to two-thirds of the time, across every mode. T and T+E frequently
   echo *identically* because they share the same retrieved-passages text
   at the start of their prompt, and the echo rarely runs long enough to
   reach the mode-specific EHR/KG content that would make them diverge.

This fully explains the label distribution: T's apparent dominance is
almost entirely a tie-breaking artifact of degenerate, near-identical
outputs across modes, not genuine T-mode superiority; T+E+K's collapse to
2-3% follows because the same degeneracy affects it too, so it rarely wins
outright even when it doesn't tie.

Ruled out: the earlier hallucination-heuristic length-bias hypothesis
(finding #7) — `hallucination_diagnostic_per_mode` in the report JSON shows
halluc_rate ≈ 0 for all three modes, so that mechanism isn't in play here.
Also ruled out a prompt-templating/generation-slicing bug:
`AnswerGenerator.generate()`'s token slicing
(`outputs[0][inputs["input_ids"].shape[1]:]`) is correct — it isn't
returning the input, the *model* is generating text that happens to
reproduce the input. The pre-audit-fix model, using the same templating
code path, did not exhibit this (it generated real, if often wrong, answers
— e.g. "Possible cervical spondylotic myelopathy" seen in the very first
n=1 eval sample from the original audit).

### Root cause assessment

Ties back directly to the overfitting signature already noted in the
2026-08-08 training-completion log entry: `eval_loss` plateaued at ~0.052
by roughly **epoch 0.5 (~60 of 240 steps)** while `train_loss` kept falling
to ~0.03 for the remaining 1.5 epochs. The adapter most likely learned a
degenerate shortcut — continue the input rather than answer the question —
during that extended past-convergence training. Also compounded by finding
#12/13-adjacent: this model was trained on the stale `data/qa` path (fixed
2026-08-09), so it never saw `contraindication_check` examples either, but
the echo-degeneracy affects the *original* template question types too, so
overfitting (not just the missing question type) is the primary suspect.

### Decision: do not proceed to router training

Building oracle labels, a router, or a final evaluation on top of a
generator that fails to perform QA at all in 50-63% of generations would
produce uninterpretable downstream results regardless of how correct the
router training procedure itself is. Recommended and agreed: retrain
QLoRA before regenerating oracle labels.

### Fixes applied

- `src/model/train_qlora.py`: added `EarlyStoppingCallback
  (early_stopping_patience=3)` to `SFTTrainer` — training now stops
  automatically once `eval_loss` shows no improvement for 3 consecutive
  eval checks (`eval_steps=15` each, so up to ~45 steps of buffer past the
  best point) instead of always running the full `num_train_epochs=2`
  regardless of convergence. Requires `load_best_model_at_end=True`
  (already set) to restore the best weights. Expected side benefit: should
  also substantially cut wall-clock training time (the 2026-08-08 run's
  eval_loss stopped improving by ~step 60 of 240), reducing exposure to the
  GPU-stability issues from the 2026-08-07 incidents.
- `LORA_DROPOUT`: 0.05 → 0.10 (standard regularization response to this
  overfitting signature).
- `OracleSelector`'s silent tie-breaking-toward-"T" bug noted but **not**
  fixed yet — deliberately deferred: fixing the underlying degenerate-
  generation problem should make near-exact ties rare again; if they
  persist after retraining, the tie-break behavior should be made explicit
  (e.g. tie → prefer the cheaper mode intentionally, logged as such) rather
  than silently falling out of dict iteration order.

### Not yet done

- Retrain QLoRA with the above fixes (project owner to run — GPU-heavy).
- Re-run oracle label generation after retraining and re-check: (a) the
  context-echo rate should drop sharply, (b) `OracleSelector` tie rate
  should drop correspondingly, (c) label distribution should reflect real
  quality differences rather than tie artifacts.
- If echoing persists even after retraining with early stopping + higher
  dropout, broaden investigation to the chat-template/generation-prompt
  mechanics specifically (not yet fully ruled out as a contributing factor,
  only deprioritized given the overfitting evidence is a simpler, more
  direct explanation).

---

### Not yet done / open questions for next attempt

- Confirm whether the retry completes further without crashing.
- If it crashes again: capture `nvidia-smi -l 2` (or Windows Task Manager's
  GPU tab, watching Dedicated vs. Shared GPU memory) *during* the run, and
  re-run with `CUDA_LAUNCH_BLOCKING=1` for a synchronous, precise stack
  trace pinpointing the exact failing kernel.
- Verify installed `bitsandbytes` / driver versions and driver channel
  (Studio vs. Game Ready) inside the actual `ehr-rag` conda environment —
  not checked yet (no access to that environment from the investigating
  session).
- Confirm AC power (not battery) during training.
- If VRAM oversubscription is still implicated after these fixes, consider
  capping retrieved-note context length specifically (would require caution:
  any change to `Retriever`'s `TOP_K` affects T/T+E/T+E+K definitions
  everywhere, not just training — should not be changed without also
  reconsidering oracle-label generation, router-dataset build, and final
  eval, which use the same `Retriever`).

---

## 2026-08-06 — Full Repository Audit (Phase 1–3)

### Context

The pipeline runs end-to-end (lakehouse → MKG → retrieval → QLoRA fine-tuning →
router → held-out evaluation) but reported research results are far weaker than
expected. Original design docs (`doc/Adaptive_RAG_Project_Documentation.pdf`,
target: Phi-3 Mini) diverge from the implementation (actual: MedGemma-1.5-4B),
which is an intentional, documented substitution — not itself a bug.

Goal of this pass: read the entire repository and design docs, and identify every
implementation bug, scientific-validity flaw, and train/inference mismatch that
could explain the weak results, before touching any code.

### Method

Read: both design docs, README.md, and the full source tree (`src/lakehouse`,
`src/ehr`, `src/mkg`, `src/retrieval`, `src/router`, `src/model`, `src/qa`,
`src/evaluation`), plus all data/model artifacts in `experiments/`, `models/`,
`data/`, `mkg/`, `splits/`, and pre-existing debugging scratch files
(`debug_router.py`, `debug_output.txt`) left in the repo from a prior manual
investigation session.

### Findings (ranked by severity; full detail in the audit report delivered in-chat)

1. **CRITICAL — Final held-out evaluation only ever ran on n=1 question.**
   `experiments/results/final_eval/` (the directory the paper's Table 2 would be
   built from) contains a single evaluated question, not the full 300-question
   `ehrqa_eval.parquet` set. Every "final" metric currently on disk is
   statistically meaningless.
   — `src/evaluation/run_evaluation.py`, `experiments/results/final_eval/*`

2. **CRITICAL — Router predictions are overridden by un-validated hardcoded
   thresholds at eval time, never used during training/validation.**
   `run_evaluation.py::router_predict()` overrides the trained XGBoost's argmax
   decision with `THRESHOLD_TEK = 0.30` / `THRESHOLD_T = 0.35` cutoffs (comment:
   "Priority 3: Threshold Calibration"). `train_router.py`'s own reported
   metrics (`Accuracy=0.60`, worse than the majority-class baseline's `0.79`)
   reflect plain `argmax`, not this rule. This is a train/inference mismatch:
   the "Router" system in the final results is not the same decision function
   that was trained and validated.
   — `src/evaluation/run_evaluation.py:250-266`

3. **CRITICAL — The `sparsity_bucket` categorical feature fed to the router is
   silently broken and largely constant.** `feature_pipeline.py`'s
   `bucket_map = {"very_sparse":0,"sparse":1,"medium":2,"dense":3,"unknown":-1}`
   does not match the actual bucket vocabulary produced anywhere in the
   pipeline (`"low"/"medium"/"high"`, confirmed in `sparsity.py` and in
   `per_question_results.csv`). Only `"medium"` happens to match; `"low"` and
   `"high"` both silently collapse to the same `-1` fallback. This directly
   destroys the router's ability to use the sparsity signal that H2 is about.
   — `src/router/feature_pipeline.py:72-99`, confirmed via
   `models/router/feature_importance.csv` and `models/router/bucket_encoder.json`

4. **CRITICAL — MedGemma was fine-tuned on a context format it never sees at
   inference.** `train_qlora.py` only ever trains on a flattened EHR-snapshot
   string (`CONTEXT_MODE = "T"`, with an explicit code comment acknowledging
   T+E/T+E+K training data is "not yet wired up... flagged for follow-up").
   The model never sees retrieved note passages or KG facts during training,
   yet at inference (`run_evaluation.py` / `retriever.py`) it is prompted with
   three structurally different context formats (T = passages only, no EHR;
   T+E = passages+EHR; T+E+K = passages+EHR+KG), none of which match the
   training format. This is a severe, self-documented train/inference
   distribution mismatch and a strong candidate root cause for the weak T-mode
   results (hallucinated, off-topic answers) and for T+E ≈ T+E+K
   (KG content is out-of-distribution either way).
   — `src/model/train_qlora.py:55-63,236-263`

5. **HIGH — Synthetic QA reference answers are template-copies of the same
   structured fields shown in the EHR context**, e.g. `primary_diagnosis`
   answer = `diagnoses[0]`, the literal top ICD description that also appears
   verbatim in the T+E/T+E+K prompt's "Diagnoses:" line. This makes the T vs.
   T+E/T+E+K comparison partly mechanical (context contains the literal
   answer vs. doesn't) rather than a genuine test of retrieval usefulness,
   confounding H1/H2 for the structured question types.
   — `src/qa/generate_synthetic_ehrqa.py:504-562`

6. **HIGH — MKG is ~10–25x smaller than the design target.** Design doc target:
   1,000–3,000 nodes / 5,000–10,000 edges over 20–50 diseases. Actual (per
   `debug_output.txt`): 114 nodes / 385 edges (25 diseases, 36 symptoms, 31
   labs, 22 drugs — roughly 1 symptom/lab/drug relation per disease). 65% of
   edges are `CO_OCCURS_WITH_LAB` (data-driven), leaving very thin
   ontology-sourced coverage. This likely limits how often KG facts can add
   real signal, independent of any bug — a legitimate but under-reported
   scope gap vs. the stated design.
   — `src/mkg/seed_diseases.py`, `debug_output.txt` Step 1

7. **MEDIUM — Oracle label hallucination penalty uses a crude keyword
   heuristic** (`HallucinationDetector` in `oracle_labels.py`) to subtract
   `0.30` from the composite score used to pick the "best" mode per question.
   Longer contexts (T+E+K) have more surface area for keyword false-positives
   (e.g. "mg" appearing anywhere in a large context), which could
   systematically bias oracle labels away from T+E+K independent of true
   answer quality. Train-set oracle distribution is heavily skewed toward T+E
   (161/200 train, 79/100 val) — consistent with this bias but not yet proven
   causal.
   — `src/router/oracle_labels.py:266-290`

8. **MEDIUM — BERTScore in the final eval silently fails and defaults to
   0.0** for every system (exception swallowed, only a console `[WARN]`,
   no persisted error). Confirmed in `full_results.json`/`summary_table.csv`
   — the field is `0.0` across all 5 systems despite BLEU/ROUGE being
   nonzero for 4 of them.
   — `src/evaluation/run_evaluation.py:332-350`

9. **MEDIUM — Duplicate, inconsistent sparsity implementations.**
   `src/ehr/sparsity.py` and `src/lakehouse/sparsity.py` both compute the same
   thing and write to the same output path, but use different bucket casing
   (`"High"` vs `"high"`). Whichever ran last silently determines behavior;
   this is exactly the kind of divergence that produced finding #3's
   downstream mismatch.
   — `src/ehr/sparsity.py`, `src/lakehouse/sparsity.py`

10. **LOW — Hallucination/faithfulness metrics reported in the paper's planned
    tables are crude string-overlap heuristics** (`ehr_contradiction_score`,
    `unsupported_score` in `run_evaluation.py`), not the human-annotated,
    Cohen's-κ-validated taxonomy the design doc specifies (§9). No
    KG-contradiction category is implemented at all. This is a fidelity gap
    between the documented methodology and what actually runs.
    — `src/evaluation/run_evaluation.py:353-381`

11. **LOW — Dead/duplicate retrieval module.** `src/retrieval/unified_retrieval.py`
    duplicates `src/retrieval/retriever.py` with a different embedding model
    (`all-MiniLM-L6-v2` vs. the actually-used `BAAI/bge-small-en-v1.5`) and is
    not imported anywhere in the live pipeline. Debug `print()` statements
    left in `retriever.py`/`run_evaluation.py` production paths.
    — `src/retrieval/unified_retrieval.py`, `src/retrieval/retriever.py:220-244`

### Positive findings (things that are NOT the problem)

- Patient-level train/router/eval splits are correctly disjoint, with
  assertion-based leakage checks (`make_splits.py`, `generate_synthetic_ehrqa.py`).
  No evidence of leakage across splits.
- Oracle label generation's core scoring (BERTScore via `roberta-large`,
  ROUGE-L, EM) is methodologically reasonable; only the hallucination-penalty
  component (finding #7) is questionable.
- `train_router.py`'s training procedure itself (class-weighted XGBoost,
  proper train/val separation, standard baselines) is sound — the problem is
  what happens to its output *after* training (findings #2, #3).

### Status

Audit complete. Findings presented to project owner, who chose: (a) fix
everything in dependency order, including retraining QLoRA on aligned
context and redesigning the QA templates, and (b) expand the MKG rather than
just document its scope as a limitation. All code changes below were made in
this same session, in dependency order, before any GPU-heavy step was run.

---

## 2026-08-06 — Fixes Applied (Phase 4, same session as audit)

Two additional bugs were found *while implementing* the fixes above — both
are documented here as new findings since they weren't in the original audit
pass.

**Finding #12 (new) — `n_meds` silently always 0.0 at eval time.**
`run_evaluation.py`'s router feature vector was built from `sparsity.parquet`
alone, which has no `n_meds` column (medication count is a real feature
XGBoost was trained on, populated correctly during training via
`generate_synthetic_ehrqa.py`'s `get_structural_features()`). Fixed by
sourcing all five router features (`n_labs`, `n_diag`, `n_meds`,
`sparsity_score`, `sparsity_bucket`) from `PatientSnapshot.get()` at eval
time too, exactly mirroring how training data is built.

**Finding #13 (new) — oracle-label generation and final-eval generation used
different prompt formats.** `oracle_labels.py`'s `AnswerGenerator.generate()`
built the user message as `"{context}\n\nQuestion: {question}"`, while
`run_evaluation.py`'s `generate()` used `"Context:\n{context}\n\nQuestion:
{question}"` — a real formatting drift between the prompt used to score/pick
oracle labels and the prompt used to generate final reported answers. Fixed
by extracting the single canonical format into `src/model/prompts.py::
build_user_message()`, now imported by `train_qlora.py`, `oracle_labels.py`,
`run_evaluation.py`, and `test_medgemma.py`. This module also fills a gap in
the original design (`src/model/prompts.py` was listed in the README's
project structure but never existed).

### Code changes (see this repo's diff for full detail; summarized per finding)

| # | Finding | File(s) | Change |
|---|---|---|---|
| 1 | n=1 final eval | `src/evaluation/run_evaluation.py` | Added `MIN_FINAL_EVAL_SAMPLES=50` hard guard; refuses to write to `final_eval/` below that unless `--allow-small-sample` is passed |
| 2 | Router threshold override | `src/evaluation/run_evaluation.py::router_predict()` | Removed the hardcoded `THRESHOLD_TEK`/`THRESHOLD_T` override; now plain `argmax(predict_proba)`, identical decision rule to `train_router.py`'s reported metrics |
| 3 | Broken sparsity_bucket encoding | `src/router/feature_pipeline.py::HybridFeaturePipeline.bucket_map` | Fixed vocabulary to `{"low":0,"medium":1,"high":2,"unknown":-1}`, matching the real values `src/lakehouse/sparsity.py` produces |
| 4 | Train/inference context mismatch | `src/model/train_qlora.py` | `load_and_prep_data()` now calls the real `Retriever` per training example with a reproducibly sampled mode (T/T+E/T+E+K, configurable via `CONTEXT_MODE_WEIGHTS`), building the exact same `RetrievalResult.prompt_context` format used at inference. Preflight checks now require the FAISS index to exist first. |
| 5 | Answer-copying confound | `src/qa/generate_synthetic_ehrqa.py` | Added a new `contraindication_check` question type grounded in `mkg/edges/ontology_edges.csv` CONTRAINDICATED_WITH/FIRST_LINE_TREATMENT facts — its correct answer is not recoverable from the EHR snapshot alone, giving T+E+K a question type where any advantage reflects genuine KG use. Existing template types (primary_diagnosis, etc.) were left as-is; their confound is now documented rather than silently present. |
| 6 | Undersized MKG | `mkg/edges/ontology_edges.csv`, `src/mkg/cooccurrence.py` | Ontology edges expanded 135→215 rows (deeper symptom/lab/treatment/contraindication coverage per the same 25 seed diseases); `TOP_N_LABS_PER_DISEASE` raised 10→15. README's MKG section rewritten to state actual scope honestly instead of the original 1k-3k/5k-10k target. |
| 7 | Hallucination-heuristic length bias (unconfirmed) | `src/router/oracle_labels.py::ReportGenerator` | Not changed blindly — added `hallucination_diagnostic_per_mode` (halluc rate + mean prompt length per mode) to the oracle report JSON so the length-bias hypothesis is empirically checkable after the next oracle run, instead of guessed at |
| 8 | BERTScore silent failure | `src/evaluation/run_evaluation.py::compute_bertscore_batch()` | Exception + full traceback now written to `experiments/results/final_eval/bertscore_failure.log` and logged loudly, not silently swallowed |
| 9 | Duplicate sparsity implementations | `src/ehr/sparsity.py` | Deleted (dead, unused, inconsistent bucket casing); `src/lakehouse/sparsity.py` is the sole canonical implementation |
| 11 | Dead code / debug prints | `src/retrieval/unified_retrieval.py`, `src/ehr/ehr_snapshot.py`, `src/ehr/snapshot.py` (deleted — confirmed unused via grep); `src/retrieval/retriever.py` (debug `print()` → `logger.debug()`) | — |
| 12 (new) | n_meds always 0 at eval | `src/evaluation/run_evaluation.py` | Router features now sourced via `PatientSnapshot.get()`, matching training-time feature construction exactly |
| 13 (new) | Oracle/eval prompt format drift | `src/model/prompts.py` (new), `src/router/oracle_labels.py`, `src/evaluation/run_evaluation.py`, `src/model/test_medgemma.py` | Single canonical `build_user_message()` used everywhere a MedGemma prompt is built |

### Verification performed this session

- All edited files pass `python -m py_compile`.
- `mkg/edges/ontology_edges.csv` re-validated after edits: 215 rows, 25/25
  seed diseases present, zero exact-duplicate `(disease, edge_type, target)`
  rows, no CSV parsing errors.
- Confirmed via `grep` that no other module imports the deleted dead files
  before deleting them.

### What was intentionally NOT changed

- `train_router.py`'s core training procedure (already sound per the audit).
- The existing template question types (`primary_diagnosis`, `diagnoses`,
  `lab`, `medication`, `summary`, `next_step`) — kept as-is; their
  answer-copying confound is now a documented, known property rather than a
  silent one. A full redesign of their ground-truth generation was judged
  out of scope for this pass (risk of drifting into fabricating answers).
- `HallucinationDetector`'s actual keyword logic in `oracle_labels.py` — only
  instrumented, not changed, since the length-bias hypothesis (finding #7)
  was not yet empirically confirmed. Revisit after the next oracle run using
  the new `hallucination_diagnostic_per_mode` field.
- Hallucination/faithfulness evaluation in `run_evaluation.py` remains the
  crude heuristic described in finding #10 (not the human-annotated,
  Cohen's-κ taxonomy from the design doc). Out of scope for this pass —
  requires actual human annotation, which is a separate research task, not
  a code fix.

### Next steps — GPU-heavy work, to be executed by the project owner

Everything below depends on everything above it; run in order. Full command
sequence and rationale given directly to the project owner in-chat.

1. Regenerate synthetic QA (new `contraindication_check` type) + sparsity table
2. Rebuild MKG in Neo4j (expanded ontology edges) + regenerate co-occurrence edges
3. Rebuild FAISS text index (only needed if not already current)
4. Re-run QLoRA fine-tuning (now retrieval-augmented — this is the expensive step)
5. Rebuild router dataset, regenerate oracle labels, retrain router
6. Run the full 300-question held-out evaluation
7. Send results back for interpretation and next-round analysis
