# Research Log — Adaptive RAG Router for EHR-Grounded Clinical QA

This log records the research history of the project: investigations, hypotheses,
discovered bugs, experiments, code changes, rollbacks, and conclusions. It is
maintained continuously as work progresses so the full history is reconstructable
without needing to read commit-by-commit diffs.

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
