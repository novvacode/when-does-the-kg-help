# SESSION STATE — Handoff Document

**Last updated:** 2026-08-17 · **Repo:** https://github.com/novvacode/when-does-the-kg-help
**HEAD:** `f62fa97` · working tree clean, **pushed, in sync with `origin/main`**

> **Phase change:** the conference paper is complete. Work since 2026-08-13 is a
> **journal extension targeting the Journal of Biomedical Informatics (JBI)**:
> SHAP explainability, a human hallucination-annotation study, and a repaired
> EHR-contradiction detector. The conference-era items in §6 (compile, author
> block, references) are still open and unchanged.

> **Read this first, then [RESEARCH_LOG.md](RESEARCH_LOG.md) if you need the *why*.**
> This file is the operational state. RESEARCH_LOG.md is the full chronological
> record of every bug, audit, wrong hypothesis, and correction — long and
> deliberately unflattering. Several headline results in this project were
> overturned by later audits. **Do not trust any intermediate artifact without
> checking the log.**

---

## 1. Project goal

M.Tech research project. Question: **can a learned router that selects among three
retrieval configurations match always-on hybrid KG-RAG at lower cost, and when does
knowledge-graph augmentation actually help?**

- **H1** — router preserves quality vs. always-on hybrid while cutting latency and hallucination.
  → **2 of 3 pre-registered criteria met.** Quality parity ✅, latency −46.8% ✅, grounding ❌.
- **H2** — KG benefit grows as EHR sparsity increases.
  → **Contradicted.** Benefit *shrinks* with sparsity (+1.95pp low vs +0.57pp high), mechanism identified.

Deliverable is a paper (`paper/main.tex`), not a product.

---

## 2. Current architecture

```
MIMIC-IV CSVs
  └─ src/lakehouse/ingest.py ──────────► data/lakehouse/*.parquet (DuckDB views)
       ├─ patient_snapshot.py ─────────► PatientSnapshot API (demographics/dx/labs/vitals/meds)
       └─ sparsity.py ─────────────────► data/lakehouse/sparsity.parquet (S score, buckets)

src/qa/generate_synthetic_ehrqa.py ────► data/qa/ehrqa_{finetune,router_train,router_val,eval}.parquet
   · patient-level split (splits/patient_splits.json, seed 42)
   · FAMILY-AWARE DISEASE SPLIT: 13 finetune diseases / 12 held-out, overlap 0
   · 6 question types; 3 are KG-dependent

mkg/edges/*.csv ─► src/mkg/neo4j_loader.py ─► Neo4j (25 diseases, 215 guideline + 375 co-occurrence edges)
src/retrieval/embedder.py ─────────────► embeddings/notes_index.faiss (2,282,927 vectors, BGE-small, 384-d)

src/retrieval/retriever.py  ── the ONE retrieval path used by training AND inference
   Mode.T   = passages only
   Mode.TE  = passages + EHR snapshot
   Mode.TEK = passages + EHR + KG facts
   └─ src/model/context_budget.py: per-section budgets 600/350/350, MAX_SEQ_LENGTH=1524

src/model/train_qlora.py ──────────────► models/medgemma-4b-qlora/  (LOCAL ONLY, gitignored)
src/router/build_router_dataset.py ────► data/router/router_{train,val}_examples.parquet
src/router/oracle_labels.py ───────────► data/router/router_{train,val}_oracle.parquet
src/router/train_router.py ────────────► models/router/*
src/evaluation/run_evaluation.py ──────► experiments/results/final_eval/*
src/evaluation/{analysis,router_ablation,recompute_grounding}.py ► post-hoc stats
```

**Critical invariant:** training prompts are built by the *same* `Retriever` used at
inference, and every mode gets an identical budgeted passage block, so modes differ
**only** by additive EHR/KG content. Breaking this invalidates every comparison.

---

## 3. What has been completed

Full pipeline runs end-to-end and the paper is drafted.

| Stage | Status | Key output |
|---|---|---|
| Lakehouse + sparsity | done | 266 admissions scored, buckets low 140 / med 81 / high 44 |
| Synthetic QA | done | 1000 / 200 / 100 / 300 (finetune/rtrain/rval/eval) |
| MKG | done | 25 diseases, 215 guideline edges, 375 co-occurrence |
| FAISS index | done | 2,282,927 chunks |
| QLoRA fine-tune | done | best `checkpoint-210`, eval_loss **0.37875**, 4h14m |
| Router dataset | done | sparsity join 200/200 & 100/100, KG facts on 100% of KG questions |
| Oracle labels | done | train T+E 118 / T 44 / T+E+K 38; val 59/23/18 |
| Router training | done | val acc 0.9200, macro-F1 0.8692 |
| Held-out evaluation | done | 300 questions × 7 systems, 2100 rows, 0 empty answers |
| Post-hoc analysis | done | bootstrap CIs, ablation, latency split, grounding |
| Paper | drafted | `paper/main.tex`, IEEEtran, 18 refs |
| GitHub release | done | repo hardened, patient data excluded |

### Journal extension (2026-08-13 → 2026-08-17)

| Stage | Status | Key output |
|---|---|---|
| SHAP explainability | done | `src/evaluation/router_shap.py`; both tiers gated (probs match 2.97e-08; modes 300/300) |
| Human annotation study | done | 75 rows × 7 systems; `unsupported_rate` κ=0.7486, `ehr_contradiction` κ=−0.0370 |
| EHR-contradiction fix | done, validated | `src/evaluation/fix_ehr_contradiction.py`; FP 20→0 on unseen rows, McNemar p=2.0e-06 |
| Full recompute | done | 2100 rows, 186→46 flags; **two reporting caveats, see §9** |
| Round-3 validation | done — **criterion FAILED** | `monitoring_labs` precision 0.7333 [0.4490, 0.9221], bar was ≥0.80. See §9. |
| Step 4: KG-contradiction | done — **VALIDATED** | contraindication-violation detector, precision 1.0000 [0.9140, 1.0000]. KG cuts violations 7→0 vs T+E (**Holm p=0.0469**, the only comparison surviving correction); **not** better than plain T. See §9. |
| SHAP paper section | done | `paper_journal/sections/shap_explainability.tex` + 3-panel figure; beeswarms rejected as unusable |
| elsarticle conversion | done, **UNCOMPILED** | IEEEtran -> elsarticle (preprint,12pt), JBI numerical style. Structural checks pass; **compile before layering more on** |
| Step 1(a): conference fix | done | invalidated EHR-contradiction metric withdrawn from the frozen `paper/`; submittable without a known-bad claim |
| Step 5a: seed variance | done | acc 0.9240 ± 0.0143 over 10 seeds; but the 100-row val CI [0.86, 0.97] is 1.9x wider. See §9. |
| Step 5b: second base model | **DEFERRED** | Phi-3-mini replication: 10+ GPU h against a documented crash history. Stated limitation, not attempted. |
| `requirements.txt` | added | never existed despite README referencing it |

### Headline results (all verified against artifacts)

| System | BLEU | ROUGE-L | BERTScore | Latency (ms) |
|---|---|---|---|---|
| T | 0.1588 | 0.2676 | 0.7937 | 2209.6 |
| T+E | 0.6859 | 0.7965 | 0.9553 | 3341.3 |
| T+E+K | 0.7481 | 0.8630 | 0.9708 | 7527.7 |
| **Router** | **0.7488** | **0.8632** | 0.9685 | **4004.9** |
| Random | 0.5131 | 0.6272 | 0.9002 | 4297.6 |
| StaticQType | 0.7338 | 0.8392 | 0.9634 | 3790.0 |
| Oracle | 0.7497 | 0.8727 | 0.9708 | 3961.7 |

**Three findings** (one positive, two negative — report all three):
1. Router ≈ T+E+K on quality (BLEU p=0.304, ROUGE p=0.391, BERTScore p=0.526), −46.8% latency.
2. **Not patient-adaptive.** Ablation: question-only router = 0.9200 acc / 0.8873 macro-F1;
   full = 0.9200 / 0.8692. Patient features contribute **−0.0181 macro-F1**.
3. **H2 inverted.** KG benefit +1.95pp (low sparsity) vs +0.57pp (high). KG retrieval is keyed
   on the diagnosis list, which sparse records lack.

---

## 4. What was changed in this session

1. **Post-hoc analysis added** — `src/evaluation/analysis.py`, `router_ablation.py`,
   `recompute_grounding.py`. The ablation **withdrew the earlier patient-adaptive claim**;
   the length-controlled grounding measure **confirmed** the H1 grounding failure is real,
   not a metric artifact (my earlier "artifact" hypothesis was wrong).
2. **Paper written** — `paper/main.tex`, IEEEtran conference format, 9 sections, 7 tables,
   3 figures, 18 references. All `[CITATION NEEDED]` replaced with web-verified refs.
3. **Table overflow fixed** — `\footnotesize` + `\tabcolsep=4pt` on single-column tables.
   ⚠ While doing this I corrupted the file via a bash heredoc (`\\f`→formfeed, `\\t`→tab,
   15 control chars). **Repaired.** Lesson: write edit scripts to a file, never heredoc LaTeX.
4. **Repo hardened for public release** —
   - `.gitignore` rewritten; `data/qa/` was **NOT** ignored and contained real MIMIC-IV
     patient data (admission IDs, diagnoses, lab values, meds, demographics).
   - Neo4j password moved from hardcoded literal → env vars.
   - README rewritten: corrected findings, data policy, limitations.
   - Paper Ethics corrected to state precisely which artifacts are released.
5. **Repo renamed** `med-rag-router` → `when-does-the-kg-help`; description and topics updated;
   README router-behaviour table corrected (it contradicted the H2 result).

---

## 5. Current implementation state

**Working tree clean, everything pushed.** Local-only artifacts (gitignored, **do not delete**):

| Path | Size | Notes |
|---|---|---|
| `embeddings/` | 4.9 G | FAISS index — expensive to rebuild (~1–2 h) |
| `models/medgemma-4b-qlora/` | 671 M | Fine-tuned adapter, best ckpt-210 |
| `models/router/` | 130 M | Trained router + feature pipeline |
| `data/qa/`, `data/router/`, `data/lakehouse/` | ~19 G | **Patient data — never commit** |
| `experiments/results/final_eval/` | 2.6 M | All result CSVs |
| `models/_ARCHIVE_*` (×3) | — | Superseded runs, kept for comparison |

Environment: conda env `ehr-rag` at `C:\Users\jangi\anaconda3\envs\ehr-rag\python.exe`
(torch 2.5.1+cu121, transformers 5.12.1, trl 1.6.0, peft 0.19.1). The system Python has
neither duckdb nor pyarrow — **always use the ehr-rag interpreter.**

---

## 6. What remains to be done

**Blocking submission:**
1. **Fill the author block** — 5 placeholders in `paper/main.tex` (`[AUTHOR NAME]`,
   `[AFFILIATION]`, `[UNIVERSITY]`, `[CITY, COUNTRY]`, `[EMAIL]`).
2. **Compile the PDF — never done.** No LaTeX toolchain on this machine (checked:
   pdflatex/xelatex/lualatex/latexmk/tectonic all absent). Use Overleaf (upload
   `paper/main.tex` + `paper/plots/`) or install MiKTeX. Then check the log for
   `Overfull \hbox`. Table widths were estimated statically, **not verified by TeX**.
3. **Verify the 9 unverified references.** Web-verified this session: MedGemma, MIMIC-IV,
   FAISS, C-Pack, DuckDB, MedRAG, GraphRAG, Adaptive-RAG, Thirunavukarasu. **Not re-checked:**
   RAG (Lewis), SBERT, Self-RAG, QLoRA, LoRA, XGBoost, BLEU, ROUGE, BERTScore.

**Security:**
4. **Change the Neo4j password — DEFERRED by the project owner (2026-08-17).**
   `medrag123` is in public git history from commit `9d83e1a` and was re-exposed in a
   working session. The owner reviewed this and chose not to rotate for now; pushing
   adds no new exposure, since the string was already published in this file. It is a
   local-only development database. Revisit before any deployment or credential reuse.

**Journal extension (JBI):**
5. **Decide how to report the split validation.** Round 3 failed its bar
   (precision 0.7333 vs ≥0.80). Options, in order of preference:
   (a) report the qualitative finding — the original metric fails in both directions,
   contradiction concentrates in `monitoring_labs` — which is well supported and does
   not depend on the failed bar; (b) report the rate per question type with round 2 /
   round 3 validation status attached to each; (c) fix the `no abnormal` artifact and
   run a **fourth** disjoint validation round. Do **not** report one pooled number.
6. **Replace the paper's EHR-contradiction column** with the paired
   negation-contradiction numbers (§9), under the new name, with Type 2 scoped out and
   the per-type validation status stated.
7. **Swap the scorer into `run_evaluation.py`.** It still holds the ORIGINAL function on
   purpose, so the conference paper's numbers stay reproducible from committed code.
   Do this only after item 6 is settled.
8. **Compile `paper_journal/main.tex`.** Never compiled. Structural checks pass (braces
   382/382, 29/29 refs, 18/18 cite↔bibitem, tables column-consistent, 0 control chars)
   but those do not catch overfull boxes or float placement. Owner is compiling on
   Overleaf.
9. **Trim the abstract.** Now ~382 words against a typical 250 cap. Deliberately
   deferred to the Step-6 journal reformatting, so it is not trimmed twice.
10. **Draft the SHAP section** for `paper_journal/sections/`; results already exist in
    `experiments/results/final_eval/shap_*`.

**Optional (venue-dependent):**
5. Abstract is ~268 words; some IEEE venues cap at 250.

---

## 7. Current bugs / issues

**None known in the code.** All fixed issues are logged in RESEARCH_LOG.md.

**Open risks:**

| Issue | Status |
|---|---|
| **GPU instability** | 3 crashes: illegal-instruction + BSOD (`nvlddmkm.sys`), 2× illegal-memory-access. Root cause **never confirmed**. Mitigated: paged optimizer removed, checkpointing everywhere, `UnrecoverableCudaError` → save-and-exit. A CUDA illegal-memory-access **poisons the process** — never retry in-process, restart. |
| `expandable_segments` | Unsupported on this Windows build; the fragmentation mitigation is inert. |
| Paper never compiled | Overfull boxes / page breaks / figure clipping all unverified. |
| Abstract length | 268 words vs 250 cap at some venues. |

**Scientific limitations (stated in paper §Limitations, not bugs):**
template-generated questions · 300 eval questions / 61 admissions / 1 seed / 1 model ·
25-disease hand-curated KG · heuristic hallucination metrics, no human evaluation ·
EHR-contradiction **undefined** for mode T (reported `n/a`, never 0%).

---

## 8. Tests / checks already run

| Check | Result |
|---|---|
| Patient/admission leakage, all 6 split pairs | hadm_id 0, subject_id 0 |
| Disease-family leakage | 7/7 families intact, finetune ∩ eval = 0 |
| Loss masking on real data | 2.97% supervised tokens (was 100% pre-fix) |
| Sequence overflow | 0/200; max 1369 < 1524 |
| Context-echo rate (post-retrain) | **0/54 = 0.0%** (broken model was 52–63%) |
| KG memorisation | T dropped 62.5% → 16.7% after disease split |
| KG facts on KG questions | 75.7% → **100%** after retrieval fix |
| Router vs baselines (same rows) | Router 0.92 vs qTEXT 0.82, qTYPE 0.82, majority 0.59, random 0.33 |
| Router head-to-head vs lookup | 16 disagreements: router 11 correct, lookup 1 |
| Cost is not KG-avoidance | Router predicts T+E+K 20× vs 18 actual |
| Paper numbers vs artifacts | **55/55 match, 0 mismatches** |
| Stale-number sweep | 0 hits across 12 superseded values |
| Demo-paper leakage | 0 hits across 20 distinctive terms |
| cite ↔ bibitem | 18/18, 0 orphans, 0 duplicates |
| LaTeX control chars | 0 (after repair) |
| Pre-push patient-data scan | 0 data files staged; remote verified clean |

---

## 9. Important decisions and constraints

**Non-negotiable:**
- **`paper/` IS FROZEN (from 2026-08-17).** `paper/main.tex` is the conference paper
  and may be submitted as-is. **Never edit anything under `paper/`** — not the
  withdrawn EHR-contradiction column, not formatting, not the author block. All
  journal-extension writing goes in **`paper_journal/`**, which started as a
  byte-identical copy. This overrides any earlier instruction to fix numbers "in the
  paper".
- **Never commit patient data.** MIMIC-IV is under a PhysioNet DUA prohibiting
  redistribution. `.gitignore` is the guard — verify before any `git add -A`.
- **Never regenerate `splits/patient_splits.json`.** It is the locked seed-42 partition.
- **Never force-push.** The remote has commits made outside this workflow.
- Held-out `data/qa/ehrqa_eval.parquet` is for final evaluation only — never for tuning.

**Design decisions (with rationale):**
- `ehr_covers_answer` / `ehr_lab_coverage` are **deliberately excluded** from router
  features — they are derived from the KG panel that *is* the gold answer, so using them
  would leak. They are H2 analysis variables only.
- Oracle tie-breaking prefers the **cheapest** mode within `TIE_EPSILON`, measured on
  **raw** composites (not cost-adjusted — that bug made the metric useless).
- Disease families split together (all CKD stages + AKI on one side) because they share
  contraindications.
- The router is described as **question-driven**, never "patient-adaptive" — the ablation
  disproved that. Do not let this framing drift back.
- Mode T's EHR-contradiction is **`n/a`**, never 0.0000.

**Standing corrections added by the journal extension — do not let these drift:**

- **Attribution ≠ performance contribution.** Table VI (ablation) says patient features
  add no *accuracy*; SHAP says they carry ~9% of attribution mass and `n_labs` ranks
  **#5 of 389**, nonzero on 300/300 rows. Both are true. Correct phrasing:
  **"question-driven, with performance-neutral use of patient state."** Wrong:
  "the router ignores patient state." See RESEARCH_LOG 2026-08-13.
- **The metric is `negation-contradiction rate`, not "EHR-contradiction".** It measures
  Type 1 only (answer negates what the EHR asserts). Type 2 (answer asserts what the
  record refutes) needs clinical reasoning and is **out of scope** — the one
  human-flagged miss, row A065, was Type 2. Do not let the repaired metric reclaim the
  broader name.
- **The n/a rule creates a SELECTION EFFECT.** Any system that chooses when to enter
  mode T has those rows dropped, and ~90% of contradictions live exactly there
  (T+E: 16/18 flags on the 74 Router-routed-to-T questions, Fisher p=4.9e-09). The
  unpaired per-system table shows a spurious 10× Router win; on matched questions
  **Router and T+E+K are tied at 0.0044**. Per-system rates on a mode-conditional
  metric MUST use a matched question set.
- **The `unsupported ≥ 0.5` threshold is pre-registered and locked** (2026-08-13, before
  annotation). κ peaks at 0.4 in the sweep; **do not retune** — that fits the metric to
  the reference standard judging it. The sweep is robustness only.
- **Never validate a fix on the rows that exposed the bug.** Rounds 2 and 3 draw
  samples disjoint from all prior annotation, enforced on `q_idx|system`.
- **Annotation samples carry attention checks.** When a working fix implies a uniform
  "no" response set, constructed contradictions are the only thing separating a real
  result from an inattentive one. They are excluded from all statistics.
- **The corrected scorer is validated for DIAGNOSIS questions only.** Round 2
  (`diagnoses`/`primary_diagnosis`): FP 20 → 0, McNemar p=2.0e-06 ✅. Round 3
  (`monitoring_labs`/`lab`): precision **0.7333** [0.4490, 0.9221] against a
  pre-registered ≥0.80 bar — **FAILED by one row, and the bar was not moved**. Report
  the split; never a single validated number across all question types.
- **The original metric failed in BOTH directions.** It over-flagged on diagnosis
  questions (κ=−0.037, 0 TP) *and* under-detected on `monitoring_labs` (recall 0.25 —
  it missed 9 of 12 genuine contradictions). **The true contradiction rate is HIGHER
  than the paper reports, not lower.** Contradiction concentrates in `monitoring_labs`
  (12 genuine in 20 rows) versus `diagnoses` (0 in 30).
- **The remaining artifact is known but must NOT be patched from the rows that found
  it.** All 4 round-3 false positives are one template answer, *"No abnormal labs
  available"*, triggering on `no abnormal` ("abnormal" is a generic qualifier, not a
  finding). A fix is easy and would be circular to validate on those rows — it needs a
  **fourth** fresh sample, disjoint from all three.
- **The KG's safety benefit is CORRECTIVE, not additive — and only ONE of the three
  comparisons survives multiple-comparisons correction.** Violations: T 12 → T+E **18**
  → T+E+K 11. **Holm-Bonferroni across the 3 tests (2026-08-17):**
  T+E vs T+E+K raw 0.0156 → **Holm 0.0469, SURVIVES**;
  T vs T+E raw 0.0312 → **Holm 0.0625, DOES NOT SURVIVE**;
  T vs T+E+K 1.0 → 1.0.
  So "the EHR snapshot causes the harm" is **consistent with but not independently
  significant** — never state it as confirmed. The corrective-not-additive reading is
  safe: it rests on the surviving comparison plus the T-vs-T+E+K null, not on the
  downgraded one. And never write "KG injection reduces contraindication violations"
  without "relative to T+E" — unqualified it overclaims.
- **This measures agreement with the KG's assertions, not clinical safety**, and covers
  **one relation type of five** (`CONTRAINDICATED_WITH`). The other four are
  structurally uncheckable — three are non-exhaustive `LIMIT 3` lists, and
  `CO_OCCURS_WITH_LAB` is a frequency statistic no clinical claim can contradict.
- **Seed variance is the SMALLER uncertainty on the router.** Across 10 seeds accuracy is
  0.9240 ± 0.0143 (range 0.9000-0.9500) and the deployed 0.9200 sits *below* that mean, so
  it is not cherry-picked. But bootstrapping the 100 validation rows gives
  **[0.8600, 0.9700]** — a CI 1.9x wider than the seed span. Never report seed stability
  without the evaluation CI; alone it understates how imprecisely 0.9200 is known.
  T+E+K F1 swings 0.7568-0.8571 because that class has 18 validation rows.
  Scope: this varies **router training only** — the QLoRA fine-tune, oracle labels and the
  train/val partition are fixed. It is not end-to-end pipeline variance. Also
  `subsample=1.0` disables row subsampling, so a modest spread is partly a property of the
  deployed configuration.
- **The KG is incomplete in two distinct ways, both worth reporting on their own.**
  (a) K026: a drug with a well-known renal risk carries no caution edge for a patient
  with impaired renal function — adjudicated 2026-08-17 as a **KG coverage gap, not a
  detector miss**; headline recall stays 1.0000. (b) The linearisation flattens
  conditional prohibitions. `get_subgraph_facts()`
  renders "contraindicated if eGFR<30" as an unconditional "contraindicated in T2DM",
  which is literally wrong — metformin is first-line for T2DM unless renal function is
  poor. A defect worth reporting on its own, independent of any detector.
- **Attention checks must be independently constructed and internally coherent.** The
  first KG checks failed 0/3 because all three were one repeated answer string on an
  incoherent question — a construction bug, not inattention. Rebuilt properly: 3/3.
  A malformed check does not measure attention, and voiding good work on the strength
  of one would have been the wrong call. Verify what a check *contains*.

**Workflow constraints:**
- GPU-heavy steps are run by the user, not the agent.
- Before any long GPU run: **archive `models/medgemma-4b-qlora/`** or `get_last_checkpoint()`
  will silently resume the old model. This nearly cost 13 h once.
- Use `C:\Users\jangi\anaconda3\envs\ehr-rag\python.exe`, not system Python.
- Write edit scripts to files; **do not pipe LaTeX/regex through bash heredocs.**
- **Neo4j steps need `NEO4J_PASSWORD` set, in SINGLE quotes in PowerShell**
  (`$env:NEO4J_PASSWORD='...'`) — double quotes expand `$` and silently corrupt it.
  A failed auth does NOT stop retrieval: `Retriever` swallows KG errors and returns
  contexts with no KG facts, which once produced a corrupt annotation sample that
  reported "0 failures". Scripts now run a real Cypher preflight; keep it that way.
- **`run_evaluation.py` deliberately still contains the ORIGINAL (broken) contradiction
  scorer**, so the conference paper's numbers stay reproducible from committed code.
  The corrected one lives in `src/evaluation/fix_ehr_contradiction.py`. Do not "tidy"
  this without doing §6 item 6 first.

---

## 10. Exact next step

**Journal extension (current work):**

1. **Finish round 3.** Build (`build_validation_sample_v3.py`, needs Neo4j), annotate
   `experiments/results/annotation_v3/annotate_v3.html` (23 rows, ~10 min), then score
   it. Pre-registered: validated if ≥ 12/15 stratum-A rows are confirmed genuine.
2. **Then** replace the paper's contradiction column with the paired numbers under the
   name `negation-contradiction rate`, Type 2 out of scope.
3. Rotate the Neo4j password, then push the 4 unpushed commits.

**Conference-paper mechanics (unchanged, still open):**

**Step 1 — compile the paper (still never done; the only unverified item):**

Upload `paper/main.tex` and the `paper/plots/` folder to Overleaf, set compiler to
pdfLaTeX, and compile. Then:
- search the log for `Overfull \hbox` — table widths were estimated, not TeX-verified;
- confirm the 3 figures render and are not clipped;
- confirm all 18 references appear and no `[?]` citation markers remain.

If tables still overflow, the next lever is `\resizebox{\columnwidth}{!}{...}` around the
offending `tabular`, or promoting it to `table*`.

**Step 2 — fill the author block** in `paper/main.tex` (5 placeholders, lines ~17–21).

**Step 3 — verify the 9 unchecked references** (listed in §6.3).

**Step 4 — change the Neo4j password** (in public git history, and re-exposed 2026-08-16).

Nothing requires re-running the GPU pipeline. If a fresh run ever becomes necessary,
the command order is in README "Running the Pipeline" — and archive the existing
adapter first.
