<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/MedGemma-1.5--4B-4285F4?style=flat-square&logo=google&logoColor=white"/>
<img src="https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white"/>
<img src="https://img.shields.io/badge/Neo4j-5.x-008CC1?style=flat-square&logo=neo4j&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Research-orange?style=flat-square"/>

<br/><br/>

# When Does the Knowledge Graph Help?

### Question-Driven Retrieval Routing for EHR-Grounded Clinical Question Answering

*M.Tech Research Project · MAHE Bengaluru*

**Router matches always-on hybrid KG-RAG quality (all *p* > 0.05) at 46.8% lower latency —
but patient state adds no routing signal, and KG benefit *decreases* with EHR sparsity.**

</div>

---

## The Problem

Most medical RAG systems apply the same expensive retrieval pipeline to every question — text search, EHR lookup, and Knowledge Graph traversal, every single time. This is wasteful and sometimes harmful: when a patient's EHR already contains all the relevant facts, extra KG context adds noise, not signal.

This project asks a sharper question: **can a learned router that decides *when* to use each retrieval source reduce both hallucinations and latency, compared to always-on hybrid KG-RAG?**

We evaluated this on 300 held-out MIMIC-IV questions across seven systems. The answer is
partly yes and partly no, and both halves are reported here: the router preserves quality
at substantially lower cost, but it routes on the *question*, not the patient, and it gives
up measurable grounding to do so. Two of our three pre-registered H1 criteria are met; the
third is not. H2 is contradicted outright, with an identified mechanism.

---

## What This System Does

Given a clinical question about a specific patient, the system picks among three retrieval strategies in real time:

```
Clinical Question + Patient ID
           │
           ▼
  ┌─────────────────┐
  │ Adaptive Router │  ← XGBoost classifier trained on oracle labels
  │  (XGBoost)      │    Features: EHR sparsity, KG coverage,
  └────────┬────────┘    question type, retrieval scores
           │
     ┌─────┴──────┐
     │            │
 ┌───▼───┐  ┌────▼────┐  ┌──────▼──────┐
 │   T   │  │  T + E  │  │  T + E + K  │
 │ Text  │  │ Text +  │  │ Text + EHR  │
 │ Only  │  │   EHR   │  │  + KG Graph │
 └───────┘  └─────────┘  └─────────────┘
           │
           ▼
   MedGemma 1.5-4B (QLoRA fine-tuned)
           │
           ▼
   Grounded Clinical Answer
```

| Mode | When the Router Chooses It |
|---|---|
| **T** — Text only | Definitional questions; dense, note-rich patients |
| **T+E** — Text + EHR | Patient-specific questions with rich structured data |
| **T+E+K** — Text + EHR + KG | Sparse EHR; complex disease–drug interactions; gap-filling |

---

## Research Questions

**H1 (Main):** Can an adaptive retrieval router reduce clinically relevant hallucinations and latency compared to always-on hybrid KG-RAG, while preserving or improving answer quality?

**H2 (Sub-question):** Under what EHR sparsity conditions does KG augmentation help vs. hurt accuracy and hallucination rates?

---

## Key Results

Held-out evaluation, **300 questions × 7 systems**, patient-disjoint from all
training and routing data. Full analysis in [RESEARCH_LOG.md](RESEARCH_LOG.md).

| System | BLEU | ROUGE-L | BERTScore-F1 | EHR-Contra. | Unsupported | Latency (ms) | KG facts |
|---|---|---|---|---|---|---|---|
| T | 0.1588 | 0.2676 | 0.7937 | n/a¹ | 0.5813 | 2210 | 0.00 |
| T+E | 0.6859 | 0.7965 | 0.9553 | 0.0247 | 0.3244 | 3341 | 0.00 |
| T+E+K *(always-on hybrid)* | 0.7481 | 0.8630 | **0.9708** | 0.0247 | **0.1832** | 7528 | 8.88 |
| **Router** *(ours)* | **0.7488** | **0.8632** | 0.9685 | **0.0233** | 0.2554 | **4005** | 2.04 |
| Random | 0.5131 | 0.6272 | 0.9002 | 0.0193 | 0.3493 | 4298 | 3.27 |
| StaticQType *(lookup baseline)* | 0.7338 | 0.8392 | 0.9634 | 0.0240 | 0.2696 | 3790 | 1.34 |
| Oracle *(upper bound)* | 0.7497 | 0.8727 | 0.9708 | 0.0233 | 0.2552 | 3962 | 1.66 |

¹ EHR-contradiction is **not measurable** for mode T: the detector scans the
EHR snapshot, which mode T does not receive. It is not 0% — it is undefined.

### H1 — partially supported (2 of 3 pre-registered criteria)

- ✅ **Quality parity with always-on hybrid.** Router vs T+E+K is
  statistically indistinguishable on all three quality metrics
  (BERTScore p=0.53, BLEU p=0.30, ROUGE-L p=0.39; paired Wilcoxon, n=300).
- ✅ **46.8% lower latency** (4005 ms vs 7528 ms), by invoking the KG on only
  **17.3%** of questions.
- ❌ **Grounding is worse** than T+E+K. The raw unsupported-rate metric is
  confounded with context length (pooled r = −0.52), so we also report a
  length-controlled measure (`grounding_excess`: answer scored against its
  real context vs a length-matched decoy). **Both agree**: raw 0.2554 vs
  0.1832 (p=2.2e-10); length-controlled 0.5379 vs 0.6271 (p=1.2e-05). The
  router genuinely trades grounding for latency by routing ~25% of
  questions to a weaker mode. This is a real failure, not a metric artifact.

The router also **beats the static question-type lookup** on every quality
metric and sits essentially **at the Oracle upper bound** on BLEU/ROUGE-L.

**Mode T is essentially ungrounded.** Its `grounding_excess` is **0.0152** —
a T-mode answer is barely better explained by its own retrieved passages
than by an unrelated context of the same length.

### What drives the routing (feature ablation)

| router variant | features | accuracy | macro-F1 |
|---|---|---|---|
| question embeddings only | 384 | 0.9200 | **0.8873** |
| patient features only | 5 | 0.3600 | 0.3385 |
| both (deployed) | 389 | 0.9200 | 0.8692 |

Patient features contribute **nothing** (accuracy +0.0000, macro-F1 −0.0181).
The router is a **learned question-routing policy**, not a patient-adaptive
one: on this benchmark the optimal retrieval mode is predictable from the
question, and patient state adds no measurable routing signal. It still
beats an exact-match question lookup (0.92 vs 0.82) because embeddings
generalise to unseen question phrasings, where the lookup scores 0.00.

### Where the latency saving comes from

| system | retrieval | generation | total |
|---|---|---|---|
| T+E+K | **4343 ms** | 3185 ms | 7528 ms |
| Router | **822 ms** | 3183 ms | 4005 ms |

Generation cost is identical across modes; the router's entire saving comes
from skipping the Neo4j round-trip, which is 57.7% of T+E+K's total latency.

### H2 — contradicted, with an identified mechanism

KG benefit was predicted to grow with EHR sparsity. It does the opposite:
T+E+K's BERTScore advantage over T+E is **+1.95pp in low-sparsity** and only
**+0.57pp in high-sparsity**. Cause: KG retrieval is keyed on the patient's
diagnosis list, so sparse EHRs supply fewer diagnoses to match and therefore
*less* KG signal. The KG cannot fill a gap that includes its own index key.

### Interpreting the absolute numbers

BLEU/ROUGE/BERTScore for T+E, T+E+K and Router sit **above** the
pre-registered target ranges. This is the extractive-answer confound, not
leakage: ~61% of eval questions are template questions whose gold answer is
a verbatim EHR field present in the T+E/T+E+K context. Mode T, which has no
EHR snapshot, scores BLEU 0.1588 — inside the target range — which is what
you would expect under this explanation and not under genuine leakage.

---

## System Components

### Healthcare Lakehouse (Phase 1)
- MIMIC-IV tables converted to Parquet, queried via DuckDB
- `PatientSnapshot` API: given any `hadm_id`, returns structured labs, vitals, diagnoses, medications
- EHR sparsity score: `S = α₁·𝟙(n_labs < τ) + α₂·𝟙(n_diag < τ) + α₃·𝟙(d_note > τ)` → buckets `{low, medium, high}`

### Medical Knowledge Graph (Phase 2)
- Neo4j graph covering 25 chronic/common internal-medicine conditions (T2DM,
  hypertension, CKD stages 3–5, heart failure, COPD, sepsis, AKI, and others —
  see `src/mkg/seed_diseases.py`)
- Edge types: `HAS_SYMPTOM`, `INDICATES_LAB`, `FIRST_LINE_TREATMENT`,
  `CONTRAINDICATED_WITH`, `CO_OCCURS_WITH_LAB`
- **Scope note:** the ontology edges (`mkg/edges/ontology_edges.csv`) are
  hand-curated from standard guideline knowledge (ADA 2023, ACC/AHA 2023,
  KDIGO 2022, GOLD 2023 style facts), not extracted from a licensed UMLS/
  SNOMED-CT API — an earlier design goal that was not implemented. Current
  scale after the 2026-08-06 expansion (RESEARCH_LOG.md finding #6): ~135
  nodes, ~215 ontology edges (up from ~114/135 before), plus co-occurrence
  edges computed directly from MIMIC-IV (threshold ≥ 5% of admissions per
  disease). This is materially smaller than the original 1,000–3,000 node /
  5,000–10,000 edge target, which assumed full ontology-API extraction —
  report this honestly as a scope limitation rather than the original target.
- 50-edge manual validation against clinical references
  (`mkg/validation/edge_validation_50.csv`)

### FAISS Retrieval (Phase 3)
- `BAAI/bge-small-en-v1.5` embeddings (384-dim, cached)
- MIMIC-IV discharge notes chunked at 256 tokens (32-token overlap)
- `IndexFlatIP` cosine similarity search, top-k = 5

### MedGemma Fine-Tuning (Phase 4)
- Base: `google/medgemma-1.5-4b-it`
- Fine-tuning: QLoRA (r=16, α=32, NF4, target: all linear layers)
- Training data: synthetic EHR-QA from the MIMIC-IV fine-tune patient split
  (`data/qa/ehrqa_finetune.parquet`)
- **Training prompts are retrieval-augmented**: each example's context is
  built by calling the real `Retriever` (same code path as inference) with a
  randomly, reproducibly assigned mode (T / T+E / T+E+K, equal weight by
  default — see `CONTEXT_MODE_WEIGHTS` in `src/model/train_qlora.py`), so the
  model is trained on the exact prompt structures it is evaluated on. This
  replaced an earlier version that only ever trained on a flattened EHR
  snapshot with no retrieved passages or KG facts (RESEARCH_LOG.md finding #4).
- Hardware: RTX 4050 Laptop GPU (6 GB VRAM), 2 epochs, ~4–6 hours

### Adaptive Router (Phase 5)
- **Oracle label generation:** run all 3 modes on 200 router-train questions → score with composite metric (60% BERTScore + 25% ROUGE-L + 15% EM − hallucination penalty) → pick best mode as label
- **Features:** BGE-small question embeddings + EHR sparsity features + KG coverage + retrieval scores + question-type one-hot + surface meta-features (~395-dim total)
- **Classifier:** XGBoost with class-weight balancing, early stopping, optional RandomizedSearchCV tuning
- **Baselines:** Random routing · Majority class · Always-T+E+K · Oracle (upper bound)

### Evaluation (Phase 6)
- 300–500 held-out EHR-QA pairs (never used in training or routing)
- Hallucination taxonomy: EHR-contradicting · KG-contradicting · Unsupported
- Inter-annotator agreement: Cohen's κ (target ≥ 0.6)
- Clinical face-validity check: medically trained reviewer rates 50 answers
- H2 analysis: all metrics broken down by sparsity bucket × system

---

## Project Structure

```
med-rag-router/
│
├── src/
│   ├── lakehouse/
│   │   ├── ingest.py               # CSV → Parquet conversion
│   │   ├── query.py                # DuckDB query helpers
│   │   ├── make_splits.py          # Leakage-safe patient ID splits (seed=42)
│   │   ├── patient_snapshot.py     # PatientSnapshot API (canonical EHR snapshot)
│   │   └── sparsity.py             # EHR sparsity score computation (canonical)
│   │
│   ├── mkg/
│   │   ├── seed_diseases.py        # Seed disease list + node/edge schema
│   │   ├── cooccurrence.py         # EHR co-occurrence edges from MIMIC-IV
│   │   ├── neo4j_loader.py         # Loads mkg/edges/*.csv into Neo4j
│   │   ├── retrieval.py            # Subgraph retrieval + linearization
│   │   └── sample_validation.py    # Manual edge validation sampling
│   │
│   ├── retrieval/
│   │   ├── embedder.py             # FAISS index build over note chunks
│   │   └── retriever.py            # Unified T / T+E / T+E+K retriever
│   │
│   ├── router/
│   │   ├── build_router_dataset.py # Runs all 3 modes over router splits
│   │   ├── oracle_labels.py        # Generate router training labels
│   │   ├── feature_pipeline.py     # HybridFeaturePipeline
│   │   ├── train_router.py         # XGBoost router training
│   │   └── verify_oracle.py        # Oracle label sanity checks
│   │
│   ├── model/
│   │   ├── train_qlora.py          # QLoRA fine-tuning (retrieval-augmented)
│   │   ├── prompts.py              # Single source of truth for the MedGemma
│   │   │                           # user-message format, shared by training,
│   │   │                           # oracle generation, and final evaluation
│   │   └── test_medgemma.py        # Manual single-example smoke test
│   │
│   ├── qa/
│   │   └── generate_synthetic_ehrqa.py  # Split-safe synthetic QA generation
│   │
│   └── evaluation/
│       └── run_evaluation.py       # Final held-out evaluation
│
├── data/
│   ├── raw/                        # MIMIC-IV CSVs (not in repo)
│   ├── lakehouse/                  # Processed Parquet files (not in repo)
│   │   └── qa/                     # EHR-QA datasets
│   └── router/                     # Router datasets + oracle labels
│       ├── router_train_examples.parquet
│       ├── router_val_examples.parquet
│       ├── router_train_oracle.parquet
│       └── router_val_oracle.parquet
│
├── models/
│   ├── medgemma-4b-qlora/          # Fine-tuned adapter weights (not in repo)
│   └── router/                     # Trained XGBoost router + artifacts
│       ├── router_xgb_model.json
│       ├── label_encoder.pkl
│       ├── feature_pipeline.pkl
│       └── router_metadata.json
│
├── splits/
│   └── patient_splits.json         # Locked patient ID splits (seed=42)
│
├── mkg/
│   ├── edges/                      # Ontology (hand-curated) + co-occurrence
│   │   │                           # (computed from MIMIC-IV) edge CSVs —
│   │   │                           # see "Medical Knowledge Graph" below for
│   │   │                           # actual scope vs. original design target
│   │   ├── ontology_edges.csv
│   │   └── cooccurrence_edges.csv
│   └── validation/                 # 50-edge manual validation table
│
├── experiments/
│   ├── results/
│   │   └── final_eval/             # All evaluation outputs + figures
│   └── logs/
│
├── notebooks/                      # Exploration notebooks
├── RESEARCH_LOG.md                 # Full research history — bugs, fixes,
│                                    # experiments, conclusions
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Installation

**Prerequisites:** Miniconda, NVIDIA GPU with CUDA 12.1+, Neo4j Desktop

```bash
git clone https://github.com/novvacode/med-rag-router.git
cd med-rag-router
```

```bash
conda create -n ehr-rag python=3.11 -y
conda activate ehr-rag
```

```bash
pip install torch==2.5.1+cu121 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

**Neo4j:** Download [Neo4j Desktop](https://neo4j.com/download/), create a local DBMS named `mkg`, and start it before running any MKG steps.

Connection settings are read from the environment — no credentials are stored in
this repository. Set them before running any MKG step:

```bash
export NEO4J_URI="bolt://localhost:7687"   # PowerShell: $env:NEO4J_URI="..."
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
```

---

## Running the Pipeline

> Each step depends on the previous. Run in order. As of the 2026-08-06 audit
> fix (see [RESEARCH_LOG.md](RESEARCH_LOG.md)), QLoRA fine-tuning now builds
> retrieval-augmented prompts, so the FAISS index and MKG must exist
> **before** fine-tuning — this reorders steps 4/5 relative to earlier
> versions of this README.

**Step 1 — Ingest MIMIC-IV into the lakehouse**
```bash
python src/lakehouse/ingest.py
```

**Step 2 — Lock patient ID splits** *(run once, never again)*
```bash
python -m src.lakehouse.make_splits
```

**Step 3 — Generate split-safe synthetic QA + EHR sparsity table**
```bash
python -m src.qa.generate_synthetic_ehrqa
python -m src.lakehouse.sparsity
```

**Step 4 — Build and load the Medical Knowledge Graph**
```bash
python -m src.mkg.cooccurrence
python -m src.mkg.neo4j_loader
```

**Step 5 — Build the FAISS text index**
```bash
python -m src.retrieval.embedder
```

**Step 6 — Fine-tune MedGemma with QLoRA** *(now retrieval-augmented — requires
Neo4j running and the FAISS index from Step 5)*
```bash
python -m src.model.train_qlora
```

**Step 7 — Generate router dataset**
```bash
python -m src.router.build_router_dataset
```

**Step 8 — Generate oracle labels** *(~45–60 min on RTX 4050)*
```bash
python -m src.router.oracle_labels
```

**Step 9 — Train the adaptive router**
```bash
# Default parameters
python -m src.router.train_router

# With hyperparameter tuning (slower, better results)
python -m src.router.train_router --tune
```

**Step 10 — Run final evaluation** *(held-out set, first and only use)*
```bash
# Full run — refuses to run on fewer than 50 questions unless overridden
python -m src.evaluation.run_evaluation

# Intentional quick smoke test only (never mistake this for the final run)
python -m src.evaluation.run_evaluation --max-samples 30 --allow-small-sample
```

---

## Evaluation Outputs

All outputs are written to `experiments/results/final_eval/`:

| File | Contents |
|---|---|
| `summary_table.csv` | Main results table (Table 2 in paper) |
| `sparsity_breakdown.csv` | H2 analysis by EHR sparsity bucket |
| `qtype_breakdown.csv` | Results by question type |
| `hallucination_report.csv` | Hallucination rates per system |
| `efficiency_report.csv` | Latency and token cost analysis |
| `figures/summary_metrics.png` | Bar chart: BLEU / ROUGE-L / BERTScore |
| `figures/sparsity_heatmap_*.png` | H2 heatmaps |
| `figures/latency_vs_quality.png` | Pareto plot: cost vs quality |
| `figures/hallucination_rates.png` | Hallucination breakdown by system |
| `figures/router_mode_distribution.png` | Router decisions by sparsity |

Router training additionally produces (in `models/router/`):

| File | Contents |
|---|---|
| `confusion_matrix_router.png` | Router confusion matrix |
| `shap_importance_bar.png` | SHAP feature importance (top 20) |
| `learning_curve.png` | Train/val F1 vs dataset size |
| `calibration_curve.png` | Reliability diagram |
| `baseline_comparison.png` | Router vs all baselines |
| `sparsity_breakdown.png` | Router accuracy by EHR sparsity (H2) |
| `error_analysis.csv` | Misclassified questions for qualitative analysis |

---

## Hardware

All experiments run on a single consumer-grade GPU:

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM | 6 GB |
| OS | Windows 11 |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |
| Conda env | `ehr-rag` (Python 3.11) |

QLoRA (4-bit NF4 quantization) is required to fit MedGemma 1.5-4B within 6 GB VRAM. The router (XGBoost) runs on CPU.

---

## Data Access and What This Repository Contains

This project is built on [MIMIC-IV](https://physionet.org/content/mimiciv/), a
de-identified critical-care dataset available under **credentialed access** and a
PhysioNet Data Use Agreement that prohibits redistribution.

**No patient data is in this repository, and none ever should be.** The following
are excluded by `.gitignore` and must stay local:

| Excluded | Why |
|---|---|
| `data/raw/`, `data/lakehouse/` | Raw and converted MIMIC-IV tables |
| `data/qa/`, `data/ehrqa_synthetic.*` | Generated QA pairs containing real admission IDs, diagnoses, lab values, medications, and demographics |
| `data/router/` | Router datasets embedding clinical note text in `prompt_context` |
| `embeddings/` | FAISS index built over clinical notes |
| `experiments/results/`, `experiments/logs/` | Per-question outputs containing patient content |
| `models/medgemma-4b-qlora/`, `*.safetensors` | Adapter weights fine-tuned on patient data |
| `models/_ARCHIVE_*/` | Superseded experiment artifacts (also patient-derived) |

`splits/patient_splits.json` **is** included: it holds only subject-ID integers with
no clinical content, and it is required to reproduce the exact partition (seed 42).

To obtain the data yourself: complete the [CITI](https://www.citiprogram.org/)
"Data or Specimens Only Research" course, register at
[PhysioNet](https://physionet.org), and submit a credentialed-access request for
MIMIC-IV. Every artifact above is then regenerable by running the pipeline below.

**This is a research prototype.** It has not been clinically validated, evaluated
prospectively, or reviewed for safety, and must not be used to inform patient care.

---

## Reproducing Results

Three things are needed for exact reproduction:

1. **Patient splits** — `splits/patient_splits.json` is committed to the repo. Do not regenerate it.
2. **Random seed** — All scripts use `seed=42` throughout.
3. **Model version** — Fine-tuning uses `google/medgemma-1.5-4b-it`, QLoRA r=16, α=32.

The oracle label generation script logs the exact model version, adapter path, scoring weights, and timestamp to `experiments/results/oracle_*.json` for full traceability.

### Post-hoc analysis scripts

After `run_evaluation`, these reproduce every statistic in the paper:

```bash
python -m src.evaluation.analysis            # bootstrap CIs, paired Wilcoxon, latency split
python -m src.evaluation.router_ablation     # question-only vs patient-only vs full router
python -m src.evaluation.recompute_grounding # length-controlled grounding measure
```

### A note on reproducibility and this project's history

[RESEARCH_LOG.md](RESEARCH_LOG.md) is the complete, unedited record of this
project — every bug, audit, wrong hypothesis, and correction, in order. It is
long and deliberately unflattering. Several headline results were overturned by
later audits (a context-echoing generator caused by unmasked loss, a router that
lost to a lookup table, a "patient-adaptive" claim disproved by ablation). If you
are building on this work, read it before trusting any intermediate artifact.

---

## Known Limitations

Stated plainly, because they bound what these results support:

- **Questions are template-generated** from structured EHR fields, not written by
  clinicians. For several question types the gold answer appears verbatim in the
  T+E/T+E+K context, which inflates absolute lexical scores.
- **Small scale**: 300 held-out questions over 61 admissions, one seed, one base
  model, one dataset. The high-sparsity subgroup has only 30 questions.
- **The knowledge graph is small** — 25 diseases, 215 hand-curated guideline
  edges — not extracted from a standard biomedical ontology.
- **Hallucination measures are automatic heuristics**, not clinician-adjudicated.
  There is no human evaluation and no inter-annotator agreement study.
- **EHR-contradiction is not measurable for mode T**: the detector inspects an EHR
  snapshot that mode T never receives, so it is reported as `n/a`, not 0%.

---

## Paper

The manuscript is in [`paper/main.tex`](paper/main.tex) (IEEEtran conference
format). Figures live in `paper/plots/`. Author block and final reference
verification are still outstanding.

---

## Citation

```bibtex
@misc{medragrouter2026,
  title  = {When Does the Knowledge Graph Help? Question-Driven Retrieval
            Routing for EHR-Grounded Clinical Question Answering},
  author = {Daksh},
  year   = {2026},
  note   = {M.Tech thesis work, MAHE Bengaluru},
  url    = {https://github.com/novvacode/med-rag-router}
}
```

---

## Acknowledgements

- [PhysioNet](https://physionet.org) and the MIMIC-IV team for the dataset
- [Google DeepMind](https://deepmind.google) for MedGemma
- [HuggingFace](https://huggingface.co) for Transformers, PEFT, and TRL
- [Neo4j](https://neo4j.com) for the graph database
- Prof. Jayita Saha (MAHE Bengaluru) for research supervision

---

<div align="center">
<sub>Built by <a href="https://github.com/novvacode">novvacode</a> · MAHE Bengaluru · 2026</sub>
</div>
