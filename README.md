<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/MedGemma-1.5--4B-4285F4?style=flat-square&logo=google&logoColor=white"/>
<img src="https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white"/>
<img src="https://img.shields.io/badge/Neo4j-5.x-008CC1?style=flat-square&logo=neo4j&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Research-orange?style=flat-square"/>

<br/><br/>

# When Does the Knowledge Graph Help?

### Adaptive RAG Routing for EHR-Grounded Clinical Question Answering

*M.Tech Research Project · MAHE Bengaluru · Target: CHIL / AMIA / EMNLP-BioNLP*

</div>

---

## The Problem

Most medical RAG systems apply the same expensive retrieval pipeline to every question — text search, EHR lookup, and Knowledge Graph traversal, every single time. This is wasteful and sometimes harmful: when a patient's EHR already contains all the relevant facts, extra KG context adds noise, not signal.

This project asks a sharper question: **can a learned router that decides *when* to use each retrieval source reduce both hallucinations and latency, compared to always-on hybrid KG-RAG?**

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

> Full results available after evaluation on the held-out MIMIC-IV EHR-QA set.
> A prior "final" evaluation run in this repo only covered a single question
> and does not reflect real performance — see [RESEARCH_LOG.md](RESEARCH_LOG.md)
> for the full audit that found this and the fixes applied before the real
> held-out run.

Evaluation compares five systems:

| System | Description |
|---|---|
| **T** | Text-only RAG (baseline) |
| **T+E** | Text + EHR snapshot |
| **T+E+K** | Always-on hybrid — the system we compete against |
| **Random** | Random mode selection (lower bound) |
| **Router** | Our proposed adaptive system |

Metrics: BLEU · ROUGE-L · BERTScore-F1 · EHR-contradiction rate · KG-contradiction rate · Unsupported rate · Latency (ms) · VRAM (MB)

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
  (`data/lakehouse/qa/ehrqa_finetune.parquet`)
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

## Data Access

This project uses [MIMIC-IV](https://physionet.org/content/mimiciv/), a freely available but credentialed dataset.

To access MIMIC-IV:
1. Complete the [CITI Program](https://www.citiprogram.org/) "Data or Specimens Only Research" training
2. Register at [PhysioNet](https://physionet.org) and upload your CITI certificate
3. Submit a credentialed access request for MIMIC-IV

Raw data, processed Parquet files, and fine-tuned model weights are not included in this repository. The `splits/patient_splits.json` file (patient ID assignments, seed=42) is included for reproducibility — it contains no patient data.

---

## Reproducing Results

Three things are needed for exact reproduction:

1. **Patient splits** — `splits/patient_splits.json` is committed to the repo. Do not regenerate it.
2. **Random seed** — All scripts use `seed=42` throughout.
3. **Model version** — Fine-tuning uses `google/medgemma-1.5-4b-it`, QLoRA r=16, α=32.

The oracle label generation script logs the exact model version, adapter path, scoring weights, and timestamp to `experiments/results/oracle_*.json` for full traceability.

---

## Target Venues

| Venue | Type | Notes |
|---|---|---|
| [CHIL 2027](https://chilconference.org) | Conference | Primary target — ML for health, rigorous evaluation |
| [AMIA 2026 Annual](https://amia.org) | Conference | Clinical informatics audience |
| [EMNLP BioNLP Workshop](https://aclweb.org/aclwiki/BioNLP_Workshop) | Workshop | NLP + biomedical angle |
| JAMIA / JBI | Journal | Extended version with clinical co-author |

---

## Citation

```bibtex
@article{daksh2026adaptiverag,
  title   = {When Does the Knowledge Graph Help? Adaptive RAG Routing
             for EHR-Grounded Clinical Question Answering},
  author  = {Daksh},
  journal = {Work in Progress — M.Tech Thesis, MAHE Bengaluru},
  year    = {2026}
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
