"""
src/model/train_qlora.py

Research-grade Supervised Fine-Tuning (SFT) script for MedGemma 1.5 (4B).
Optimized for 6 GB VRAM GPUs (e.g., RTX 4050) using 4-bit NF4 QLoRA.

Dependencies:
    transformers==5.12.1, trl==1.6.0, peft==0.19.1, accelerate==1.14.0,
    bitsandbytes==0.49.2, huggingface_hub>=0.24.0

Usage:
    python -m src.model.train_qlora
"""

import os
import sys
import gc
import json
import math
import glob
import platform
import importlib.metadata as importlib_metadata
from datetime import datetime

# Must be set before `torch`/CUDA context initialization to take effect.
# Reduces PyTorch caching-allocator fragmentation under the wide allocation-
# size variance introduced by retrieval-augmented (variable-length)
# training prompts. Pure memory-management setting — no effect on training
# data, labels, or results. See RESEARCH_LOG.md, 2026-08-07 BSOD incident.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from src.model.prompts import build_user_message
from src.model.context_budget import build_budgeted_context, MAX_SEQ_LENGTH

# ── Config ────────────────────────────────────────────────────────────────────

# Model & Paths
MODEL_ID = "google/medgemma-1.5-4b-it"
# data/qa/ is canonical (written by src/qa/generate_synthetic_ehrqa.py).
# Was previously data/lakehouse/qa/ — a stale, pre-2026-08-06-audit snapshot
# with different hadm_ids and zero contraindication_check examples. The
# 2026-08-08 training run was inadvertently trained on that stale file (its
# own log's "Unique admissions: 167" matches the stale file exactly, not
# the fresh file's 137). See RESEARCH_LOG.md, 2026-08-09 entry.
DATA_PATH = Path("data/qa/ehrqa_finetune.parquet")
OUTPUT_DIR = Path("models/medgemma-4b-qlora")

# Reproducibility
SEED = 42
set_seed(SEED)

# QLoRA Hyperparameters
LORA_R = 16
LORA_ALPHA = 32
# Raised 0.05 -> 0.10 after the 2026-08-08 run overfit hard: eval_loss
# plateaued by ~epoch 0.5 (~60/240 steps) while train_loss kept falling
# toward ~0.03, and the resulting adapter frequently generated degenerate
# output (echoing retrieved-passage context instead of answering — see
# RESEARCH_LOG.md, 2026-08-09 "oracle answers are context echoes" entry).
# More dropout is a standard, low-risk regularization response to this
# specific overfitting signature.
LORA_DROPOUT = 0.10
TARGET_MODULES = "all-linear"  # Best practice for Gemma architectures

# Context modes sampled during training, and their relative sampling weights.
#
# PREVIOUSLY: this script only ever trained on a hand-flattened EHR-snapshot
# string (a format distinct from ALL THREE inference-time prompt structures
# produced by src/retrieval/retriever.py::RetrievalResult.prompt_context).
# The model never saw retrieved note passages or KG facts during training,
# so at evaluation time — when it IS given those — it was operating out of
# distribution. This was a self-documented gap ("Change 9... not implemented
# yet") and is finding #4 of the 2026-08-06 audit (RESEARCH_LOG.md).
#
# FIX: load_and_prep_data() now calls the real Retriever for every training
# example, so training prompts are built with the exact same
# RetrievalResult.prompt_context formatting (## Retrieved Clinical Passages /
# ## Patient EHR Snapshot / ## Relevant Medical Knowledge) used at inference
# in run_evaluation.py. Each example's mode is sampled independently and
# reproducibly (seeded) from CONTEXT_MODE_WEIGHTS, so the model gets
# balanced exposure to all three retrieval configurations the router can
# select at inference time, rather than being systematically better at
# whichever single mode it happened to be trained on.
CONTEXT_MODE_WEIGHTS = {"T": 1.0, "T+E": 1.0, "T+E+K": 1.0}

# Fraction of ehrqa_finetune.parquet held out as the SFTTrainer eval split.
# Lowered from 0.10 to 0.04 after a 2026-08-07 crash during the first eval
# pass on a 6 GB card: with per_device_eval_batch_size=1, a 100-example eval
# set (0.10 of 1000) took ~25+ minutes of sustained load per eval pass. A
# smaller eval set still tracks eval_loss for early stopping / best-model
# selection without that much sustained-load exposure. See RESEARCH_LOG.md.
EVAL_FRACTION = 0.04

# ── Pre-Flight Validation ─────────────────────────────────────────────────────

def get_hf_auth_status():
    """
    Determines whether the current environment is authenticated with the
    Hugging Face Hub, checking (in order):
      1. HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variables
      2. A token cached locally via `huggingface-cli login` / `hf auth login`
    Returns a tuple (is_authenticated: bool, source: str, username: str | None).
    """
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return True, "environment variable", None

    # huggingface_hub >= 0.24 exposes a module-level get_token() that is the
    # supported way to read whatever `hf auth login` / `huggingface-cli login`
    # cached, and correctly follows the newer credential storage locations.
    # HfFolder.get_token() is kept only as a fallback for older huggingface_hub
    # versions where the module-level helper doesn't exist yet.
    cached_token = None
    try:
        from huggingface_hub import get_token
        cached_token = get_token()
    except ImportError:
        try:
            from huggingface_hub import HfFolder
            cached_token = HfFolder.get_token()
        except Exception:
            cached_token = None
    except Exception:
        cached_token = None

    if cached_token:
        # Try to resolve the username too, for a more informative check.
        username = None
        try:
            from huggingface_hub import whoami
            info = whoami(token=cached_token)
            username = info.get("name")
        except Exception:
            pass
        return True, "cached login (huggingface-cli / hf auth login)", username

    return False, None, None


def run_preflight_checks():
    """Validates hardware, environment, and data readiness before starting."""
    print("═" * 60)
    print(" PRE-FLIGHT SYSTEM CHECK")
    print("═" * 60)

    # 1. CUDA Validation
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires an NVIDIA GPU.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cuda_version = torch.version.cuda
    print(f"✅ GPU        : {gpu_name}")
    print(f"✅ CUDA     : {cuda_version}")
    print(f"✅ VRAM     : {vram_gb:.2f} GB")

    if vram_gb < 5.5:
        print("[WARN] VRAM is dangerously low. OOM crashes may occur.")

    # 2. Data Validation
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset missing: {DATA_PATH.resolve()}")
    if DATA_PATH.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {DATA_PATH.resolve()}")
    print(f"✅ Dataset  : {DATA_PATH.name} located.")

    # 2b. Retrieval infra validation — training now builds prompts via the
    # real Retriever (see load_and_prep_data) so it needs the FAISS index
    # to exist. KG (Neo4j) is best-effort and checked separately at load time.
    faiss_index = Path("embeddings/notes_index.faiss")
    faiss_chunks = Path("embeddings/notes_chunks.parquet")
    if not faiss_index.exists() or not faiss_chunks.exists():
        raise FileNotFoundError(
            f"FAISS index not found ({faiss_index} / {faiss_chunks}). "
            "Training now builds retrieval-augmented prompts and requires the "
            "text index to exist first. Run: python src/retrieval/embedder.py"
        )
    print(f"✅ FAISS    : index found at {faiss_index}")

    # 3. HF Authentication Validation (env var OR cached `hf auth login` token)
    is_authenticated, source, username = get_hf_auth_status()
    if not is_authenticated:
        print("[WARN] No Hugging Face authentication detected (checked HF_TOKEN")
        print("[WARN] env var and the local login cache).")
        print("[WARN] MedGemma requires accepted terms of use on Hugging Face.")
        print("[WARN] Run `hf auth login` (or `huggingface-cli login`) first.")
    else:
        if username:
            print(f"✅ HF Auth  : Authenticated via {source} (user: {username}).")
        else:
            print(f"✅ HF Auth  : Authenticated via {source}.")

    print("═" * 60)


def detect_mixed_precision():
    """Detects if bfloat16 is supported natively by the GPU."""
    if torch.cuda.is_bf16_supported():
        print("[INFO] Bfloat16 supported. Using bf16 for mixed precision.")
        return torch.bfloat16, True, False
    else:
        print("[INFO] Bfloat16 NOT supported. Falling back to fp16.")
        return torch.float16, False, True


# ── Chat Template Role Resolution ─────────────────────────────────────────────

def resolve_assistant_role(tokenizer: AutoTokenizer) -> str:
    """
    Determines which role label the tokenizer's OWN chat template expects for
    the assistant turn, rather than guessing. We do this by asking the
    tokenizer to render a minimal two-turn conversation under each candidate
    label and accepting the first one that the tokenizer processes without
    raising an error. This defers entirely to the tokenizer's official
    chat_template instead of hardcoding an assumption about MedGemma's format.
    """
    candidates = ["model", "assistant"]
    working_role = None
    for role in candidates:
        probe = [
            {"role": "user", "content": "probe"},
            {"role": role, "content": "probe"},
        ]
        try:
            tokenizer.apply_chat_template(probe, tokenize=False)
            working_role = role
            break
        except Exception:
            continue

    if working_role is None:
        raise RuntimeError(
            "Could not determine a valid assistant role label from the "
            "tokenizer's chat_template. Inspect tokenizer.chat_template "
            "manually before proceeding."
        )

    print(f"[INFO] Tokenizer chat template accepts assistant role: '{working_role}'")
    return working_role


# ── Data Preparation ──────────────────────────────────────────────────────────

def _load_kg_module():
    """Best-effort Neo4j MKG client load, mirroring the pattern used by
    build_router_dataset.py / run_evaluation.py. Training can proceed with KG
    disabled (T+E+K examples degrade to T+E-shaped context), but a run with
    it disabled should not be treated as the real final fine-tune."""
    try:
        import src.mkg.retrieval as kg_module
        print("[INFO] Neo4j KG module loaded successfully — T+E+K training "
              "examples will include real KG facts.")
        return kg_module
    except Exception as e:
        print(f"[WARN] Could not initialize KG module: {e}")
        print("[WARN] T+E+K training examples will be built WITHOUT KG facts. "
              "Start Neo4j and re-run for a fully aligned fine-tune.")
        return None


def _sample_modes(n: int, weights: dict, seed: int) -> list[str]:
    """Reproducibly assign a retrieval mode to each training example."""
    modes = list(weights.keys())
    probs = np.array([weights[m] for m in modes], dtype=float)
    probs = probs / probs.sum()
    rng = np.random.RandomState(seed)
    return rng.choice(modes, size=n, p=probs).tolist()


def load_and_prep_data(tokenizer: AutoTokenizer) -> Dataset:
    print(f"[INFO] Loading fine-tuning dataset from {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    if df.empty:
        raise ValueError("The dataset DataFrame is completely empty.")
    if "hadm_id" not in df.columns:
        raise ValueError(
            "ehrqa_finetune.parquet has no hadm_id column — retrieval-augmented "
            "training prompts require it to look up each patient's context."
        )

    # ── Dataset sanity statistics (printed before formatting) ────────────────
    num_samples = len(df)
    num_unique_patients = df["patient_id"].nunique() if "patient_id" in df.columns else None
    num_unique_admissions = df["hadm_id"].nunique()
    avg_question_len = df["question"].astype(str).str.split().apply(len).mean()
    avg_answer_len = df["answer"].astype(str).str.split().apply(len).mean()

    print("─" * 60)
    print(" DATASET SANITY STATISTICS")
    print("─" * 60)
    print(f"  Samples              : {num_samples}")
    print(f"  Unique patients      : {num_unique_patients if num_unique_patients is not None else 'N/A (no patient_id column)'}")
    print(f"  Unique admissions    : {num_unique_admissions}")
    print(f"  Avg question length  : {avg_question_len:.2f} words")
    print(f"  Avg answer length    : {avg_answer_len:.2f} words")
    print("─" * 60)

    # Resolve the assistant role label from the tokenizer's own chat template
    # instead of assuming "model" is correct — see resolve_assistant_role().
    assistant_role = resolve_assistant_role(tokenizer)

    # ── Retrieval-aligned context construction ────────────────────────────────
    # Import locally to keep this script importable even when the retrieval
    # stack (FAISS index, embeddings) isn't built yet, for tooling that only
    # needs e.g. resolve_assistant_role().
    from src.retrieval.retriever import Retriever, Mode as RMode

    print("[INFO] Initializing Retriever for training-context construction "
          "(FAISS index + embedding model must already be built — run "
          "src/retrieval/embedder.py first if this fails)...")
    kg_module = _load_kg_module()
    retriever = Retriever(kg_module=kg_module)

    mode_assignments = _sample_modes(len(df), CONTEXT_MODE_WEIGHTS, seed=SEED)
    df = df.reset_index(drop=True)
    df["_context_mode"] = mode_assignments

    mode_map = {"T": RMode.T, "T+E": RMode.TE, "T+E+K": RMode.TEK}
    n_retrieval_failures = 0

    def format_prompt(row):
        nonlocal n_retrieval_failures
        mode_str = row["_context_mode"]
        hadm_id = int(row["hadm_id"]) if pd.notna(row.get("hadm_id")) else None

        try:
            if hadm_id is None:
                raise ValueError("missing hadm_id")
            result = retriever.retrieve(
                question=str(row["question"]), hadm_id=hadm_id, mode=mode_map[mode_str]
            )
            context = result.prompt_context
        except Exception as e:
            n_retrieval_failures += 1
            # Fall back to the flattened snapshot fields already present on
            # the synthetic QA row, so a handful of bad hadm_ids can't crash
            # an entire training run — but this should stay rare (see the
            # failure-rate check after dataset.map() below).
            age = row.get("age", "Unknown")
            gender = row.get("gender", "Unknown")
            diagnoses = row.get("diagnoses", "None")
            labs = row.get("labs", "None")
            medications = row.get("medications", "None")
            context = (f"Patient EHR Snapshot:\n"
                       f"Age/Gender: {age} {gender}\n"
                       f"Diagnoses: {diagnoses}\n"
                       f"Labs: {labs}\n"
                       f"Medications: {medications}")

        # Enforce per-section token budgets so the question and the gold
        # answer can never be truncated away, and so passages/EHR/KG each get
        # a fixed allowance (keeping the T / T+E / T+E+K contrast clean).
        # See src/model/context_budget.py for the full rationale.
        context = build_budgeted_context(context, tokenizer)

        # prompt / completion (NOT a single "text" field). This is what lets
        # SFTConfig(completion_only_loss=True) mask the prompt out of the
        # loss. Previously this returned one concatenated "text" string, so
        # loss was computed over the ~2,100-2,600-token context as well as
        # the ~24-token answer — i.e. >99% of the gradient signal was
        # "reproduce the context", which is what trained the model to echo.
        # See RESEARCH_LOG.md, 2026-08-10 pre-training audit, finding A.
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_user_message(context, row["question"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt_text, "completion": str(row["answer"])}

    dataset = Dataset.from_pandas(df)
    # remove_columns=... drops every original QA column so the resulting
    # dataset has EXACTLY {prompt, completion}. TRL keys its prompt-completion
    # (loss-masked) path off that schema; leaving stray columns risks it
    # falling back to plain language-modelling, which is the bug being fixed.
    _orig_cols = list(dataset.column_names)
    dataset = dataset.map(
        format_prompt,
        desc="Formatting Prompts (retrieval-augmented, budgeted)",
        remove_columns=_orig_cols,
    )
    retriever.close()
    # Free the retriever's GPU-resident embedding model before SFTTrainer
    # allocates for the (much larger) MedGemma model — this project targets
    # a 6 GB VRAM card, so freeing the small embedding model's memory here
    # is cheap insurance against VRAM contention with the 4-bit QLoRA model
    # already loaded by setup_model_and_tokenizer().
    del retriever
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    failure_rate = n_retrieval_failures / max(num_samples, 1)
    print(f"[INFO] Training context mode distribution: "
          f"{pd.Series(mode_assignments).value_counts().to_dict()}")
    print(f"[INFO] Retrieval fallback rate: {n_retrieval_failures}/{num_samples} "
          f"({failure_rate:.1%})")
    if failure_rate > 0.05:
        print("[WARN] More than 5% of training examples fell back to the "
              "non-retrieval-augmented context format. This weakens the "
              "train/inference alignment fix — investigate before treating "
              "this as the final fine-tune run.")

    # Token-length diagnostic + HARD ASSERTION.
    # The 2026-08-08 run silently truncated away the question and answer in
    # 88.3% of examples (max_length=768 vs. p50 context of ~2,575 tokens,
    # with truncation_mode="keep_start"). Nothing failed loudly; the model
    # just learned to continue context. This check makes that class of bug
    # impossible to repeat silently: if budgeting ever fails to keep full
    # sequences under MAX_SEQ_LENGTH, training aborts instead of starting a
    # ~27-hour run on quietly-corrupted data.
    n_check = min(200, len(dataset))
    full_lens, prompt_lens, completion_lens = [], [], []
    for i in range(n_check):
        ex = dataset[i]
        p_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
        prompt_lens.append(len(p_ids))
        completion_lens.append(len(c_ids))
        full_lens.append(len(p_ids) + len(c_ids))

    def _pct(vals, q):
        s = sorted(vals)
        return s[min(int(len(s) * q), len(s) - 1)]

    print(f"[INFO] Token lengths over {n_check} budgeted examples:")
    print(f"         prompt     p50={_pct(prompt_lens,.5)} p90={_pct(prompt_lens,.9)} max={max(prompt_lens)}")
    print(f"         completion p50={_pct(completion_lens,.5)} p90={_pct(completion_lens,.9)} max={max(completion_lens)}")
    print(f"         full       p50={_pct(full_lens,.5)} p90={_pct(full_lens,.9)} max={max(full_lens)}"
          f"   (MAX_SEQ_LENGTH={MAX_SEQ_LENGTH})")

    n_over = sum(1 for L in full_lens if L > MAX_SEQ_LENGTH)
    if n_over:
        raise ValueError(
            f"ABORT: {n_over}/{n_check} budgeted examples still exceed "
            f"MAX_SEQ_LENGTH={MAX_SEQ_LENGTH} (max seen {max(full_lens)}). "
            "Training would silently truncate the question/answer away — the "
            "exact bug that produced the 2026-08-08 context-echoing model. "
            "Lower the per-section budgets in src/model/context_budget.py."
        )
    print(f"[INFO] ✅ All {n_check} sampled sequences fit within MAX_SEQ_LENGTH "
          f"— question and gold answer are guaranteed intact.")

    dataset = dataset.train_test_split(test_size=EVAL_FRACTION, seed=SEED)

    print(f"[INFO] Train size : {len(dataset['train'])} samples")
    print(f"[INFO] Eval size  : {len(dataset['test'])} samples")
    return dataset, dict(pd.Series(mode_assignments).value_counts()), failure_rate


# ── Model Initialization ──────────────────────────────────────────────────────

def setup_model_and_tokenizer(torch_dtype):
    print(f"\n[INFO] Initializing tokenizer: {MODEL_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        raise RuntimeError(f"Failed to download/load tokenizer: {e}. Check HF Authentication.")

    # Safe padding configuration
    if tokenizer.pad_token is None:
        print("[INFO] Tokenizer pad_token is None. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[INFO] Configuring BitsAndBytes (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True
    )

    print(f"[INFO] Loading base model: {MODEL_ID}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=torch_dtype,
            attn_implementation="sdpa",
)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Enable gradient checkpointing (use_reentrant=False is required for newer transformers)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    print("\n[INFO] Trainable Parameters:")
    model.print_trainable_parameters()

    return model, tokenizer, peft_config


# Files a checkpoint written by SFTTrainer/Trainer.save_model()+_save_checkpoint()
# must all contain to be safely resumable. Discovered via the 2026-08-07
# BSOD incident: a crash mid-write left a "checkpoint-40" directory with
# complete model weights but a truncated optimizer.pt and no scheduler.pt/
# rng_state*.pth/trainer_state.json — silently trusting that directory would
# have either crashed resume_from_checkpoint or, worse, resumed with
# optimizer/scheduler/RNG state reset while believing it was a real resume.
REQUIRED_CHECKPOINT_FILES = [
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "adapter_model.safetensors",
    "adapter_config.json",
]


def _is_complete_checkpoint(ckpt_dir: str) -> tuple[bool, list[str]]:
    """Returns (is_complete, missing_file_list). RNG state filename varies
    by transformers version / distributed setup (rng_state.pth vs.
    rng_state_0.pth etc.), so it's checked via glob rather than exact name."""
    missing = [f for f in REQUIRED_CHECKPOINT_FILES
               if not os.path.exists(os.path.join(ckpt_dir, f))]
    if not glob.glob(os.path.join(ckpt_dir, "rng_state*.pth")):
        missing.append("rng_state*.pth")
    return (len(missing) == 0, missing)


def get_last_checkpoint() -> str | None:
    """Finds the most recent VALID (complete, safely resumable) checkpoint.
    Skips — with a loud warning — any checkpoint left incomplete by a crash
    mid-write, and automatically falls back to the next-newest valid one
    instead of either resuming from a corrupt checkpoint or silently
    restarting from step 0 without explanation."""
    if not OUTPUT_DIR.exists():
        print("[INFO] No output directory yet — starting from step 0.")
        return None

    checkpoints = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
        key=os.path.getmtime, reverse=True,
    )
    if not checkpoints:
        print("[INFO] No checkpoints found — starting from step 0.")
        return None

    for ckpt in checkpoints:
        is_complete, missing = _is_complete_checkpoint(ckpt)
        if is_complete:
            try:
                with open(os.path.join(ckpt, "trainer_state.json")) as f:
                    state = json.load(f)
                print(f"[INFO] Valid checkpoint found: {ckpt} "
                      f"(global_step={state.get('global_step')}, "
                      f"epoch={state.get('epoch'):.3f})")
            except Exception as e:
                print(f"[INFO] Valid checkpoint found: {ckpt} "
                      f"(could not read trainer_state.json for step/epoch: {e})")
            return ckpt
        else:
            print(f"[WARN] Skipping INCOMPLETE checkpoint {ckpt} "
                  f"(missing: {missing}) — likely interrupted mid-write by a crash.")

    print("[WARN] All checkpoints found were incomplete. Starting from step 0.")
    return None


# ── Experiment Metadata ───────────────────────────────────────────────────────

def _get_version(pkg_name: str) -> str:
    try:
        return importlib_metadata.version(pkg_name)
    except Exception:
        return "unknown"


class _NumpyJSONEncoder(json.JSONEncoder):
    """json.dump can't serialize numpy scalar types natively.
    context_mode_distribution comes from pd.Series(...).value_counts(),
    whose values are numpy int64 — this crashed the 2026-08-08 training run
    (27h48m of training completed successfully; only this final metadata
    write failed, after model/tokenizer/trainer-state were already safely on
    disk). See RESEARCH_LOG.md."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return super().default(o)


def save_training_metadata(peft_config: LoraConfig, training_args: SFTConfig,
                            epochs: int, learning_rate: float,
                            context_mode_distribution: dict, retrieval_failure_rate: float):
    """
    Persists a JSON manifest alongside the saved adapters capturing everything
    needed to reproduce this run: model id, dataset path, LoRA config,
    training hyperparameters, seed, library versions, and a timestamp.
    """
    metadata = {
        "model_id": MODEL_ID,
        "dataset_path": str(DATA_PATH),
        "context_mode_weights": CONTEXT_MODE_WEIGHTS,
        "context_mode_distribution": context_mode_distribution,
        "retrieval_fallback_rate": retrieval_failure_rate,
        "lora_config": {
            "r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "lora_dropout": peft_config.lora_dropout,
            "target_modules": peft_config.target_modules
                if isinstance(peft_config.target_modules, str)
                else list(peft_config.target_modules),
            "bias": peft_config.bias,
            "task_type": str(peft_config.task_type),
        },
        "training_args": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "max_length": training_args.max_length,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "optim": training_args.optim,
        },
        "seed": SEED,
        "library_versions": {
            "python": platform.python_version(),
            "torch": _get_version("torch"),
            "transformers": _get_version("transformers"),
            "trl": _get_version("trl"),
            "peft": _get_version("peft"),
            "accelerate": _get_version("accelerate"),
            "bitsandbytes": _get_version("bitsandbytes"),
        },
        "training_date": datetime.now().isoformat(timespec="seconds"),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT_DIR / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, cls=_NumpyJSONEncoder)
    print(f"[INFO] Saved training metadata to {metadata_path}")


# ── Training Pipeline ─────────────────────────────────────────────────────────

def main():
    run_preflight_checks()
    torch_dtype, use_bf16, use_fp16 = detect_mixed_precision()

    model, tokenizer, peft_config = setup_model_and_tokenizer(torch_dtype)
    dataset, context_mode_distribution, retrieval_failure_rate = load_and_prep_data(tokenizer)

    # Calculate estimated steps (ceil so a trailing partial batch is counted)
    batch_size = 1
    grad_accum = 8
    epochs = 2
    learning_rate = 2e-4
    effective_bs = batch_size * grad_accum
    steps_per_epoch = math.ceil(len(dataset["train"]) / effective_bs)
    total_steps = steps_per_epoch * epochs

    print("\n" + "═" * 60)
    print(f" TRAINING CONFIGURATION")
    print("═" * 60)
    print(f"  Effective Batch Size : {effective_bs}")
    print(f"  Steps / Epoch        : ~{steps_per_epoch}")
    print(f"  Estimated Steps      : ~{total_steps}")
    print(f"  Precision            : {'bf16' if use_bf16 else 'fp16'}")
    print("═" * 60)

    # TRL 1.6.0 SFTConfig
    # max_length lowered 1024 -> 768 after a 2026-08-07 crash: retrieval-
    # augmented prompts (up to 5 note passages + EHR snapshot + KG facts for
    # T+E+K) are structurally much longer than the old flattened-snapshot
    # text this was originally tuned for, and step time on a 6 GB card
    # correlates strongly with sequence length. See the token-length
    # diagnostic printed in load_and_prep_data() and RESEARCH_LOG.md.
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),

        # completion_only_loss=True masks the prompt out of the loss so ONLY
        # the gold answer contributes gradient. Requires the prompt/completion
        # dataset schema built in load_and_prep_data(). This is the primary
        # fix for the 2026-08-08 context-echoing model: previously the config
        # used dataset_text_field="text" with completion_only_loss unset
        # (None), so loss ran over the full ~2,100-2,600-token context too and
        # >99% of the training signal was "reproduce the context".
        completion_only_loss=True,

        # Single shared sequence budget (was 768 here, 2048 in
        # run_evaluation.py, and unbounded in oracle_labels.py — three
        # different effective context lengths across the pipeline).
        # Contexts are pre-budgeted per-section by context_budget.py, and the
        # assertion in load_and_prep_data() proves nothing overflows, so this
        # limit should never actually truncate anything.
        max_length=MAX_SEQ_LENGTH,

        # Memory & Batching
        # optim: non-paged adamw_8bit, not paged_adamw_8bit. Paged optimizers
        # use CUDA Unified Memory (UVM) demand-paging, which is documented as
        # unstable under sustained load on Windows WDDM (unlike Linux) and is
        # the primary suspect for the 2026-08-07 BSOD (KMODE_EXCEPTION_NOT_
        # HANDLED / nvlddmkm.sys). Unnecessary here regardless: LoRA has only
        # ~38M trainable params, so 8-bit optimizer state is tens of MB —
        # far too small to need paging. See RESEARCH_LOG.md.
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=1,
        optim="adamw_8bit",
        dataloader_pin_memory=True,
        remove_unused_columns=False,

        # Precision
        bf16=use_bf16,
        fp16=use_fp16,

        # Learning Dynamics
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        num_train_epochs=epochs,
        max_grad_norm=0.3,
        warmup_ratio=0.03,

        # Evaluation & Saving — fault-tolerance settings per the 2026-08-07
        # incomplete-checkpoint incident (RESEARCH_LOG.md):
        # eval_steps/save_steps: 20 (original) -> 40 -> 25 -> 15 now. A crash
        # can only ever lose progress made *since* the last complete
        # checkpoint, so this bounds that loss to ~15 steps regardless of
        # cause. save_steps must stay a multiple of eval_steps for
        # load_best_model_at_end to work, so both move together.
        # save_total_limit: 2 -> 3. The 2026-08-07 crash corrupted the
        # checkpoint being written *at the moment of the crash* — a second
        # crash could do the same to whatever is then the newest checkpoint.
        # Keeping 3 means get_last_checkpoint()'s validity-skip-and-fall-back
        # logic (see that function) has two older checkpoints in reserve,
        # not just one, before falling back to step 0.
        eval_strategy="steps",
        eval_steps=15,
        save_strategy="steps",
        save_steps=15,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Reproducibility
        seed=SEED,
        data_seed=SEED,

        # Logging
        logging_steps=5,
        report_to="none"
    )

    # Added after the 2026-08-08 run trained the full 2 epochs (240 steps)
    # despite eval_loss plateauing by ~step 60 — the ~180 extra steps of
    # continued training on a near-zero train_loss almost certainly drove
    # the overfitting that produced degenerate (context-echoing) generation.
    # patience=3 means training stops after 3 consecutive eval checks
    # (eval_steps=15 each) with no eval_loss improvement — i.e. up to ~45
    # steps of no-improvement buffer past the best point, not stopping on a
    # single noisy eval. Requires load_best_model_at_end=True (already set)
    # to actually restore the best checkpoint's weights at the end.
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    callbacks=[early_stopping],
)

    last_checkpoint = get_last_checkpoint()
    print("\n" + "═" * 60)
    if last_checkpoint:
        print(f" RESUMING FROM CHECKPOINT: {last_checkpoint}")
        print(" Model weights, optimizer state, LR scheduler state, RNG")
        print(" state, and Trainer state (global_step/epoch/best-metric")
        print(" tracking) will all be restored from this checkpoint by")
        print(" transformers' built-in resume_from_checkpoint mechanism —")
        print(" this only runs at all because get_last_checkpoint() already")
        print(" validated every required file is present (see that function).")
    else:
        print(" STARTING FROM STEP 0 — no valid checkpoint was found.")
    print("═" * 60)

    print("\n[INFO] 🚀 Commencing QLoRA Fine-Tuning...")
    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[ERROR] CUDA Out of Memory: {e}")
        print("[HINT] Try one or more of the following to reduce VRAM usage:")
        print("       1. Reduce `max_length` in SFTConfig (currently 1024).")
        print(f"       2. Reduce LoRA rank `LORA_R` (currently {LORA_R}).")
        print(f"       3. Increase `gradient_accumulation_steps` (currently {grad_accum})")
        print("          while keeping per_device_train_batch_size at 1.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        sys.exit(1)

    print(f"\n[INFO] Training complete! Saving final model to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    trainer.save_state()
    save_training_metadata(peft_config, training_args, epochs, learning_rate,
                            context_mode_distribution, retrieval_failure_rate)
    print("[INFO] ✅ Saved successfully.")


if __name__ == "__main__":
    main()