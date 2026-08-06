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

import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from src.model.prompts import build_user_message

# ── Config ────────────────────────────────────────────────────────────────────

# Model & Paths
MODEL_ID = "google/medgemma-1.5-4b-it"
DATA_PATH = Path("data/lakehouse/qa/ehrqa_finetune.parquet")
OUTPUT_DIR = Path("models/medgemma-4b-qlora")

# Reproducibility
SEED = 42
set_seed(SEED)

# QLoRA Hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
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

        messages = [
            {"role": "user", "content": build_user_message(context, row["question"])},
            {"role": assistant_role, "content": str(row["answer"])}
        ]

        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": formatted_text}

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_prompt, desc="Formatting Prompts (retrieval-augmented)")
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

    dataset = dataset.remove_columns(["_context_mode"])
    dataset = dataset.train_test_split(test_size=0.1, seed=SEED)

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


def get_last_checkpoint() -> str | None:
    """Finds the most recent checkpoint if training was interrupted."""
    if OUTPUT_DIR.exists():
        checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
        if checkpoints:
            return max(checkpoints, key=os.path.getmtime)
    return None


# ── Experiment Metadata ───────────────────────────────────────────────────────

def _get_version(pkg_name: str) -> str:
    try:
        return importlib_metadata.version(pkg_name)
    except Exception:
        return "unknown"


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
        json.dump(metadata, f, indent=2)
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
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        dataset_text_field="text",
        max_length=1024,  # TRL 1.6.0 renamed this from `max_seq_length` -> `max_length`

        # Memory & Batching
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=1,
        optim="paged_adamw_8bit",
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

        # Evaluation & Saving
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
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

    trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

    last_checkpoint = get_last_checkpoint()
    if last_checkpoint:
        print(f"\n[INFO] Resuming training from checkpoint: {last_checkpoint}")

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