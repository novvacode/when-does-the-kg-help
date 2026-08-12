import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel

from src.model.prompts import build_user_message

# ==========================================================
# Configuration
# ==========================================================
# NOTE: this is a manual, single-example smoke-test script, not part of the
# pipeline in README's "Running the Pipeline" steps. As of the 2026-08-06
# audit fix, actual training/eval prompts are retrieval-augmented (built via
# src.retrieval.retriever.Retriever, see src/model/train_qlora.py and
# src/evaluation/run_evaluation.py) — the flattened EHR-snapshot context
# below is a simplified approximation for a quick manual check, not the
# real training/inference format.

BASE_MODEL = "google/medgemma-1.5-4b-it"
ADAPTER_PATH = "models/medgemma-4b-qlora"
EVAL_DATA = "data/qa/ehrqa_eval.parquet"  # data/qa/ is canonical — see RESEARCH_LOG.md, 2026-08-09

# ==========================================================
# Load Evaluation Sample
# ==========================================================

print("=" * 70)
print("Loading evaluation dataset...")
print("=" * 70)

df = pd.read_parquet(EVAL_DATA)

# Change index if you want another example
row = df.iloc[0]

print(f"\nQuestion Type : {row['question_type']}")
print(f"Question      : {row['question']}")
print(f"Ground Truth  : {row['answer']}")

# ==========================================================
# Build Prompt (same format as training)
# ==========================================================

context = (
    f"Patient EHR Snapshot:\n"
    f"Age/Gender: {row['age']} {row['gender']}\n"
    f"Diagnoses: {row['diagnoses']}\n"
    f"Labs: {row['labs']}\n"
    f"Medications: {row['medications']}"
)

user_prompt = build_user_message(context, row["question"])

# ==========================================================
# Load Tokenizer
# ==========================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Detect correct assistant role (same as training)

assistant_role = "model"

try:
    tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "hi"},
            {"role": "model", "content": "hello"},
        ],
        tokenize=False,
    )
    assistant_role = "model"
except Exception:
    assistant_role = "assistant"

print(f"Using assistant role: {assistant_role}")

# ==========================================================
# Load Base Model
# ==========================================================

print("Loading base model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# ==========================================================
# Load LoRA Adapter
# ==========================================================

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
)

model.eval()

# ==========================================================
# Apply Chat Template
# ==========================================================

messages = [
    {
        "role": "user",
        "content": user_prompt,
    }
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    formatted_prompt,
    return_tensors="pt",
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

# ==========================================================
# Generate
# ==========================================================

print("\nGenerating answer...\n")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

generated_answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

# ==========================================================
# Results
# ==========================================================

print("=" * 70)
print("GENERATED ANSWER")
print("=" * 70)
print(generated_answer)

print()

print("=" * 70)
print("GROUND TRUTH")
print("=" * 70)
print(row["answer"])

print()

print("=" * 70)
print("QUESTION")
print("=" * 70)
print(row["question"])

print()

print("=" * 70)
print("PATIENT SNAPSHOT")
print("=" * 70)
print(context)