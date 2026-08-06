"""
oracle_labels.py
================
Oracle Label Generation for the Adaptive Retrieval Router.

Integrates advanced CUDA recovery, Clinical BERT semantic scoring,
deterministic GPU math, prompt archiving, and comprehensive reporting.

Usage:
    python -m src.router.oracle_labels
    python -m src.router.oracle_labels --checkpoint-interval 20 --use-cache False
"""

import gc
import json
import logging
import random
import sys
import time
import argparse
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from bert_score import BERTScorer
from peft import PeftModel
from rouge_score import rouge_scorer as rouge_lib
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.model.prompts import build_user_message

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OracleConfig:
    base_model:           str = "google/medgemma-1.5-4b-it"
    adapter_path:         str = "models/medgemma-4b-qlora"
    router_train:         str = "data/router/router_train_examples.parquet"
    router_val:           str = "data/router/router_val_examples.parquet"
    output_dir:           str = "data/router"
    checkpoint_dir:       str = "data/router/checkpoints"
    report_dir:           str = "experiments/results"

    modes: list = field(default_factory=lambda: ["T", "T+E", "T+E+K"])

    # ── Generation settings ──
    max_new_tokens:       int   = 128
    do_sample:            bool  = False
    use_cache:            bool  = True  
    
    # ── Scoring weights ──
    weight_bert:          float = 0.60
    weight_rouge:         float = 0.25
    weight_em:            float = 0.15
    hallucination_lambda: float = 0.30

    # ── Semantic Evaluation Model Choice ──
    bert_score_model:     str = "roberta-large"
    bert_score_device:    str = "cpu"

    checkpoint_interval:  int = 10
    seed:                 int = 42
    imbalance_threshold:  float = 0.75


# ══════════════════════════════════════════════════════════════════════════════
# Helper Utilities
# ══════════════════════════════════════════════════════════════════════════════

def str2bool(v) -> bool:
    """Robust string to boolean parser for argparse to fix type=bool evaluation bug."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False).")


def setup_logging(log_dir: Path, split_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"oracle_{split_name}_{timestamp}.log"

    logger = logging.getLogger(f"oracle.{split_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s", datefmt="%H:%M:%S"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  |  %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# Module 1 — Model Loader & Validator
# ══════════════════════════════════════════════════════════════════════════════

class ModelLoader:
    def __init__(self, cfg: OracleConfig, logger: logging.Logger):
        self.cfg    = cfg
        self.logger = logger
        self.model     = None
        self.tokenizer = None

    def load(self):
        self.logger.info(f"Loading tokenizer from: {self.cfg.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.logger.info("Loading base model (4-bit NF4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )

        self.logger.info(f"Loading LoRA adapter from: {self.cfg.adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, self.cfg.adapter_path)
        self.model.eval()
        return self.model, self.tokenizer

def validate_schema(df: pd.DataFrame, split_name: str, logger: logging.Logger):
    required_columns = [
        "hadm_id",
        "question",
        "answer",
        "prompt_context",
        "mode",
        "n_labs",
        "n_diag",
        "n_meds",
        "sparsity_score",
        "sparsity_bucket",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        error_msg = f"Schema validation failed for {split_name}. Missing required columns: {missing}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info(f"Schema validation passed successfully for split: {split_name}")


# ══════════════════════════════════════════════════════════════════════════════
# Module 2 — Answer Generator
# ══════════════════════════════════════════════════════════════════════════════

class AnswerGenerator:
    def __init__(self, model, tokenizer, cfg: OracleConfig, logger: logging.Logger):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.logger    = logger

    def generate(self, prompt_context: str, question: str) -> tuple[str, float, int]:
        messages = [{"role": "user", "content": build_user_message(prompt_context, question)}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        for attempt in range(2):
            try:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.cfg.max_new_tokens,
                        do_sample=self.cfg.do_sample,
                        use_cache=self.cfg.use_cache,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                latency_ms = (time.perf_counter() - t0) * 1000
                generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                return answer, round(latency_ms, 1), len(generated_tokens)

            except RuntimeError as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ["cuda", "launch failure", "device-side assert", "out of memory"]):
                    self.logger.warning(f"CUDA Exception triggered during attempt {attempt+1}: {e}. Running safety recovery sequence...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    if attempt == 1:
                        self.logger.error("CUDA infrastructure failed recovery attempt. Skipping sample configuration sequence.")
                        return "", 0.0, 0
                else:
                    self.logger.error(f"Inference execution engine failure: {e}")
                    return "", 0.0, 0

        return "", 0.0, 0

    def cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# Module 3 — Metric Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class MetricEvaluator:
    def __init__(self, cfg: OracleConfig, logger: logging.Logger):
        self.cfg    = cfg
        self.logger = logger
        self.logger.info(f"Initialising Evaluation Engine with Semantic Scorer: {cfg.bert_score_model}")
        self.bert_scorer = BERTScorer(model_type=cfg.bert_score_model, device=cfg.bert_score_device, rescale_with_baseline=False)
        self.rouge = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)

    @staticmethod
    def exact_match(hypothesis: str, reference: str) -> float:
        return 1.0 if " ".join(hypothesis.lower().strip().split()) == " ".join(reference.lower().strip().split()) else 0.0

    def score_batch(self, hypotheses: list[str], reference: str) -> list[dict]:
        references = [reference] * len(hypotheses)
        _, _, F1 = self.bert_scorer.score(hypotheses, references, verbose=False)
        bert_f1s = F1.tolist()

        scores = []
        for i, hyp in enumerate(hypotheses):
            rouge_l = self.rouge.score(reference, hyp)["rougeL"].fmeasure
            em      = self.exact_match(hyp, reference)
            scores.append({
                "bert_f1": round(bert_f1s[i], 4),
                "rouge_l": round(rouge_l, 4),
                "em":      round(em, 4),
            })
        return scores

    def composite_score(self, bert_f1: float, rouge_l: float, em: float, hallucination_flag: int) -> float:
        raw = (self.cfg.weight_bert * bert_f1 + self.cfg.weight_rouge * rouge_l + self.cfg.weight_em * em)
        return round(raw - self.cfg.hallucination_lambda * hallucination_flag, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Module 4 — Hallucination Detector
# ══════════════════════════════════════════════════════════════════════════════

class HallucinationDetector:
    _ABNORMAL_SIGNALS = ["abnormal", "high", "low", "elevated", "reduced", "critical", "flag", "h]", "l]"]
    _NORMAL_CLAIMS    = ["no abnormal", "labs are normal", "all labs normal", "within normal"]
    _NO_MED_CLAIMS    = ["no medications", "not on any medications", "not taking any"]
    
    def detect(self, answer: str, prompt_context: str) -> tuple[int, list[str]]:
        if not answer: return 0, []
        ans_lower, ctx_lower, reasons = answer.lower(), prompt_context.lower(), []

        if any(sig in ctx_lower for sig in self._ABNORMAL_SIGNALS) and any(claim in ans_lower for claim in self._NORMAL_CLAIMS):
            reasons.append("H1:claims_normal_labs_contradicts_context")

        if "medication" in ctx_lower and any(claim in ans_lower for claim in self._NO_MED_CLAIMS):
            reasons.append("H2:claims_no_meds_contradicts_context")

        if "mg" in ans_lower and "mg" not in ctx_lower:
            reasons.append("H6:invented_dosage_mg")

        _specific_diseases = ["heart failure", "copd", "cirrhosis", "sepsis", "acute kidney injury", "myocardial infarction", "stroke"]
        for disease in _specific_diseases:
            if disease in ans_lower and disease not in ctx_lower:
                reasons.append(f"H3:unsupported_diagnosis:{disease}")
                break

        return 1 if reasons else 0, reasons


# ══════════════════════════════════════════════════════════════════════════════
# Module 5 — Oracle Selector
# ══════════════════════════════════════════════════════════════════════════════

class OracleSelector:
    def __init__(self, cfg: OracleConfig):
        self.cfg = cfg
        self.mode_costs = {"T": 0.000, "T+E": 0.001, "T+E+K": 0.002}

    def select(self, composite_scores: dict[str, float]) -> str:
        adjusted_scores = {mode: score - self.mode_costs.get(mode, 0) for mode, score in composite_scores.items()}
        return max(adjusted_scores, key=adjusted_scores.get)


# ══════════════════════════════════════════════════════════════════════════════
# Module 6 & 7 — Checkpoints & Reports
# ══════════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    def __init__(self, cfg: OracleConfig, split_name: str, logger: logging.Logger):
        self.ckpt_file = Path(cfg.checkpoint_dir) / f"{split_name}_checkpoint.parquet"
        self.ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def load_completed(self) -> tuple[list[dict], set[str]]:
        if not self.ckpt_file.exists(): return [], set()
        df = pd.read_parquet(self.ckpt_file)
        return df.to_dict(orient="records"), set(df["question_key"].tolist())

    def save(self, results: list[dict]):
        if results: pd.DataFrame(results).to_parquet(self.ckpt_file, index=False)

    def cleanup(self):
        if self.ckpt_file.exists(): self.ckpt_file.unlink()

class ReportGenerator:
    def __init__(self, cfg: OracleConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        Path(cfg.report_dir).mkdir(parents=True, exist_ok=True)

    def generate(self, split_name: str, results: list[dict], start_time: float, cfg: OracleConfig, total_tokens: int):
        if not results: return
        df = pd.DataFrame(results)
        total_time = time.time() - start_time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        n_skipped = sum(1 for r in results if r.get("skipped", False))
        n_successful = len(results) - n_skipped
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

        report = {
            "meta": {
                "split": split_name, 
                "total_questions": len(df), 
                "successful": n_successful,
                "skipped": n_skipped,
                "total_runtime_s": round(total_time, 1),
                "throughput_q_per_sec": round(n_successful / total_time, 2) if total_time > 0 else 0,
                "total_generated_tokens": total_tokens,
                "peak_vram_gb": round(peak_vram, 2)
            },
            "model": {"base_model": cfg.base_model, "seed": cfg.seed},
            "label_distribution": {k: {"count": v, "pct": round(v / len(df) * 100, 1)} for k, v in df["best_mode"].value_counts().to_dict().items() if k not in ["FAILED", "FAILED_GENERATION", "MISSING_MODES"]},
            "mean_metrics_per_mode": {mode: {"composite": round(df[f"composite_{mode.replace('+','').lower()}"].mean(), 4)} for mode in cfg.modes if f"composite_{mode.replace('+','').lower()}" in df.columns},
            # Diagnostic for RESEARCH_LOG.md finding #7: HallucinationDetector
            # uses keyword heuristics that could plausibly fire more often on
            # longer contexts (T+E+K has strictly more context than T/T+E),
            # independent of true answer quality, which would bias oracle
            # labels away from T+E+K for reasons unrelated to hallucination.
            # These per-mode rates make that hypothesis checkable rather than
            # assumed — compare halluc_rate and mean prompt length across
            # modes after each oracle generation run.
            "hallucination_diagnostic_per_mode": {
                mode: {
                    "halluc_rate": round(df[f"halluc_{mode.replace('+','').lower()}"].mean(), 4),
                    "mean_prompt_chars": round(
                        df[f"prompt_{mode.replace('+','').lower()}"].astype(str).str.len().mean(), 1
                    ),
                }
                for mode in cfg.modes
                if f"halluc_{mode.replace('+','').lower()}" in df.columns
                and f"prompt_{mode.replace('+','').lower()}" in df.columns
            },
        }

        json_path = Path(self.cfg.report_dir) / f"oracle_{split_name}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f: json.dump(report, f, indent=2)
        
        print(f"\n{'─'*50}\n Oracle Metric Summary — {split_name}\n{'─'*50}")
        print(f" Total Questions Processed : {len(df)}")
        print(f" Successful Adaptations   : {n_successful}")
        print(f" Pipeline Failures/Skipped: {n_skipped}")
        print(f" System Execution Runtime  : {report['meta']['total_runtime_s']} sec")
        print(f" Compute Throughput Metrics: {report['meta']['throughput_q_per_sec']} Q/sec")
        print(f" Peak Hardware VRAM Usage  : {peak_vram:.2f} GB")
        print(f" Aggregated Output Tokens  : {total_tokens}")
        print(f"{'─'*50}\n Target Label Distribution Selection Strategy:")
        for mode, stats in report["label_distribution"].items():
            print(f"   {mode:<10}  {stats['count']:>4}  ({stats['pct']:>5.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# Core Pipeline Process
# ══════════════════════════════════════════════════════════════════════════════

def process_split(split_name: str, input_path: str, output_path: str, cfg: OracleConfig, components: dict):
    logger = components["logger"]
    df = pd.read_parquet(input_path)
    
    # Validation step execution
    validate_schema(df, split_name, logger)
    
    # Fixed Peak VRAM Reset accumulation carryover bug
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"Reset peak CUDA memory statistics tracker for split: {split_name}")

    df["question_key"] = df["hadm_id"].astype(str) + "||" + df["question"]
    
    ckpt = CheckpointManager(cfg, split_name, logger)
    results, done_keys = ckpt.load_completed()
    pending = [(k, g) for k, g in df.groupby("question_key", sort=False) if k not in done_keys]
    

    total_tokens_generated = 0
    start_time = time.time()  
    
    pbar = tqdm(pending, total=len(pending), desc=f"Oracle Label Processing [{split_name}]")

    try:
        for idx, (qkey, group) in enumerate(pbar):
            
            # Missing Mode Validation Gate Check
            expected = set(cfg.modes)
            available = set(group["mode"].unique())
            if expected != available:
                logger.warning(f"Key alignment failure for {qkey}. Expected retrieval modes {expected}, but only found {available}. Skipping sample configuration.")
                results.append({"question_key": qkey, "best_mode": "MISSING_MODES", "skipped": True})
                # Edge-case saving on last element even when skipped
                if idx == len(pending) - 1:
                    ckpt.save(results)
                continue

            group = group.set_index("mode")
            ref, question, hadm_id = str(group.iloc[0]["answer"]), str(group.iloc[0]["question"]), group.iloc[0]["hadm_id"]

            feature_source = group.iloc[0]

            n_labs = int(feature_source.get("n_labs", 0))
            n_diag = int(feature_source.get("n_diag", 0))
            n_meds = int(feature_source.get("n_meds", 0))
            sparsity_score = float(feature_source.get("sparsity_score", 0.0))
            sparsity_bucket = str(feature_source.get("sparsity_bucket", "unknown"))

            mode_answers, mode_latencies, halluc_flags, halluc_reasons, prompts_used = {}, {}, {}, {}, {}
            generation_failed = False

            try:
                for mode in cfg.modes:
                    ctx = str(group.loc[mode]["prompt_context"])
                    ans, ms, tokens = components["generator"].generate(ctx, question)
                    
                    # Empty Generation Early Warning Deflector Fix
                    if not ans:
                        logger.warning(f"Generation completely failed or timed out for execution window: {qkey} [{mode}]. Deflecting evaluation matrix paths.")
                        generation_failed = True
                        break

                    prompts_used[mode] = ctx
                    mode_answers[mode] = ans
                    mode_latencies[mode] = ms
                    total_tokens_generated += tokens

                    flag, reasons = components["detector"].detect(ans, ctx)
                    halluc_flags[mode], halluc_reasons[mode] = flag, reasons

                if generation_failed:
                    results.append({
                        "question_key": qkey,
                        "hadm_id": hadm_id,
                        "question": question,
                        "reference_answer": ref,
                        "best_mode": "FAILED_GENERATION",
                        "skipped": True,
                        "n_labs": n_labs,
                        "n_diag": n_diag,
                        "n_meds": n_meds,
                        "sparsity_score": sparsity_score,
                        "sparsity_bucket": sparsity_bucket,
                    })
                    if (idx + 1) % cfg.checkpoint_interval == 0 or idx == len(pending) - 1:
                        ckpt.save(results)
                    continue

                raw_scores = components["evaluator"].score_batch([mode_answers[m] for m in cfg.modes], ref)
                composite_scores = {mode: components["evaluator"].composite_score(sc["bert_f1"], sc["rouge_l"], sc["em"], halluc_flags[mode]) 
                                    for mode, sc in zip(cfg.modes, raw_scores)}
                
                best_mode = components["selector"].select(composite_scores)

                row_dict = {
                    "question_key": qkey,
                    "hadm_id": hadm_id,
                    "question": question,
                    "reference_answer": ref,
                    "best_mode": best_mode,
                    "skipped": False,
                    "n_labs": n_labs,
                    "n_diag": n_diag,
                    "n_meds": n_meds,
                    "sparsity_score": sparsity_score,
                    "sparsity_bucket": sparsity_bucket,
                }
                
                for j, mode in enumerate(cfg.modes):
                    mk = mode.replace("+", "").lower()
                    row_dict.update({
                        f"prompt_{mk}": prompts_used[mode],
                        f"answer_{mk}": mode_answers[mode], f"composite_{mk}": composite_scores[mode],
                        f"bert_{mk}": raw_scores[j]["bert_f1"], f"rouge_{mk}": raw_scores[j]["rouge_l"], f"em_{mk}": raw_scores[j]["em"],
                        f"halluc_{mk}": halluc_flags[mode], f"latency_{mk}": mode_latencies[mode]
                    })
                results.append(row_dict)

            except Exception as exc:
                logger.error(f"Unrecoverable runtime error on group execution sequence {qkey}: {exc}")
                results.append({
                    "question_key": qkey,
                    "best_mode": "FAILED",
                    "skipped": True,
                    "n_labs": n_labs,
                    "n_diag": n_diag,
                    "n_meds": n_meds,
                    "sparsity_score": sparsity_score,
                    "sparsity_bucket": sparsity_bucket,
                })

            finally:
                components["generator"].cleanup()

            # Boundary saving logic improvement (saves last index explicitly)
            if (idx + 1) % cfg.checkpoint_interval == 0 or idx == len(pending) - 1:
                ckpt.save(results)

    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by User. Saving progress before safe system shutdown...")
        ckpt.save(results)
        sys.exit(1)

    pbar.close()
    pd.DataFrame(results).to_parquet(output_path, index=False)
    ckpt.cleanup()
    
    components["reporter"].generate(split_name, results, start_time, cfg, total_tokens_generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    # Fixed standard non-deterministic interpretation type=bool bug via string validator helper
    parser.add_argument("--use-cache", type=str2bool, default=True)
    args = parser.parse_args()

    cfg = OracleConfig(checkpoint_interval=args.checkpoint_interval, use_cache=args.use_cache)

    # Global Deterministic Multi-GPU Seeding Sequence
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    log_dir = Path("experiments/logs")
    logger_main = setup_logging(log_dir, "main")
    model, tokenizer = ModelLoader(cfg, logger_main).load()

    components = {
        "logger": logger_main,
        "generator": AnswerGenerator(model, tokenizer, cfg, logger_main),
        "evaluator": MetricEvaluator(cfg, logger_main),
        "detector": HallucinationDetector(),
        "selector": OracleSelector(cfg),
        "reporter": ReportGenerator(cfg, logger_main)
    }

    splits = [
        ("router_val", cfg.router_val, str(Path(cfg.output_dir) / "router_val_oracle.parquet")),
        ("router_train", cfg.router_train, str(Path(cfg.output_dir) / "router_train_oracle.parquet")),
    ]

    for split_name, in_path, out_path in splits:
        split_logger = setup_logging(log_dir, split_name)
        
        # Fixed Logger Reference Sync Mismatch Bug
        components["logger"] = split_logger
        components["generator"].logger = split_logger
        components["evaluator"].logger = split_logger
        components["reporter"].logger = split_logger
        
        process_split(split_name, in_path, out_path, cfg, components)

if __name__ == "__main__":
    main()