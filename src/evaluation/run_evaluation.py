"""
src/evaluation/run_evaluation.py
=================================
Phase 6 — Final Held-Out Evaluation.

Evaluates all four systems on ehrqa_eval.parquet (NEVER touched before):
    1. T          — Text-only RAG
    2. T+E        — Text + EHR
    3. T+E+K      — Always-on hybrid (KG every time)
    4. Router     — Adaptive routing (our proposed system)
    5. Random     — Random mode selection (lower bound baseline)

Metrics computed:
    Text Quality : BLEU, ROUGE-L, BERTScore F1
    Hallucination: EHR-contradiction rate, unsupported rate
    Efficiency   : latency (ms), prompt tokens, VRAM (MB)

Outputs (all in experiments/results/final_eval/):
    summary_table.csv          <- main results table (for paper Table 2)
    per_question_results.csv   <- full per-question breakdown
    sparsity_breakdown.csv     <- H2 analysis: results by sparsity bucket
    qtype_breakdown.csv        <- results by question type
    hallucination_report.csv  <- hallucination analysis per system
    efficiency_report.csv     <- latency / token cost analysis
    figures/                   <- all plots

Usage:
    python -m src.evaluation.run_evaluation
    python -m src.evaluation.run_evaluation --max-samples 30   # quick test
    python -m src.evaluation.run_evaluation --skip-bertscore   # faster run
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import time
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# IMPORTANT:
# Import the stable module that owns the pickled RouterConfig and HybridFeaturePipeline
# classes before calling pickle.load(). This prevents __main__-namespace deserialization
# failures when train_router.py was executed directly during artifact creation.
from src.router.feature_pipeline import HybridFeaturePipeline, RouterConfig  # noqa: F401
from src.lakehouse.patient_snapshot import PatientSnapshot
from src.model.prompts import build_user_message

warnings.filterwarnings("ignore")


# -- Paths --------------------------------------------------------------------
EVAL_QA_FILE   = Path("data/lakehouse/qa/ehrqa_eval.parquet")
ROUTER_DIR     = Path("data/router")
ROUTER_MODEL   = Path("models/router/router_xgb_model.json")
LABEL_ENC      = Path("models/router/label_encoder.pkl")
FEAT_PIPELINE  = Path("models/router/feature_pipeline.pkl")
GEN_MODEL_BASE = "google/medgemma-1.5-4b-it"
GEN_ADAPTER    = Path("models/medgemma-4b-qlora")
OUT_DIR        = Path("experiments/results/final_eval")

# Below this many questions, a run cannot support the paper's held-out claims.
# A prior "quick test" run with --max-samples 1 was accidentally written into
# this directory and mistaken for the final result (RESEARCH_LOG.md,
# 2026-08-06 audit, finding #1) — this guard makes that mistake loud instead
# of silent.
MIN_FINAL_EVAL_SAMPLES = 50


MODES           = ["T", "T+E", "T+E+K"]
SYSTEMS         = ["T", "T+E", "T+E+K", "Router", "Random"]
SEED            = 42
MAX_NEW_TOKENS  = 256
BERTSCORE_MODEL = "distilbert-base-uncased"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger("run_evaluation")


random.seed(SEED)
np.random.seed(SEED)


# -- Generator loader -----------------------------------------------------------


def load_generator():
    print(f"[INFO] Loading generator: {GEN_MODEL_BASE}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(GEN_ADAPTER)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_BASE,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, str(GEN_ADAPTER))
    model.eval()
    print("[INFO] Generator ready.")
    return model, tok


# -- Router loader --------------------------------------------------------------


def load_router():
    """Load XGBoost router + label encoder + feature pipeline."""
    import xgboost as xgb

    if not ROUTER_MODEL.exists():
        raise FileNotFoundError(f"Router model not found: {ROUTER_MODEL}")

    clf = xgb.XGBClassifier()
    clf.load_model(str(ROUTER_MODEL))

    with open(LABEL_ENC, "rb") as f:
        le = pickle.load(f)

    feat_pipeline = None
    if FEAT_PIPELINE.exists():
        with open(FEAT_PIPELINE, "rb") as f:
            feat_pipeline = pickle.load(f)

    print(f"[INFO] Router loaded. Classes: {list(le.classes_)}")
    return clf, le, feat_pipeline


# -- Retrieval helper ------------------------------------------------------------

def load_kg_module():
    """Attempt to dynamically load the Neo4j MKG client."""
    try:
        import src.mkg.retrieval as kg_module
        print("[INFO] Neo4j KG module loaded successfully.")
        return kg_module
    except Exception as e:
        print(f"[WARN] Could not initialize KG module: {e}")
        print("[WARN] Proceeding with KG component disabled.")
        return None

def get_retrieval_context(question: str, hadm_id: int, mode: str) -> dict:
    """
    Call the unified retriever for a given mode.
    Returns dict with prompt_context, latency_ms, n_kg_facts, prompt_tokens.
    """
    try:
        from src.retrieval.retriever import Retriever, Mode as RMode
        _retriever_cache = getattr(get_retrieval_context, "_cache", None)
        if _retriever_cache is None:
            kg_module = load_kg_module()
            get_retrieval_context._cache = Retriever(kg_module=kg_module)
            _retriever_cache = get_retrieval_context._cache

        mode_map = {"T": RMode.T, "T+E": RMode.TE, "T+E+K": RMode.TEK}
        result = _retriever_cache.retrieve(
            question=question, hadm_id=hadm_id, mode=mode_map[mode]
        )

        stats = getattr(result, "stats", {})
        n_kg_facts = stats.get("n_kg_facts", len(getattr(result, "kg_facts", [])))

        return {
            "prompt_context": result.prompt_context,
            "latency_ms": result.latency_ms,
            "n_kg_facts": n_kg_facts,
            "prompt_tokens": result.n_tokens_approx,
            "has_ehr": bool(result.ehr_snapshot),
        }
    except Exception as e:
        print(f"[WARN] Retrieval failed for hadm_id={hadm_id} mode={mode}: {e}")
        return {
            "prompt_context": f"Question: {question}",
            "latency_ms": 0.0,
            "n_kg_facts": 0,
            "prompt_tokens": len(question.split()) * 2,
            "has_ehr": False,
        }


# -- Router prediction -----------------------------------------------------------


def router_predict(
    question: str,
    hadm_id: int,
    struct_features: dict,
    n_kg_facts: int,
    prompt_tokens_t: int,
    latency_t: float,
    clf,
    le,
    feat_pipeline,
) -> str:
    """
    Build feature vector and predict best mode using real structural patient features.

    Uses plain argmax over the trained classifier's predict_proba output — the
    exact same decision rule train_router.py uses to report Accuracy/F1/the
    confusion matrix. A previous version of this function overrode the
    trained model's decision with hardcoded, never-validated probability
    thresholds (THRESHOLD_TEK=0.30 / THRESHOLD_T=0.35), so the "Router" system
    in past evaluation runs was not actually the model that was trained and
    validated. See RESEARCH_LOG.md, 2026-08-06 audit, finding #2. If routing
    behavior needs calibration, that must happen as a documented, validated
    step against router_val (e.g. temperature scaling or a cost-sensitive
    decision rule fit and reported in train_router.py), not as an
    undocumented override at evaluation time.
    """
    if clf is None or le is None:
        return "T"

    try:
        if feat_pipeline is not None:
            # Feature schema must match HybridFeaturePipeline.cfg.ehr_feature_cols
            # exactly: ["n_labs", "n_diag", "n_meds", "sparsity_score", "sparsity_bucket"].
            feature_row = pd.DataFrame([{
                "question":        question,
                "n_labs":          struct_features.get("n_labs", 0.0),
                "n_diag":          struct_features.get("n_diag", 0.0),
                "n_meds":          struct_features.get("n_meds", 0.0),
                "sparsity_score":  struct_features.get("sparsity_score", 0.0),
                "sparsity_bucket": struct_features.get("sparsity_bucket", "unknown"),
            }])
            X = feat_pipeline.transform(feature_row)
        else:
            fallback_vec = np.array([
                len(question.split()),
                float(n_kg_facts),
                float(prompt_tokens_t),
                float(latency_t),
                float(struct_features.get("sparsity_score", 0.0)),
            ], dtype=np.float32).reshape(1, -1)
            X = fallback_vec

        probs = clf.predict_proba(X)[0]
        classes = list(le.classes_)
        pred_mode = classes[int(np.argmax(probs))]
        return pred_mode

    except Exception as e:
        logger.warning(f"Router prediction failed for hadm_id={hadm_id}: {e}. Using T.")
        return "T"

# -- Generation -------------------------------------------------------------------


@torch.inference_mode()
def generate(model, tok, question: str, context: str) -> tuple[str, float]:
    """Generate answer. Returns (answer_str, latency_ms)."""
    messages = [{"role": "user", "content": build_user_message(context, question)}]

    try:
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    t0 = time.time()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    latency_ms = (time.time() - t0) * 1000

    new_ids = out[0][inputs["input_ids"].shape[1]:]
    answer = tok.decode(new_ids, skip_special_tokens=True).strip()
    return answer, latency_ms


# -- Metrics ------------------------------------------------------------------


def compute_bleu(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    sf = SmoothingFunction().method1
    try:
        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=sf)
    except Exception:
        return 0.0


def compute_rouge_l(prediction: str, reference: str) -> float:
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    try:
        return sc.score(reference, prediction)["rougeL"].fmeasure
    except Exception:
        return 0.0


def compute_bertscore_batch(
    predictions: list[str],
    references: list[str],
    skip: bool = False,
) -> list[float]:
    if skip or not predictions:
        return [0.0] * len(predictions)
    try:
        _, _, F1 = bert_score_fn(
            predictions, references,
            model_type=BERTSCORE_MODEL,
            lang="en",
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return F1.tolist()
    except Exception as e:
        # BERTScore is a primary reported metric — a silent 0.0 fallback here
        # previously produced a "final" results table where every system
        # showed BERTScore-F1=0.0 with no persisted record of why (see
        # RESEARCH_LOG.md, 2026-08-06 audit, finding #8). Log the full
        # traceback to disk and surface it loudly instead of swallowing it.
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "bertscore_failure.log", "w", encoding="utf-8") as f:
            f.write(f"BERTScore failed: {e}\n\n{traceback.format_exc()}")
        logger.error(
            f"BERTScore computation FAILED: {e}. Falling back to 0.0 for all "
            f"{len(predictions)} predictions. Full traceback written to "
            f"{OUT_DIR / 'bertscore_failure.log'}. This metric cannot be "
            f"trusted for this run — fix the underlying issue and re-run."
        )
        return [0.0] * len(predictions)


def ehr_contradiction_score(answer: str, ehr_context: str) -> float:
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


def unsupported_score(answer: str, context: str) -> float:
    if not context or not answer:
        return 0.0
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    common_words = {"the", "a", "an", "is", "was", "are", "for", "in", "of",
                    "to", "and", "or", "this", "that", "with", "has", "have"}
    answer_words -= common_words
    if not answer_words:
        return 0.0
    unsupported = answer_words - context_words
    return len(unsupported) / len(answer_words)


def vram_usage_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


# -- Main evaluation loop ------------------------------------------------------


def run_evaluation(
    max_samples: int | None = None,
    skip_bertscore: bool = False,
    allow_small_sample: bool = False,
) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(exist_ok=True)

    print(f"[INFO] Loading held-out eval set: {EVAL_QA_FILE}")
    df_eval = pd.read_parquet(EVAL_QA_FILE)
    print(f"[INFO] Eval set: {len(df_eval):,} questions")

    if max_samples:
        df_eval = df_eval.head(max_samples)
        print(f"[INFO] Limited to {max_samples} questions for quick test.")

    if len(df_eval) < MIN_FINAL_EVAL_SAMPLES and not allow_small_sample:
        raise ValueError(
            f"Refusing to write to {OUT_DIR} with only {len(df_eval)} question(s) "
            f"(< MIN_FINAL_EVAL_SAMPLES={MIN_FINAL_EVAL_SAMPLES}). This directory "
            "is treated as the paper's final result. If this is intentionally a "
            "quick smoke test, pass --allow-small-sample so it can't be mistaken "
            "for a full run later (see RESEARCH_LOG.md, 2026-08-06 audit, "
            "finding #1)."
        )

    # Structural routing features — sourced via PatientSnapshot so eval-time
    # features (n_labs, n_diag, n_meds, sparsity_score, sparsity_bucket) are
    # computed exactly the way generate_synthetic_ehrqa.py computes them for
    # router training data. A previous version read only n_labs/n_diag/score/
    # bucket from sparsity.parquet directly and never populated n_meds at
    # all (silently always 0.0 at eval time, despite being a real trained
    # feature) — see RESEARCH_LOG.md, 2026-08-06 audit, finding #3.
    snapshot_api = PatientSnapshot()
    struct_feature_cache: dict[int, dict] = {}

    def get_struct_features(hadm_id: int) -> dict:
        if hadm_id not in struct_feature_cache:
            try:
                snap = snapshot_api.get(hadm_id)
                struct_feature_cache[hadm_id] = {
                    "n_labs": float(len(snap.get("labs", []))),
                    "n_diag": float(len(snap.get("diagnoses", []))),
                    "n_meds": float(len(snap.get("medications", []))),
                    "sparsity_score": float(snap.get("sparsity_score") or 0.0),
                    "sparsity_bucket": str(snap.get("sparsity_bucket") or "unknown"),
                }
            except Exception as e:
                logger.warning(f"PatientSnapshot lookup failed for hadm_id={hadm_id}: {e}")
                struct_feature_cache[hadm_id] = {
                    "n_labs": 0.0, "n_diag": 0.0, "n_meds": 0.0,
                    "sparsity_score": 0.0, "sparsity_bucket": "unknown",
                }
        return struct_feature_cache[hadm_id]

    gen_model, gen_tok = load_generator()

    try:
        router_clf, router_le, feat_pipeline = load_router()
    except Exception as e:
        print(f"[WARN] Router not available: {e}. Router system will use T fallback.")
        router_clf = router_le = feat_pipeline = None

    records: list[dict] = []
    t_start = time.time()

    for q_idx, row in df_eval.iterrows():
        question = str(row["question"])
        reference = str(row["answer"])
        hadm_id = int(row["hadm_id"])
        q_type = str(row.get("question_type", "unknown"))
        struct_features = get_struct_features(hadm_id)

        ctx: dict[str, dict] = {}
        for mode in MODES:
            ctx[mode] = get_retrieval_context(question, hadm_id, mode)

        router_mode = router_predict(
            question=question,
            hadm_id=hadm_id,
            struct_features=struct_features,
            n_kg_facts=ctx["T+E+K"]["n_kg_facts"],
            prompt_tokens_t=ctx["T"]["prompt_tokens"],
            latency_t=ctx["T"]["latency_ms"],
            clf=router_clf,
            le=router_le,
            feat_pipeline=feat_pipeline,
        )

        random_mode = random.choice(MODES)

        system_answers: dict[str, str] = {}
        system_latency: dict[str, float] = {}

        for mode in MODES:
            ans, gen_lat = generate(gen_model, gen_tok, question, ctx[mode]["prompt_context"])
            system_answers[mode] = ans
            system_latency[mode] = ctx[mode]["latency_ms"] + gen_lat

        system_answers["Router"] = system_answers[router_mode]
        system_latency["Router"] = system_latency[router_mode]
        system_answers["Random"] = system_answers[random_mode]
        system_latency["Random"] = system_latency[random_mode]

        for system in SYSTEMS:
            mode_used = (router_mode if system == "Router"
                         else random_mode if system == "Random"
                         else system)

            rec = {
                "q_idx": q_idx,
                "hadm_id": hadm_id,
                "question": question,
                "reference": reference,
                "question_type": q_type,
                "sparsity_score": struct_features["sparsity_score"],
                "sparsity_bucket": struct_features["sparsity_bucket"],
                "system": system,
                "mode_used": mode_used,
                "predicted_answer": system_answers[system],
                "retrieval_latency_ms": ctx[mode_used]["latency_ms"],
                "total_latency_ms": system_latency[system],
                "prompt_tokens": ctx[mode_used]["prompt_tokens"],
                "n_kg_facts": ctx[mode_used]["n_kg_facts"],
                "has_ehr": ctx[mode_used]["has_ehr"],
                "vram_mb": vram_usage_mb(),
                "bleu": 0.0,
                "rouge_l": 0.0,
                "bertscore_f1": 0.0,
                "ehr_contradiction": ehr_contradiction_score(
                    system_answers[system],
                    ctx[mode_used]["prompt_context"],
                ),
                "unsupported_rate": unsupported_score(
                    system_answers[system],
                    ctx[mode_used]["prompt_context"],
                ),
            }
            rec["bleu"] = compute_bleu(system_answers[system], reference)
            rec["rouge_l"] = compute_rouge_l(system_answers[system], reference)
            records.append(rec)

        if len(records) % (len(SYSTEMS) * 10) == 0:
            elapsed = time.time() - t_start
            print(f"[INFO] {len(records)//len(SYSTEMS)}/{len(df_eval)} questions | "
                  f"{elapsed:.0f}s elapsed")

    df_results = pd.DataFrame(records)

    if not skip_bertscore:
        print("[INFO] Computing BERTScore (batched)...")
        preds = df_results["predicted_answer"].tolist()
        refs = df_results["reference"].tolist()
        bs = compute_bertscore_batch(preds, refs, skip=skip_bertscore)
        df_results["bertscore_f1"] = bs

    df_results.to_csv(OUT_DIR / "per_question_results.csv", index=False)
    print(f"[INFO] Per-question results saved: {OUT_DIR / 'per_question_results.csv'}")

    snapshot_api.close()
    return df_results


# -- Analysis & reporting --------------------------------------------------------


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system in SYSTEMS:
        sub = df[df["system"] == system]
        rows.append({
            "System": system,
            "BLEU": round(sub["bleu"].mean(), 4),
            "ROUGE-L": round(sub["rouge_l"].mean(), 4),
            "BERTScore-F1": round(sub["bertscore_f1"].mean(), 4),
            "EHR-Contradiction": round(sub["ehr_contradiction"].mean(), 4),
            "Unsupported-Rate": round(sub["unsupported_rate"].mean(), 4),
            "Avg-Latency-ms": round(sub["total_latency_ms"].mean(), 1),
            "Avg-Prompt-Tokens": round(sub["prompt_tokens"].mean(), 0),
            "Avg-KG-Facts": round(sub["n_kg_facts"].mean(), 2),
        })
    return pd.DataFrame(rows)


def build_sparsity_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket in ["high", "medium", "low"]:
        sub_b = df[df["sparsity_bucket"] == bucket]
        for system in SYSTEMS:
            sub = sub_b[sub_b["system"] == system]
            if sub.empty:
                continue
            rows.append({
                "sparsity_bucket": bucket,
                "system": system,
                "n": len(sub),
                "BLEU": round(sub["bleu"].mean(), 4),
                "ROUGE-L": round(sub["rouge_l"].mean(), 4),
                "BERTScore-F1": round(sub["bertscore_f1"].mean(), 4),
                "EHR-Contradiction": round(sub["ehr_contradiction"].mean(), 4),
                "Avg-Latency-ms": round(sub["total_latency_ms"].mean(), 1),
            })
    return pd.DataFrame(rows)


def build_qtype_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qtype in df["question_type"].unique():
        sub_q = df[df["question_type"] == qtype]
        for system in SYSTEMS:
            sub = sub_q[sub_q["system"] == system]
            if sub.empty:
                continue
            rows.append({
                "question_type": qtype,
                "system": system,
                "n": len(sub),
                "BLEU": round(sub["bleu"].mean(), 4),
                "ROUGE-L": round(sub["rouge_l"].mean(), 4),
                "BERTScore-F1": round(sub["bertscore_f1"].mean(), 4),
            })
    return pd.DataFrame(rows)


def build_hallucination_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system in SYSTEMS:
        sub = df[df["system"] == system]
        rows.append({
            "system": system,
            "n": len(sub),
            "mean_ehr_contradiction": round(sub["ehr_contradiction"].mean(), 4),
            "pct_ehr_contradiction": round((sub["ehr_contradiction"] > 0.3).mean() * 100, 2),
            "mean_unsupported": round(sub["unsupported_rate"].mean(), 4),
            "pct_unsupported_high": round((sub["unsupported_rate"] > 0.5).mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def build_efficiency_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system in SYSTEMS:
        sub = df[df["system"] == system]
        rows.append({
            "system": system,
            "avg_latency_ms": round(sub["total_latency_ms"].mean(), 1),
            "median_latency_ms": round(sub["total_latency_ms"].median(), 1),
            "p95_latency_ms": round(sub["total_latency_ms"].quantile(0.95), 1),
            "avg_prompt_tokens": round(sub["prompt_tokens"].mean(), 0),
            "avg_kg_facts": round(sub["n_kg_facts"].mean(), 2),
            "avg_vram_mb": round(sub["vram_mb"].mean(), 1),
        })
    return pd.DataFrame(rows)


# -- Figures ----------------------------------------------------------------------


def plot_summary_bars(summary: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    metrics = ["BLEU", "ROUGE-L", "BERTScore-F1"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(summary["System"], summary[metric], color=colors)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim(0, min(1.0, summary[metric].max() * 1.3))
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, summary[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("System Comparison -- Text Quality Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "summary_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {fig_dir / 'summary_metrics.png'}")


def plot_sparsity_heatmap(sparsity: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"

    for metric in ["BERTScore-F1", "ROUGE-L"]:
        pivot = sparsity.pivot_table(
            index="system", columns="sparsity_bucket", values=metric, aggfunc="mean"
        )
        col_order = [c for c in ["high", "medium", "low"] if c in pivot.columns]
        pivot = pivot[col_order]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="YlGnBu",
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title(f"{metric} by System x EHR Sparsity Bucket\n(H2 Analysis)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("EHR Sparsity Bucket")
        ax.set_ylabel("System")
        plt.tight_layout()
        fname = f"sparsity_heatmap_{metric.replace('-','_').lower()}.png"
        plt.savefig(fig_dir / fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved: {fig_dir / fname}")


def plot_latency_vs_quality(summary: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    for i, (_, row) in enumerate(summary.iterrows()):
        ax.scatter(row["Avg-Latency-ms"], row["BERTScore-F1"],
                   s=200, color=colors[i], zorder=5, label=row["System"])
        ax.annotate(row["System"],
                    (row["Avg-Latency-ms"], row["BERTScore-F1"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax.set_xlabel("Average Total Latency (ms)", fontsize=12)
    ax.set_ylabel("BERTScore F1", fontsize=12)
    ax.set_title("Latency vs Quality Trade-off\n(lower-left = efficient; upper = high quality)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "latency_vs_quality.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {fig_dir / 'latency_vs_quality.png'}")


def plot_hallucination(hallucination: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(hallucination))
    width = 0.35
    bars1 = ax.bar(x - width/2, hallucination["pct_ehr_contradiction"],
                   width, label="EHR Contradiction >30%", color="#C44E52", alpha=0.85)
    bars2 = ax.bar(x + width/2, hallucination["pct_unsupported_high"],
                   width, label="High Unsupported Rate >50%", color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(hallucination["system"])
    ax.set_ylabel("% of Questions")
    ax.set_title("Hallucination Rates by System", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "hallucination_rates.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {fig_dir / 'hallucination_rates.png'}")


def plot_router_mode_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    router_df = df[df["system"] == "Router"]

    if router_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mc = router_df["mode_used"].value_counts()
    axes[0].pie(mc.values, labels=mc.index, autopct="%1.1f%%",
                colors=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("Router Mode Distribution\n(Overall)", fontweight="bold")

    pivot = router_df.groupby(["sparsity_bucket", "mode_used"]).size().unstack(fill_value=0)
    col_order = [c for c in ["T", "T+E", "T+E+K"] if c in pivot.columns]
    pivot = pivot[col_order]
    pivot.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452", "#55A868"], alpha=0.85)
    axes[1].set_title("Router Mode Choices by EHR Sparsity\n(H2 Evidence)", fontweight="bold")
    axes[1].set_xlabel("Sparsity Bucket")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Mode")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "router_mode_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {fig_dir / 'router_mode_distribution.png'}")


# -- Final print --------------------------------------------------------------


def print_final_results(summary: pd.DataFrame, sparsity: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)
    print(summary.to_string(index=False))

    print("\n" + "-" * 70)
    print("H2 ANALYSIS -- BERTScore-F1 by Sparsity Bucket")
    print("-" * 70)
    pivot = sparsity.pivot_table(
        index="system", columns="sparsity_bucket",
        values="BERTScore-F1", aggfunc="mean"
    )
    col_order = [c for c in ["high", "medium", "low"] if c in pivot.columns]
    if col_order:
        print(pivot[col_order].round(4).to_string())

    router_bs = summary.loc[summary["System"] == "Router", "BERTScore-F1"].values
    tek_bs = summary.loc[summary["System"] == "T+E+K", "BERTScore-F1"].values
    random_bs = summary.loc[summary["System"] == "Random", "BERTScore-F1"].values

    print("\n" + "-" * 70)
    print("H1 VERDICT")
    print("-" * 70)
    if len(router_bs) and len(random_bs):
        if router_bs[0] > random_bs[0]:
            print(f"  Router ({router_bs[0]:.4f}) > Random ({random_bs[0]:.4f})")
        else:
            print(f"  Router ({router_bs[0]:.4f}) <= Random ({random_bs[0]:.4f})")

    if len(router_bs) and len(tek_bs):
        lat_router = summary.loc[summary["System"] == "Router", "Avg-Latency-ms"].values[0]
        lat_tek = summary.loc[summary["System"] == "T+E+K", "Avg-Latency-ms"].values[0]
        print(f"  Router latency: {lat_router:.0f}ms vs T+E+K: {lat_tek:.0f}ms")
        if lat_router < lat_tek:
            print(f"  Router is faster than always-on T+E+K")

    print("=" * 70)


# -- Entry point ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final held-out evaluation.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to N questions (e.g. --max-samples 30 for quick test)")
    parser.add_argument("--skip-bertscore", action="store_true",
                        help="Skip BERTScore for faster runs")
    parser.add_argument("--allow-small-sample", action="store_true",
                        help="Allow writing < %d questions to experiments/results/final_eval/ "
                             "(only for intentional smoke tests — never for the real final run)"
                             % MIN_FINAL_EVAL_SAMPLES)
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 6 -- FINAL HELD-OUT EVALUATION")
    print("ehrqa_eval.parquet is being used for the FIRST TIME.")
    print("=" * 70)

    df = run_evaluation(
        max_samples=args.max_samples,
        skip_bertscore=args.skip_bertscore,
        allow_small_sample=args.allow_small_sample,
    )

    summary = build_summary_table(df)
    sparsity = build_sparsity_breakdown(df)
    qtype = build_qtype_breakdown(df)
    hallucination = build_hallucination_report(df)
    efficiency = build_efficiency_report(df)

    summary.to_csv(OUT_DIR / "summary_table.csv", index=False)
    sparsity.to_csv(OUT_DIR / "sparsity_breakdown.csv", index=False)
    qtype.to_csv(OUT_DIR / "qtype_breakdown.csv", index=False)
    hallucination.to_csv(OUT_DIR / "hallucination_report.csv", index=False)
    efficiency.to_csv(OUT_DIR / "efficiency_report.csv", index=False)

    print(f"[INFO] All CSVs saved to: {OUT_DIR}")

    plot_summary_bars(summary, OUT_DIR)
    plot_sparsity_heatmap(sparsity, OUT_DIR)
    plot_latency_vs_quality(summary, OUT_DIR)
    plot_hallucination(hallucination, OUT_DIR)
    plot_router_mode_distribution(df, OUT_DIR)

    print_final_results(summary, sparsity)

    results_json = {
        "summary": summary.to_dict(orient="records"),
        "sparsity": sparsity.to_dict(orient="records"),
        "hallucination": hallucination.to_dict(orient="records"),
        "efficiency": efficiency.to_dict(orient="records"),
    }
    with open(OUT_DIR / "full_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nEvaluation complete. All outputs in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()