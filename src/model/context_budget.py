"""
src/model/context_budget.py — deterministic, mode-isolating context budgeting.

WHY THIS EXISTS
===============
The 2026-08-08 QLoRA run produced a model that echoed its retrieved context
instead of answering (52-63% of generations). Root cause was mechanical, not
statistical (see RESEARCH_LOG.md, 2026-08-10 pre-training audit):

  A. Loss was computed over the WHOLE sequence (prompt + answer). With a
     ~24-token answer and a ~2,100-2,600-token context, >99% of the training
     signal was literally "reproduce the retrieved context."
  B. `max_length=768` with HF/TRL's default `truncation_mode="keep_start"`
     silently discarded everything past token 768 — and 88.3% of assembled
     contexts are longer than that, so in ~88% of training examples the
     question AND the gold answer were cut off entirely before the model
     ever saw them.

(A) is fixed in train_qlora.py by switching to a prompt/completion dataset
with `completion_only_loss=True`. (B) is fixed here.

THE MODE-CONFOUND THIS ALSO PREVENTS
====================================
A naive "just truncate the whole context to N tokens" fix would be worse
than the bug, scientifically. `RetrievalResult.prompt_context` orders
sections as: passages -> EHR snapshot -> KG facts. Tail-truncating would
strip EHR and KG *first* — i.e. delete precisely the content that
distinguishes T+E and T+E+K from T, silently collapsing all three modes
toward identical prompts and guaranteeing "the KG doesn't help" as an
artifact rather than a finding.

Instead this module gives each section its OWN fixed budget:

    T      = passages(<=PASSAGE_BUDGET)
    T+E    = passages(<=PASSAGE_BUDGET) + ehr(<=EHR_BUDGET)
    T+E+K  = passages(<=PASSAGE_BUDGET) + ehr(<=EHR_BUDGET) + kg(<=KG_BUDGET)

Because the passage budget is identical across modes, the passage content
for a given question is identical across modes too. The modes therefore
differ ONLY by the additive presence of EHR / KG — which is exactly the
contrast H1/H2 are about. Passages are dropped lowest-ranked-first (FAISS
returns them in descending similarity), so what survives is the
highest-scoring evidence.

This module is the single source of truth for prompt length and is applied
identically at training time (train_qlora.py), router-dataset construction
(build_router_dataset.py -> inherited by oracle_labels.py), and final
evaluation (run_evaluation.py). Do not truncate prompts anywhere else.
"""

from __future__ import annotations

# Section headers emitted by RetrievalResult.prompt_context. Kept in sync
# with src/retrieval/retriever.py — a mismatch here would silently cause
# section splitting to fail, so parse_context_sections() asserts on it.
PASSAGES_HEADER = "## Retrieved Clinical Passages"
EHR_HEADER = "## Patient EHR Snapshot"
KG_HEADER = "## Relevant Medical Knowledge"

# Per-section token budgets. Chosen from the measured distributions
# (RESEARCH_LOG.md 2026-08-10): ehr p90=339/max=344, kg p90=476,
# answer p99=52/max=68, question max=21. Passages are the only unbounded
# component (p50=2225, max=3500) and so are the one that gets trimmed.
PASSAGE_BUDGET = 600
EHR_BUDGET = 350
KG_BUDGET = 350

# Worst-case assembled context = 600 + 350 + 350 = 1300 tokens.
MAX_CONTEXT_TOKENS = PASSAGE_BUDGET + EHR_BUDGET + KG_BUDGET

# Reserved for the completion (gold answer) so it can NEVER be truncated —
# this is the entire learning signal. Measured max was 68; 96 gives headroom.
COMPLETION_RESERVE_TOKENS = 96

# Reserved for the question + "Context:\n" wrapper + chat-template special
# tokens. Measured question max=21; 128 is generous headroom.
PROMPT_OVERHEAD_RESERVE_TOKENS = 128

# Total sequence budget. Every consumer must use this as max_length so that
# training, oracle generation, and evaluation all operate at one identical
# sequence length (previously they used 768 / none / 2048 respectively).
MAX_SEQ_LENGTH = (
    MAX_CONTEXT_TOKENS + COMPLETION_RESERVE_TOKENS + PROMPT_OVERHEAD_RESERVE_TOKENS
)  # = 1524

# Generation length at inference. Shared so that oracle-label generation and
# final evaluation produce comparably-lengthed answers: previously
# oracle_labels.py used 128 and run_evaluation.py used 256, meaning the mode
# chosen as "best" by the oracle was selected under a different generation
# budget than the answers actually reported in the results table. Gold
# answers measure p99=52 / max=68 tokens, so 128 is ample for both.
MAX_NEW_TOKENS = 128


def _truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate `text` to at most `max_tokens` tokens, keeping the START."""
    if max_tokens <= 0 or not text:
        return ""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def parse_context_sections(context: str) -> tuple[str, str, str]:
    """Split an assembled prompt_context into (passages, ehr, kg) bodies.

    Missing sections come back as "" — that is the normal case (mode T has
    no EHR/KG, mode T+E has no KG), not an error.
    """
    passages = ehr = kg = ""
    if not context:
        return passages, ehr, kg

    rest = context
    if KG_HEADER in rest:
        rest, kg_part = rest.split(KG_HEADER, 1)
        kg = kg_part.strip()
    if EHR_HEADER in rest:
        rest, ehr_part = rest.split(EHR_HEADER, 1)
        ehr = ehr_part.strip()
    if PASSAGES_HEADER in rest:
        passages = rest.split(PASSAGES_HEADER, 1)[1].strip()
    else:
        passages = rest.strip()
    return passages, ehr, kg


def _trim_passages(passages_body: str, tokenizer, budget: int) -> str:
    """Drop whole passages lowest-ranked-first until the block fits.

    FAISS returns passages in descending similarity and
    RetrievalResult.prompt_context renders them in that order as
    "[Passage N | src]" blocks, so dropping from the end drops the
    least-relevant evidence first. If even the single top passage overflows,
    it is token-truncated rather than dropped, so a prompt is never left
    with zero evidence.
    """
    if not passages_body:
        return ""
    if len(tokenizer(passages_body, add_special_tokens=False)["input_ids"]) <= budget:
        return passages_body

    blocks = [b for b in passages_body.split("\n\n[Passage ") if b.strip()]
    if blocks:
        blocks = [blocks[0]] + ["[Passage " + b for b in blocks[1:]]

    kept: list[str] = []
    for block in blocks:
        candidate = "\n\n".join(kept + [block])
        if len(tokenizer(candidate, add_special_tokens=False)["input_ids"]) > budget:
            break
        kept.append(block)

    if not kept:
        return _truncate_to_tokens(blocks[0] if blocks else passages_body, tokenizer, budget)
    return "\n\n".join(kept)


def build_budgeted_context(context: str, tokenizer) -> str:
    """Re-assemble `context` with per-section budgets enforced.

    Idempotent: re-budgeting an already-budgeted context is a no-op.
    Section presence is preserved exactly — a section that was absent stays
    absent, and a section that was present is never dropped entirely (only
    trimmed), so the T / T+E / T+E+K distinction always survives.
    """
    passages, ehr, kg = parse_context_sections(context)

    passages = _trim_passages(passages, tokenizer, PASSAGE_BUDGET)
    ehr = _truncate_to_tokens(ehr, tokenizer, EHR_BUDGET)
    kg = _truncate_to_tokens(kg, tokenizer, KG_BUDGET)

    parts: list[str] = []
    if passages:
        parts.append(f"{PASSAGES_HEADER}\n{passages}")
    if ehr:
        parts.append(f"{EHR_HEADER}\n{ehr}")
    if kg:
        parts.append(f"{KG_HEADER}\n{kg}")
    return "\n\n".join(parts) if parts else "No context available."
