"""
src/model/prompts.py — single source of truth for the user-message format
sent to MedGemma, at fine-tuning time, oracle-label-generation time, and
final-evaluation time.

Before this module existed, three call sites (train_qlora.py, oracle_labels.py,
run_evaluation.py) each built this string independently and had drifted:
run_evaluation.py used "Context:\n{context}\n\nQuestion: {question}" while
oracle_labels.py used "{context}\n\nQuestion: {question}" (no "Context:\n"
prefix). That mismatch meant the composite scores used to pick oracle router
labels were computed from a subtly different prompt than the one used to
generate the final reported answers. See RESEARCH_LOG.md, 2026-08-06 audit.

Import build_user_message() everywhere a prompt is built for MedGemma instead
of formatting the string inline.
"""

from __future__ import annotations


def build_user_message(context: str, question: str) -> str:
    """The single canonical user-turn content sent to MedGemma."""
    return f"Context:\n{context}\n\nQuestion: {question}"
