"""
src/evaluation/recompute_grounding.py — length-controlled grounding metric.

WHY
===
`run_evaluation.unsupported_score()` is
    |answer_words - context_words| / |answer_words|
which falls mechanically as the context grows: a longer context covers more
vocabulary, so ANY system that retrieves less scores worse regardless of
whether its answer is actually ungrounded. analysis.py measures this
confound directly — pooled Pearson r = -0.52 between prompt length and
unsupported_rate (r^2 = 0.27, p = 8e-144). Roughly a quarter of the metric's
variance is explained by length alone.

That bias is why the router failed H1 criterion 3: it routes ~25% of
questions to T (short context) and inherits the penalty.

THE FIX
=======
Compare each answer against its REAL context and against a LENGTH-MATCHED
DECOY context drawn from a different question, then report the excess:

    grounding_excess = unsupported(answer, decoy) - unsupported(answer, real)

Both terms carry the same length bias, so it cancels. The residual measures
what we actually care about: how much better the real context explains the
answer than an unrelated context of the same size. Higher = better grounded.

`unsupported_vs_decoy` is also reported so the raw numbers stay inspectable.
The original metric is recomputed too, as a reproduction check that these
rebuilt contexts match the ones used at evaluation time.

Contexts must be rebuilt because run_evaluation.py does not persist
prompt_context. This is retrieval-only — no LLM is loaded.

Usage:
    python -m src.evaluation.recompute_grounding
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.retrieval.retriever import Retriever, Mode as RMode
from src.model.context_budget import build_budgeted_context

RESULTS = Path("experiments/results/final_eval/per_question_results.csv")
OUT_DIR = Path("experiments/results/final_eval")
SEED = 42

COMMON = {"the", "a", "an", "is", "was", "are", "for", "in", "of",
          "to", "and", "or", "this", "that", "with", "has", "have"}


def unsupported(answer: str, context: str) -> float:
    """Identical formula to run_evaluation.unsupported_score()."""
    if not context or not answer:
        return 0.0
    aw = set(str(answer).lower().split()) - COMMON
    if not aw:
        return 0.0
    cw = set(str(context).lower().split())
    return len(aw - cw) / len(aw)


def main() -> None:
    df = pd.read_csv(RESULTS)
    print(f"[INFO] {len(df)} rows / {df.q_idx.nunique()} questions")

    try:
        import src.mkg.retrieval as kg
    except Exception as e:
        print(f"[WARN] KG module unavailable ({e}); T+E+K contexts will lack KG facts.")
        kg = None

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/medgemma-1.5-4b-it")
    r = Retriever(kg_module=kg)

    # Rebuild each (hadm_id, question, mode) context once.
    need = df[["hadm_id", "question", "mode_used"]].drop_duplicates()
    print(f"[INFO] Rebuilding {len(need)} unique contexts (retrieval only)...")
    mode_map = {"T": RMode.T, "T+E": RMode.TE, "T+E+K": RMode.TEK}
    ctx_map: dict[tuple, str] = {}
    for i, (_, row) in enumerate(need.iterrows(), 1):
        key = (row["hadm_id"], row["question"], row["mode_used"])
        try:
            res = r.retrieve(question=str(row["question"]),
                             hadm_id=int(row["hadm_id"]),
                             mode=mode_map[row["mode_used"]])
            ctx_map[key] = build_budgeted_context(res.prompt_context, tok)
        except Exception as e:
            print(f"[WARN] retrieval failed for {key}: {e}")
            ctx_map[key] = ""
        if i % 100 == 0:
            print(f"       {i}/{len(need)}")
    r.close()

    df["_ctx"] = [ctx_map.get((h, q, m), "")
                  for h, q, m in zip(df.hadm_id, df.question, df.mode_used)]

    # Length-matched decoy: for each row, a context of similar token length
    # belonging to a DIFFERENT question.
    rng = np.random.RandomState(SEED)
    pool = df[["q_idx", "_ctx", "prompt_tokens"]].drop_duplicates(subset=["q_idx", "_ctx"])
    pool = pool[pool["_ctx"].str.len() > 0].reset_index(drop=True)
    pool_len = pool["prompt_tokens"].to_numpy()

    decoys = []
    for qi, plen in zip(df.q_idx, df.prompt_tokens):
        cand = pool.index[(pool["q_idx"] != qi)].to_numpy()
        if len(cand) == 0:
            decoys.append("")
            continue
        # nearest 25 by length, then pick one at random -> length matched
        order = cand[np.argsort(np.abs(pool_len[cand] - plen))][:25]
        decoys.append(pool.loc[rng.choice(order), "_ctx"])
    df["_decoy"] = decoys

    df["unsupported_recomputed"] = [unsupported(a, c) for a, c in zip(df.predicted_answer, df._ctx)]
    df["unsupported_vs_decoy"]   = [unsupported(a, c) for a, c in zip(df.predicted_answer, df._decoy)]
    df["grounding_excess"]       = df["unsupported_vs_decoy"] - df["unsupported_recomputed"]

    agg = df.groupby("system").agg(
        unsupported_original=("unsupported_rate", "mean"),
        unsupported_recomputed=("unsupported_recomputed", "mean"),
        unsupported_vs_decoy=("unsupported_vs_decoy", "mean"),
        grounding_excess=("grounding_excess", "mean"),
        mean_prompt_tokens=("prompt_tokens", "mean"),
    ).round(4)
    order = ["T", "T+E", "T+E+K", "Router", "Random", "StaticQType", "Oracle"]
    agg = agg.reindex([s for s in order if s in agg.index])

    from scipy import stats
    piv = df.pivot_table(index="q_idx", columns="system", values="grounding_excess")
    print("\n" + "=" * 92)
    print("LENGTH-CONTROLLED GROUNDING  (grounding_excess = decoy - real; HIGHER is better)")
    print("=" * 92)
    print(agg.to_string())

    if "Router" in piv.columns and "T+E+K" in piv.columns:
        a, b = piv["Router"], piv["T+E+K"]
        _, p = stats.wilcoxon(a, b)
        print(f"\n  Router vs T+E+K grounding_excess: {a.mean():.4f} vs {b.mean():.4f} "
              f"(diff {a.mean()-b.mean():+.4f}, wilcoxon p={p:.4g})")
        print(f"  ORIGINAL metric said: {df[df.system=='Router'].unsupported_rate.mean():.4f} vs "
              f"{df[df.system=='T+E+K'].unsupported_rate.mean():.4f} (router WORSE)")

    r_orig = stats.pearsonr(df.prompt_tokens, df.unsupported_recomputed)[0]
    r_new  = stats.pearsonr(df.prompt_tokens, df.grounding_excess)[0]
    print(f"\n  length confound  original r={r_orig:+.4f}   length-controlled r={r_new:+.4f}")

    agg.to_csv(OUT_DIR / "grounding_length_controlled.csv")
    df.drop(columns=["_ctx", "_decoy"]).to_csv(
        OUT_DIR / "per_question_grounding.csv", index=False)
    print(f"\n[INFO] Saved grounding_length_controlled.csv and per_question_grounding.csv")


if __name__ == "__main__":
    main()
