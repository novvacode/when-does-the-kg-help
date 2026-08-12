"""
src/evaluation/analysis.py — post-hoc statistical analysis of the held-out run.

Reads experiments/results/final_eval/per_question_results.csv (produced by
run_evaluation.py) and adds three things the raw results table lacks:

  1. BOOTSTRAP CONFIDENCE INTERVALS. The main table reports point estimates
     with no uncertainty, which is not defensible for a paper. Resampling is
     PAIRED over questions (all systems answer the same 300 questions), so
     between-system comparisons keep their pairing and the CIs on differences
     are correct.

  2. LATENCY DECOMPOSITION into retrieval vs generation. The headline
     "T+E+K is 7528 ms" conflates a ~4.3 s Neo4j round-trip with model
     generation; a reviewer needs those separated to judge where the
     router's saving actually comes from.

  3. CONTEXT-LENGTH CONFOUND DIAGNOSTIC for unsupported_rate. That metric is
     |answer_words - context_words| / |answer_words|, which falls
     mechanically as context grows — so it penalises any system that
     retrieves less, independently of whether the answer is actually
     ungrounded. This quantifies the confound rather than asserting it.
     The corrected, length-controlled metric is computed separately by
     recompute_grounding.py (it needs the context text, which
     run_evaluation.py does not persist).

Usage:
    python -m src.evaluation.analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

RESULTS = Path("experiments/results/final_eval/per_question_results.csv")
OUT_DIR = Path("experiments/results/final_eval")
N_BOOT = 10000
SEED = 42
SYSTEMS = ["T", "T+E", "T+E+K", "Router", "Random", "StaticQType", "Oracle"]
METRICS = ["bleu", "rouge_l", "bertscore_f1", "ehr_contradiction",
           "unsupported_rate", "total_latency_ms"]


def _pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """questions x systems matrix, so bootstrap resampling stays paired."""
    return df.pivot_table(index="q_idx", columns="system", values=metric)


def bootstrap_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile bootstrap CIs for each system x metric, paired over questions."""
    rng = np.random.RandomState(SEED)
    rows = []
    for metric in METRICS:
        piv = _pivot(df, metric)
        n = len(piv)
        # one index draw per replicate, shared across systems -> paired
        idx = rng.randint(0, n, size=(N_BOOT, n))
        for sysname in SYSTEMS:
            if sysname not in piv.columns:
                continue
            vals = piv[sysname].to_numpy()
            boots = vals[idx].mean(axis=1)
            rows.append({
                "metric": metric,
                "system": sysname,
                "mean": vals.mean(),
                "ci_lo": np.percentile(boots, 2.5),
                "ci_hi": np.percentile(boots, 97.5),
            })
    return pd.DataFrame(rows)


def paired_differences(df: pd.DataFrame, ref: str = "T+E+K",
                       focus: str = "Router") -> pd.DataFrame:
    """CI and Wilcoxon p for focus - ref on the SAME questions.

    This is the statistic H1 actually rests on: whether the router is
    distinguishable from always-on hybrid, not whether their separate CIs
    happen to overlap (which is a weaker and often misleading test).
    """
    rng = np.random.RandomState(SEED)
    rows = []
    for metric in METRICS:
        piv = _pivot(df, metric)
        if focus not in piv.columns or ref not in piv.columns:
            continue
        d = (piv[focus] - piv[ref]).to_numpy()
        n = len(d)
        idx = rng.randint(0, n, size=(N_BOOT, n))
        boots = d[idx].mean(axis=1)
        try:
            _, p = stats.wilcoxon(piv[focus], piv[ref])
        except ValueError:      # all differences zero
            p = 1.0
        rows.append({
            "metric": metric,
            "comparison": f"{focus} - {ref}",
            "diff": d.mean(),
            "ci_lo": np.percentile(boots, 2.5),
            "ci_hi": np.percentile(boots, 97.5),
            "wilcoxon_p": p,
            "significant_at_.05": p < 0.05,
        })
    return pd.DataFrame(rows)


def latency_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Split total latency into retrieval and generation components."""
    d = df.copy()
    d["generation_latency_ms"] = d["total_latency_ms"] - d["retrieval_latency_ms"]
    agg = d.groupby("system").agg(
        retrieval_ms=("retrieval_latency_ms", "mean"),
        generation_ms=("generation_latency_ms", "mean"),
        total_ms=("total_latency_ms", "mean"),
        p95_total_ms=("total_latency_ms", lambda s: s.quantile(0.95)),
    ).reindex(SYSTEMS).round(1)
    agg["retrieval_pct"] = (100 * agg["retrieval_ms"] / agg["total_ms"]).round(1)
    return agg.reset_index()


def unsupported_confound(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify how much unsupported_rate is explained by context length alone.

    If the metric were measuring grounding, it should not be strongly
    predictable from prompt length. A large negative correlation means the
    metric mostly rewards having more context, which is exactly the bias
    that made the router fail H1 criterion 3.
    """
    rows = []
    for sysname in SYSTEMS:
        s = df[df["system"] == sysname]
        if len(s) < 3:
            continue
        r, p = stats.pearsonr(s["prompt_tokens"], s["unsupported_rate"])
        rows.append({"system": sysname, "n": len(s),
                     "pearson_r_len_vs_unsupported": round(r, 4),
                     "p": p, "r_squared": round(r * r, 4)})
    overall_r, overall_p = stats.pearsonr(df["prompt_tokens"], df["unsupported_rate"])
    rows.append({"system": "ALL POOLED", "n": len(df),
                 "pearson_r_len_vs_unsupported": round(overall_r, 4),
                 "p": overall_p, "r_squared": round(overall_r ** 2, 4)})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(RESULTS)
    print(f"[INFO] Loaded {len(df)} rows / {df.q_idx.nunique()} questions / "
          f"{df.system.nunique()} systems")

    ci = bootstrap_ci(df)
    ci.to_csv(OUT_DIR / "bootstrap_ci.csv", index=False)
    print("\n" + "=" * 78)
    print(f"1. BOOTSTRAP 95% CIs  ({N_BOOT} paired resamples over questions)")
    print("=" * 78)
    for metric in ["bertscore_f1", "bleu", "rouge_l", "unsupported_rate"]:
        sub = ci[ci.metric == metric]
        print(f"\n  {metric}")
        for _, r in sub.iterrows():
            print(f"    {r['system']:12s} {r['mean']:.4f}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")

    pd_ = paired_differences(df)
    pd_.to_csv(OUT_DIR / "paired_differences_router_vs_tek.csv", index=False)
    print("\n" + "=" * 78)
    print("   PAIRED DIFFERENCES: Router - T+E+K (same 300 questions)")
    print("=" * 78)
    for _, r in pd_.iterrows():
        flag = "SIGNIFICANT" if r["significant_at_.05"] else "n.s."
        print(f"  {r['metric']:20s} {r['diff']:+.4f}  "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  p={r['wilcoxon_p']:.3g}  {flag}")

    lat = latency_decomposition(df)
    lat.to_csv(OUT_DIR / "latency_decomposition.csv", index=False)
    print("\n" + "=" * 78)
    print("2. LATENCY DECOMPOSITION (retrieval vs generation)")
    print("=" * 78)
    print(lat.to_string(index=False))

    conf = unsupported_confound(df)
    conf.to_csv(OUT_DIR / "unsupported_length_confound.csv", index=False)
    print("\n" + "=" * 78)
    print("3. unsupported_rate vs CONTEXT LENGTH  (confound diagnostic)")
    print("=" * 78)
    print(conf.to_string(index=False))
    print("\n  A strong negative r means the metric largely rewards longer")
    print("  context rather than measuring grounding. See recompute_grounding.py")
    print("  for the length-controlled replacement.")

    print(f"\n[INFO] Written to {OUT_DIR}/: bootstrap_ci.csv, "
          "paired_differences_router_vs_tek.csv, latency_decomposition.csv, "
          "unsupported_length_confound.csv")


if __name__ == "__main__":
    main()
