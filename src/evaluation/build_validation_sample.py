"""
src/evaluation/build_validation_sample.py — fresh sample to validate the
corrected negation-contradiction scorer.

WHY A SECOND SAMPLE
===================
The 75 rows annotated on 2026-08-16 are what EXPOSED the bug. Reusing them to
show the fix works would be circular — the fix was designed with those failures
in view. This draws a disjoint sample from rows never annotated, so the check
is on data the fix has not seen.

DESIGN (n = 30 real rows + 3 attention checks)
----------------------------------------------
  stratum A  20 rows  unseen diagnoses/primary_diagnosis that the ORIGINAL
                      detector flags. This is where the fix acts: if these are
                      genuinely not contradictions, the corrected scorer must
                      go silent on them.
  stratum B  10 rows  unseen rows the original does NOT flag, weighted to the
                      same two question types. Guards against a "fix" that
                      simply switches everything off.

All rows are mode != T, so the n/a rule never applies and every row is scorable.

The sample is deliberately ENRICHED for originally-flagged rows, so its
prevalence is not the eval set's. Cohen's kappa would be unstable or undefined
here; validation_agreement.py reports a paired McNemar test of original vs
corrected against the human labels instead. This is a targeted diagnostic and
does NOT replace the 75-row kappa.

ATTENTION CHECKS
----------------
If the fix works, the expected result is 33 straight "no" answers — a response
set indistinguishable from an annotator who stopped reading. Three rows are
therefore GENUINE Type-1 contradictions constructed by perturbing a real
answer ("<diagnosis the EHR asserts>" -> "No <term>."). They are recorded with
is_attention_check=True, EXCLUDED from every agreement statistic, and reported
separately as a check that the uniform "no" reflects the data rather than
fatigue. The annotator is not told which rows they are until afterwards.

PRE-REGISTERED, LOCKED BEFORE ANNOTATION
----------------------------------------
  · label rule        : corrected score > 0 -> yes (same > 0 rule as before)
  · human question    : identical wording to session 1, so judgements are
                        comparable across both studies
  · success criterion : the fix passes if <=1 of the 20 stratum-A rows is a
                        genuine contradiction per the human AND the corrected
                        scorer flags <=1 of them, AND all synthetic controls in
                        fix_ehr_contradiction.py pass.

Usage:
    python -m src.evaluation.build_validation_sample      # needs Neo4j running
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.build_annotation_sample import rebuild_contexts
from src.evaluation.fix_ehr_contradiction import (
    negation_contradiction_score, ehr_contradiction_score_v1, _tokens, _ehr_asserts,
)

RESULTS = Path("experiments/results/final_eval/per_question_results.csv")
PRIOR_ANNOT = Path("experiments/results/annotation/annotations_filled.csv")
OUT_DIR = Path("experiments/results/annotation_v2")
SAMPLE_CSV = OUT_DIR / "validation_sample.csv"
HTML_OUT = OUT_DIR / "annotate_v2.html"

N_FLAGGED = 20          # stratum A
N_CONTROL = 10          # stratum B
N_ATTENTION = 3
SEED = 42
DX_TYPES = ["diagnoses", "primary_diagnosis"]


def build_sample(df: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    seen = set(prior.q_idx.astype(str) + "|" + prior.system)
    df = df.copy()
    df["rid"] = df.q_idx.astype(str) + "|" + df.system
    pool = df[(~df.rid.isin(seen)) & (df.mode_used != "T")].copy()
    print(f"[INFO] unseen scorable pool: {len(pool)} rows")

    rng = np.random.default_rng(SEED)

    flagged = pool[(pool.ehr_contradiction > 0) & (pool.question_type.isin(DX_TYPES))]
    control = pool[(pool.ehr_contradiction == 0) & (pool.question_type.isin(DX_TYPES))]
    print(f"[INFO] stratum A candidates (flagged, dx types): {len(flagged)}")
    print(f"[INFO] stratum B candidates (unflagged, dx types): {len(control)}")

    def take(sub: pd.DataFrame, n: int, label: str) -> pd.DataFrame:
        idx = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
        out = sub.iloc[np.sort(idx)].copy()
        out["stratum"] = label
        return out

    sample = pd.concat([take(flagged, N_FLAGGED, "A_originally_flagged"),
                        take(control, N_CONTROL, "B_control")], ignore_index=True)
    sample["is_attention_check"] = False
    return sample


def make_attention_checks(pool_rows: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """
    Build genuine Type-1 contradictions by negating a term the EHR asserts.
    Uses real contexts with a PERTURBED answer; excluded from all statistics.
    """
    checks = []
    for _, r in pool_rows.iterrows():
        key = (r["hadm_id"], r["question"], r["mode_used"])
        budgeted = ctx.get(key, ("", "", 0))[1]
        if not budgeted:
            continue
        ehr_tokens = _tokens(budgeted)
        term = next((t for t in _tokens(str(r["reference"]))
                     if len(t) > 5 and t.isalnum() and _ehr_asserts(ehr_tokens, t)), None)
        if term is None:
            continue
        row = r.copy()
        row["predicted_answer"] = f"No {term}."
        row["stratum"] = "ATTENTION_CHECK"
        row["is_attention_check"] = True
        checks.append(row)
        if len(checks) == N_ATTENTION:
            break
    return pd.DataFrame(checks)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Contradiction validation — round 2</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#15181d;--muted:#5b6472;--line:#dfe3e9;
      --accent:#2b6cb0;--yes:#c53030;--no:#2f855a;}
@media (prefers-color-scheme: dark){:root{--bg:#14171c;--panel:#1c2027;--ink:#e8eaed;
      --muted:#9aa4b2;--line:#2c323b;--accent:#63a4dd;--yes:#f56565;--no:#68d391;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:10;background:var(--panel);
       border-bottom:1px solid var(--line);padding:10px 18px;display:flex;
       gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:650}
.bar{flex:1;min-width:120px;height:8px;background:var(--line);border-radius:99px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent);width:0%}
.count{font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px}
main{max-width:960px;margin:0 auto;padding:18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      padding:16px 18px;margin-bottom:14px}
.meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.tag{font-size:11.5px;padding:2px 8px;border-radius:99px;background:var(--bg);
     border:1px solid var(--line);color:var(--muted)}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:0 0 7px}
.q{font-size:17px;font-weight:600;margin:0}
pre.ctx{white-space:pre-wrap;word-break:break-word;background:var(--bg);
        border:1px solid var(--line);border-radius:8px;padding:12px;max-height:340px;
        overflow:auto;font:12.5px/1.5 ui-monospace,Consolas,monospace;margin:0}
.ans{background:var(--bg);border-left:3px solid var(--accent);padding:10px 12px;
     border-radius:0 8px 8px 0;white-space:pre-wrap}
.gold{background:var(--bg);border-left:3px solid var(--no);padding:10px 12px;
      border-radius:0 8px 8px 0;white-space:pre-wrap}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.qbox{border:1px solid var(--line);border-radius:8px;padding:12px;margin-top:12px}
.qtxt{font-weight:600;margin-bottom:9px}
.hint{color:var(--muted);font-size:12.5px;font-weight:400;margin-top:3px}
.btns{display:flex;gap:8px;flex-wrap:wrap}
button{font:inherit;padding:7px 15px;border-radius:7px;border:1px solid var(--line);
       background:var(--panel);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
button.sel[data-v="yes"]{background:var(--yes);border-color:var(--yes);color:#fff}
button.sel[data-v="no"]{background:var(--no);border-color:var(--no);color:#fff}
button:disabled{opacity:.4;cursor:not-allowed}
textarea{width:100%;min-height:52px;padding:8px;border-radius:7px;border:1px solid var(--line);
         background:var(--bg);color:var(--ink);font:inherit}
nav{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:4px 0 30px}
.rev{margin-top:10px;padding:9px 12px;border:1px dashed var(--line);border-radius:8px;
     font-size:13px;color:var(--muted)}
kbd{font:11px ui-monospace,monospace;background:var(--bg);border:1px solid var(--line);
    border-radius:4px;padding:1px 5px;color:var(--muted)}
.done{color:var(--no);font-weight:600}
</style></head><body>
<header><h1>Contradiction validation — round 2</h1>
  <div class="bar"><i id="pbar"></i></div>
  <span class="count" id="pcount"></span>
  <button id="exp">Export CSV</button></header>
<main>
  <div class="card">
    <div class="meta" id="meta"></div>
    <h2>Question</h2><p class="q" id="q"></p>
  </div>
  <div class="card">
    <h2>Retrieved context the model actually received (budgeted)</h2>
    <pre class="ctx" id="ctx"></pre>
  </div>
  <div class="card grid">
    <div><h2>Generated answer</h2><div class="ans" id="ans"></div></div>
    <div><h2>Gold answer</h2><div class="gold" id="gold"></div></div>
  </div>
  <div class="card">
    <div class="qbox">
      <div class="qtxt">Does the answer contradict the EHR context?
        <div class="hint">Same question as session 1. Yes = the answer asserts
          something the EHR context directly refutes. Negation that is part of
          the diagnosis name itself (e.g. "Spondylosis without myelopathy") is
          the record's own wording, not a contradiction.</div>
      </div>
      <div class="btns" id="b_ehr">
        <button data-v="yes">Yes <kbd>1</kbd></button>
        <button data-v="no">No <kbd>2</kbd></button>
      </div>
    </div>
    <div class="qbox">
      <div class="qtxt">Notes <span class="hint">(optional)</span></div>
      <textarea id="note" placeholder="Anything worth recording..."></textarea>
    </div>
    <div class="rev" id="rev"></div>
  </div>
  <nav><button id="prev">&larr; Prev <kbd>&#8592;</kbd></button>
    <span class="count" id="status"></span>
    <button id="next">Next &rarr; <kbd>&#8594;</kbd></button></nav>
</main>
<script>
const ROWS = __ROWS_JSON__;
const KEY = "medrag_annot_v2";
let ann = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;
const $ = id => document.getElementById(id);
const save = () => localStorage.setItem(KEY, JSON.stringify(ann));
const rec = id => (ann[id] = ann[id] || {ehr:"", note:""});
function done(){ return ROWS.filter(r => ann[r.annot_id] && ann[r.annot_id].ehr).length; }

function render(){
  const r = ROWS[i], a = rec(r.annot_id);
  $("meta").innerHTML = `<span class="tag">${r.annot_id}</span>`
    + `<span class="tag">mode ${r.mode_used}</span>`
    + `<span class="tag">${r.question_type}</span>`
    + `<span class="tag">q_idx ${r.q_idx}</span>`;
  $("q").textContent = r.question;
  $("ctx").textContent = r.context || "(context rebuild failed — skip)";
  $("ans").textContent = r.predicted_answer;
  $("gold").textContent = r.reference;
  $("note").value = a.note || "";
  [...$("b_ehr").children].forEach(b => b.classList.toggle("sel", a.ehr === b.dataset.v));
  $("rev").innerHTML = a.ehr
    ? `<b>Detectors (revealed):</b> original <b>${r.det_v1}</b> · corrected <b>${r.det_v2}</b>`
    : `Detector labels hidden until you answer (blind annotation).`;
  const d = done();
  $("pbar").style.width = (100*d/ROWS.length) + "%";
  $("pcount").textContent = `${d}/${ROWS.length} done`;
  $("status").innerHTML = `Row ${i+1} of ${ROWS.length}` + (a.ehr ? ` · <span class="done">answered</span>` : "");
  $("prev").disabled = i === 0;
  $("next").disabled = i === ROWS.length - 1;
}
function set(v){
  rec(ROWS[i].annot_id).ehr = v; save(); render();
  if (i < ROWS.length-1) setTimeout(()=>{ i++; render(); }, 160);
}
$("b_ehr").addEventListener("click", e => { const b = e.target.closest("button"); if (b) set(b.dataset.v); });
$("note").addEventListener("input", e => { rec(ROWS[i].annot_id).note = e.target.value; save(); });
$("prev").onclick = () => { if(i>0){ i--; render(); } };
$("next").onclick = () => { if(i<ROWS.length-1){ i++; render(); } };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "1") { set("yes"); e.preventDefault(); }
  if (e.key === "2") { set("no");  e.preventDefault(); }
  if (e.key === "ArrowLeft"  && i>0)             { i--; render(); }
  if (e.key === "ArrowRight" && i<ROWS.length-1) { i++; render(); }
});
$("exp").onclick = () => {
  const cols = ["annot_id","q_idx","system","mode_used","question_type",
                "human_contradiction","human_note","det_v1","det_v2"];
  const esc = v => { v = (v===null||v===undefined) ? "" : String(v);
    return /[",\n]/.test(v) ? '"'+v.replace(/"/g,'""')+'"' : v; };
  const lines = [cols.join(",")];
  for (const r of ROWS){
    const a = ann[r.annot_id] || {};
    lines.push([r.annot_id, r.q_idx, r.system, r.mode_used, r.question_type,
                a.ehr||"", a.note||"", r.det_v1, r.det_v2].map(esc).join(","));
  }
  const blob = new Blob([lines.join("\n")], {type:"text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const el = document.createElement("a");
  el.href = url; el.download = "validation_filled.csv"; el.click();
  URL.revokeObjectURL(url);
};
render();
</script></body></html>
"""


def build_html(sample: pd.DataFrame) -> str:
    rows = []
    for _, r in sample.iterrows():
        rows.append({
            "annot_id": r["annot_id"], "q_idx": int(r["q_idx"]),
            "system": r["system"], "mode_used": r["mode_used"],
            "question_type": r["question_type"], "question": r["question"],
            "context": r["context_budgeted"],
            "predicted_answer": r["predicted_answer"], "reference": r["reference"],
            "det_v1": r["det_v1_label"], "det_v2": r["det_v2_label"],
        })
    return HTML_TEMPLATE.replace("__ROWS_JSON__", json.dumps(rows, ensure_ascii=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS)
    prior = pd.read_csv(PRIOR_ANNOT)

    sample = build_sample(df, prior)

    # Attention-check candidates come from unused control-stratum rows, so they
    # never cannibalise the strata under test.
    seen_rids = set(sample.rid)
    spare = df.copy()
    spare["rid"] = spare.q_idx.astype(str) + "|" + spare.system
    prior_rids = set(prior.q_idx.astype(str) + "|" + prior.system)
    spare = spare[(~spare.rid.isin(seen_rids)) & (~spare.rid.isin(prior_rids))
                  & (spare.mode_used != "T") & (spare.question_type.isin(DX_TYPES))]
    spare = spare.sample(frac=1.0, random_state=SEED).head(40)

    need = pd.concat([sample, spare], ignore_index=True)
    ctx = rebuild_contexts(need)

    checks = make_attention_checks(spare, ctx)
    if len(checks) < N_ATTENTION:
        raise SystemExit(f"[FATAL] only built {len(checks)}/{N_ATTENTION} attention checks.")

    sample = pd.concat([sample, checks], ignore_index=True)
    keys = list(zip(sample.hadm_id, sample.question, sample.mode_used))
    sample["context_unbudgeted"] = [ctx.get(k, ("", "", 0))[0] for k in keys]
    sample["context_budgeted"] = [ctx.get(k, ("", "", 0))[1] for k in keys]
    sample["n_kg_facts_rebuilt"] = [ctx.get(k, ("", "", 0))[2] for k in keys]

    tek = sample[sample.mode_used == "T+E+K"]
    lost = tek[(tek.n_kg_facts > 0) & (tek.n_kg_facts_rebuilt == 0)]
    if len(lost):
        raise SystemExit(f"[FATAL] {len(lost)} T+E+K rows lost their KG facts. Nothing written.")

    sample["ehr_contradiction_v1"] = [ehr_contradiction_score_v1(a, c)
                                      for a, c in zip(sample.predicted_answer, sample.context_budgeted)]
    sample["ehr_contradiction_v2"] = [negation_contradiction_score(a, c)
                                      for a, c in zip(sample.predicted_answer, sample.context_budgeted)]
    sample["det_v1_label"] = np.where(sample.ehr_contradiction_v1 > 0, "yes", "no")
    sample["det_v2_label"] = np.where(sample.ehr_contradiction_v2 > 0, "yes", "no")

    # Shuffle for presentation so strata and attention checks are interleaved.
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "annot_id", [f"V{i:03d}" for i in range(1, len(sample) + 1)])
    sample["human_contradiction"] = ""
    sample["human_note"] = ""

    sample.to_csv(SAMPLE_CSV, index=False)
    HTML_OUT.write_text(build_html(sample), encoding="utf-8")

    real = sample[~sample.is_attention_check]
    a_rows = real[real.stratum == "A_originally_flagged"]
    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED, "n_presented": int(len(sample)),
        "n_real": int(len(real)), "n_attention_checks": int(sample.is_attention_check.sum()),
        "strata": real.stratum.value_counts().to_dict(),
        "question_types": real.question_type.value_counts().to_dict(),
        "modes": real.mode_used.value_counts().to_dict(),
        "overlap_with_session1": 0,
        "prereg": {
            "label_rule": "corrected score > 0 -> yes",
            "success_criterion": "<=1 of 20 stratum-A rows a genuine contradiction per human "
                                 "AND corrected flags <=1 of them AND all synthetic controls pass",
            "locked": "2026-08-16, before annotation",
        },
        "detector_on_sample": {
            "stratum_A_original_flags": int((a_rows.det_v1_label == "yes").sum()),
            "stratum_A_corrected_flags": int((a_rows.det_v2_label == "yes").sum()),
            "attention_checks_corrected_flags":
                int((sample[sample.is_attention_check].det_v2_label == "yes").sum()),
        },
    }
    (OUT_DIR / "validation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("VALIDATION SAMPLE BUILT")
    print("=" * 74)
    print(f"  presented rows        : {len(sample)}  ({len(real)} real + "
          f"{int(sample.is_attention_check.sum())} attention checks)")
    print(f"  strata                : {meta['strata']}")
    print(f"  overlap with 75-row set: 0 (enforced)")
    print(f"  question types        : {meta['question_types']}")
    print(f"\n  stratum A (n={len(a_rows)}): original flags "
          f"{meta['detector_on_sample']['stratum_A_original_flags']}"
          f" -> corrected flags {meta['detector_on_sample']['stratum_A_corrected_flags']}")
    print(f"  attention checks      : corrected flags "
          f"{meta['detector_on_sample']['attention_checks_corrected_flags']}"
          f"/{int(sample.is_attention_check.sum())} (should be all)")
    print(f"\n  {SAMPLE_CSV}\n  {HTML_OUT}")


if __name__ == "__main__":
    main()
