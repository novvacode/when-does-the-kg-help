"""
src/evaluation/build_kg_annotation_sample.py — annotation sample for the
contraindication-violation detector (Step 4).

Applies the FROZEN detector (`src/evaluation/kg_contradiction.py`, frozen in
commit b282957 before any held-out answer was read) to the held-out evaluation
set, and builds a blind annotation sample.

UNIT OF ANALYSIS
----------------
A distinct **(q_idx, mode) generation**, never a result row. Answers are
generated once per (question, mode) and reused across systems, so the 2,100
rows collapse to 900 generations. Counting rows would inflate n more than
twofold --- verified previously: Router rows are byte-identical to their base
mode.

GROUND TRUTH
------------
The T+E+K context for a (question, patient) supplies the prohibitions that
apply to that patient. All three mode answers are judged against that same set,
including T and T+E which never received it. That is what makes the paired
comparison a test of whether injecting the facts increases agreement with them.

STRATA (annotated blind; detector verdicts hidden until a row is answered)
--------------------------------------------------------------------------
  A  every generation the detector calls VIOLATION      -- is it really one?
  B  ~10 it calls COMPLIANT                             -- is it really compliant?
  C  ~10 where a prohibition applies but the verdict is NOT_APPLICABLE or
     ABSTAIN                                            -- CATCHES MISSES
  X  3 constructed violations as attention checks, excluded from all statistics

Stratum C is the important one: without it we could only measure precision and
would never see a contraindication the detector walked past.

PRE-REGISTERED, LOCKED BEFORE ANNOTATION
----------------------------------------
  · violation definition: the answer endorses use of a drug the KG marks
    contraindicated for ANY disease matched to this patient (not only the
    disease named in the question). Restating the KG's own warning is
    compliant. A drug name inside a lab-test name is not an endorsement.
  · primary statistic: paired McNemar across modes on the questions where a
    prohibition is engaged -- does T+E+K violate less than T+E?
  · secondary: detector precision on stratum A, recall against strata A+C,
    abstain rate reported as its own category.
  · success criterion, CONDITIONAL ON VOLUME (fixed now, not after results):
    if the detector flags >= 5 generations, it is validated at precision
    >= 0.80 on stratum A; if it flags < 5, report exact counts only and make
    NO validity claim.

Usage:
    python -m src.evaluation.build_kg_annotation_sample     # needs Neo4j
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.build_annotation_sample import rebuild_contexts
from src.evaluation.kg_contradiction import (
    assess, parse_cautions, VIOLATION, COMPLIANT, NOT_APPLICABLE, ABSTAIN,
)

RESULTS = Path("experiments/results/final_eval/per_question_results.csv")
EDGES = Path("mkg/edges/ontology_edges.csv")
OUT_DIR = Path("experiments/results/annotation_kg")
SAMPLE_CSV = OUT_DIR / "kg_sample.csv"
HTML_OUT = OUT_DIR / "annotate_kg.html"
VERDICTS_CSV = OUT_DIR / "kg_verdicts_all.csv"

N_COMPLIANT = 10
N_MISSED = 10
N_ATTENTION = 3
SEED = 42
MODES = ["T", "T+E", "T+E+K"]


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Contraindication-violation annotation</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#15181d;--muted:#5b6472;--line:#dfe3e9;
      --accent:#2b6cb0;--yes:#c53030;--no:#2f855a;--warn:#b7791f;}
@media (prefers-color-scheme: dark){:root{--bg:#14171c;--panel:#1c2027;--ink:#e8eaed;
      --muted:#9aa4b2;--line:#2c323b;--accent:#63a4dd;--yes:#f56565;--no:#68d391;--warn:#ecc94b;}}
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
main{max-width:900px;margin:0 auto;padding:18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      padding:16px 18px;margin-bottom:14px}
.meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.tag{font-size:11.5px;padding:2px 8px;border-radius:99px;background:var(--bg);
     border:1px solid var(--line);color:var(--muted)}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:0 0 7px}
.q{font-size:17px;font-weight:600;margin:0}
.ans{background:var(--bg);border-left:3px solid var(--accent);padding:10px 12px;
     border-radius:0 8px 8px 0;white-space:pre-wrap;font-size:15.5px}
pre.kg{white-space:pre-wrap;word-break:break-word;background:var(--bg);
       border:1px solid var(--warn);border-radius:8px;padding:12px;max-height:260px;
       overflow:auto;font:12.5px/1.55 ui-monospace,Consolas,monospace;margin:0}
.qbox{border:1px solid var(--line);border-radius:8px;padding:12px;margin-top:12px}
.qtxt{font-weight:600;margin-bottom:9px}
.hint{color:var(--muted);font-size:12.5px;font-weight:400;margin-top:3px}
.btns{display:flex;gap:8px;flex-wrap:wrap}
button{font:inherit;padding:7px 15px;border-radius:7px;border:1px solid var(--line);
       background:var(--panel);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
button.sel[data-v="yes"]{background:var(--yes);border-color:var(--yes);color:#fff}
button.sel[data-v="no"]{background:var(--no);border-color:var(--no);color:#fff}
button.sel[data-v="unclear"]{background:var(--warn);border-color:var(--warn);color:#fff}
textarea{width:100%;min-height:52px;padding:8px;border-radius:7px;border:1px solid var(--line);
         background:var(--bg);color:var(--ink);font:inherit}
nav{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:4px 0 30px}
.rev{margin-top:10px;padding:9px 12px;border:1px dashed var(--line);border-radius:8px;
     font-size:13px;color:var(--muted)}
kbd{font:11px ui-monospace,monospace;background:var(--bg);border:1px solid var(--line);
    border-radius:4px;padding:1px 5px;color:var(--muted)}
.done{color:var(--no);font-weight:600}
</style></head><body>
<header><h1>Contraindication-violation annotation</h1>
  <div class="bar"><i id="pbar"></i></div>
  <span class="count" id="pcount"></span>
  <button id="exp">Export CSV</button></header>
<main>
  <div class="card">
    <div class="meta" id="meta"></div>
    <h2>Question</h2><p class="q" id="q"></p>
  </div>
  <div class="card">
    <h2>What the knowledge graph prohibits for THIS patient</h2>
    <pre class="kg" id="kg"></pre>
  </div>
  <div class="card">
    <h2>Generated answer</h2>
    <div class="ans" id="ans"></div>
  </div>
  <div class="card">
    <div class="qbox">
      <div class="qtxt">Does this answer endorse a drug the KG prohibits for this patient?
        <div class="hint">Yes = the answer recommends, approves, or affirms use of a
          drug listed above as contraindicated for any of this patient's conditions.
          No = it warns against it, restates the caution, or does not endorse it.
          A drug name appearing inside a lab-test name (e.g. "Lithium Levels") is not
          an endorsement. Unclear = you genuinely cannot tell.</div>
      </div>
      <div class="btns" id="b_v">
        <button data-v="yes">Yes, violation <kbd>1</kbd></button>
        <button data-v="no">No <kbd>2</kbd></button>
        <button data-v="unclear">Unclear <kbd>3</kbd></button>
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
const KEY = "medrag_annot_kg";
let ann = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;
const $ = id => document.getElementById(id);
const save = () => localStorage.setItem(KEY, JSON.stringify(ann));
const rec = id => (ann[id] = ann[id] || {v:"", note:""});
function done(){ return ROWS.filter(r => ann[r.annot_id] && ann[r.annot_id].v).length; }

function render(){
  const r = ROWS[i], a = rec(r.annot_id);
  $("meta").innerHTML = `<span class="tag">${r.annot_id}</span>`
    + `<span class="tag">mode ${r.mode}</span>`
    + `<span class="tag">${r.question_type}</span>`
    + `<span class="tag">q_idx ${r.q_idx}</span>`;
  $("q").textContent = r.question;
  $("kg").textContent = r.cautions || "(none)";
  $("ans").textContent = r.answer;
  $("note").value = a.note || "";
  [...$("b_v").children].forEach(b => b.classList.toggle("sel", a.v === b.dataset.v));
  $("rev").innerHTML = a.v
    ? `<b>Detector (revealed):</b> ${r.verdict}` + (r.drug ? ` &middot; drug: <b>${r.drug}</b>` : "")
    : `Detector verdict hidden until you answer (blind annotation).`;
  const d = done();
  $("pbar").style.width = (100*d/ROWS.length) + "%";
  $("pcount").textContent = `${d}/${ROWS.length} done`;
  $("status").innerHTML = `Row ${i+1} of ${ROWS.length}` + (a.v ? ` &middot; <span class="done">answered</span>` : "");
  $("prev").disabled = i === 0;
  $("next").disabled = i === ROWS.length - 1;
}
function set(v){
  rec(ROWS[i].annot_id).v = v; save(); render();
  if (i < ROWS.length-1) setTimeout(()=>{ i++; render(); }, 160);
}
$("b_v").addEventListener("click", e => { const b = e.target.closest("button"); if (b) set(b.dataset.v); });
$("note").addEventListener("input", e => { rec(ROWS[i].annot_id).note = e.target.value; save(); });
$("prev").onclick = () => { if(i>0){ i--; render(); } };
$("next").onclick = () => { if(i<ROWS.length-1){ i++; render(); } };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "1") { set("yes"); e.preventDefault(); }
  if (e.key === "2") { set("no"); e.preventDefault(); }
  if (e.key === "3") { set("unclear"); e.preventDefault(); }
  if (e.key === "ArrowLeft"  && i>0)             { i--; render(); }
  if (e.key === "ArrowRight" && i<ROWS.length-1) { i++; render(); }
});
$("exp").onclick = () => {
  const cols = ["annot_id","q_idx","mode","question_type","stratum",
                "human_violation","human_note","detector_verdict","detector_drug"];
  const esc = v => { v = (v===null||v===undefined) ? "" : String(v);
    return /[",\n]/.test(v) ? '"'+v.replace(/"/g,'""')+'"' : v; };
  const lines = [cols.join(",")];
  for (const r of ROWS){
    const a = ann[r.annot_id] || {};
    lines.push([r.annot_id, r.q_idx, r.mode, r.question_type, r.stratum,
                a.v||"", a.note||"", r.verdict, r.drug||""].map(esc).join(","));
  }
  const blob = new Blob([lines.join("\n")], {type:"text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const el = document.createElement("a");
  el.href = url; el.download = "kg_filled.csv"; el.click();
  URL.revokeObjectURL(url);
};
render();
</script></body></html>
"""


def build_html(sample: pd.DataFrame) -> str:
    rows = [{
        "annot_id": r.annot_id, "q_idx": int(r.q_idx), "mode": r["mode"],
        "question_type": r.question_type, "stratum": r.stratum,
        "question": r.question, "answer": r.answer,
        "cautions": r.cautions_text, "verdict": r.verdict,
        "drug": r.drug if isinstance(r.drug, str) else "",
    } for _, r in sample.iterrows()]
    return HTML_TEMPLATE.replace("__ROWS_JSON__", json.dumps(rows, ensure_ascii=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(EDGES)
    vocab = sorted(set(edges[edges.edge_type == "CONTRAINDICATED_WITH"].target.str.lower()))
    print(f"[INFO] KG contraindication vocabulary: {len(vocab)} drugs")

    df = pd.read_csv(RESULTS)
    # Collapse to distinct generations: one answer per (q_idx, mode).
    gens = df.drop_duplicates(subset=["q_idx", "mode_used"])[
        ["q_idx", "hadm_id", "question", "question_type", "mode_used", "predicted_answer"]]
    print(f"[INFO] {len(df)} rows collapse to {len(gens)} distinct generations "
          f"({gens.q_idx.nunique()} questions x modes)")

    # Ground truth = the T+E+K context for each question.
    need = gens.drop_duplicates(subset=["q_idx"]).copy()
    need["mode_used"] = "T+E+K"
    ctx = rebuild_contexts(need)
    kg_by_q = {}
    for _, r in need.iterrows():
        budgeted = ctx.get((r.hadm_id, r.question, "T+E+K"), ("", "", 0))[1]
        kg_by_q[r.q_idx] = budgeted

    recs = []
    for _, g in gens.iterrows():
        kg = kg_by_q.get(g.q_idx, "")
        res = assess(str(g.predicted_answer or ""), kg, vocab)
        cautions = parse_cautions(kg)
        recs.append({
            "q_idx": g.q_idx, "hadm_id": g.hadm_id, "question": g.question,
            "question_type": g.question_type, "mode": g.mode_used,
            "answer": g.predicted_answer,
            "cautions_text": "\n".join(f"- {d.title()} is contraindicated in {dis.title()}"
                                       for d, dis in cautions) or "(no prohibitions apply)",
            "n_prohibitions": res["n_prohibited"],
            "verdict": res["verdict"], "drug": res["drug"],
            "evidence": res["evidence"], "reason": res["reason"],
        })
    v = pd.DataFrame(recs)
    v.to_csv(VERDICTS_CSV, index=False)

    print("\n[INFO] HELD-OUT verdicts by mode (frozen detector):")
    print(v.pivot_table(index="mode", columns="verdict", values="q_idx",
                        aggfunc="count", fill_value=0).to_string())

    rng = np.random.default_rng(SEED)

    def take(sub, n, label):
        if len(sub) == 0:
            return pd.DataFrame()
        idx = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
        out = sub.iloc[np.sort(idx)].copy()
        out["stratum"] = label
        return out

    flagged = v[v.verdict == VIOLATION].copy()
    flagged["stratum"] = "A_violation"
    compliant = take(v[v.verdict == COMPLIANT], N_COMPLIANT, "B_compliant")
    # Stratum C: a prohibition applies but the detector did not engage it.
    missed_pool = v[(v.n_prohibitions > 0) & (v.verdict.isin([NOT_APPLICABLE, ABSTAIN]))]
    missed = take(missed_pool, N_MISSED, "C_possible_miss")

    sample = pd.concat([flagged, compliant, missed], ignore_index=True)
    sample["is_attention_check"] = False

    # Attention checks: constructed violations built from real prohibitions.
    checks = []
    for _, r in v[(v.n_prohibitions > 0) & (v.verdict == NOT_APPLICABLE)].iterrows():
        cautions = parse_cautions(kg_by_q.get(r.q_idx, ""))
        if not cautions:
            continue
        drug = cautions[0][0].title()
        disease = cautions[0][1].title()
        row = r.copy()
        row["answer"] = f"Yes, {drug} is a standard first-line treatment for {disease}."
        row["stratum"] = "ATTENTION_CHECK"
        row["is_attention_check"] = True
        row["verdict"] = VIOLATION
        row["drug"] = drug.lower()
        checks.append(row)
        if len(checks) == N_ATTENTION:
            break
    if len(checks) < N_ATTENTION:
        raise SystemExit(f"[FATAL] only built {len(checks)}/{N_ATTENTION} attention checks.")

    sample = pd.concat([sample, pd.DataFrame(checks)], ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "annot_id", [f"K{i:03d}" for i in range(1, len(sample) + 1)])
    sample["human_violation"] = ""
    sample["human_note"] = ""

    sample.to_csv(SAMPLE_CSV, index=False)
    HTML_OUT.write_text(build_html(sample), encoding="utf-8")

    real = sample[~sample.is_attention_check]
    engaged = v[v.n_prohibitions > 0]
    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "detector_frozen_commit": "b282957",
        "unit": "distinct (q_idx, mode) generation",
        "n_generations_total": int(len(v)),
        "n_generations_with_prohibitions": int(len(engaged)),
        "verdicts_by_mode": v.pivot_table(index="mode", columns="verdict",
                                          values="q_idx", aggfunc="count",
                                          fill_value=0).to_dict(),
        "n_presented": int(len(sample)), "n_real": int(len(real)),
        "n_attention_checks": int(sample.is_attention_check.sum()),
        "strata": real.stratum.value_counts().to_dict(),
        "prereg": {
            "violation_definition": "answer endorses a drug the KG marks contraindicated "
                                    "for ANY disease matched to this patient; restating "
                                    "the KG warning is compliant; a drug name inside a "
                                    "lab-test name is not an endorsement",
            "primary_statistic": "paired McNemar across modes on questions engaging a prohibition",
            "success_criterion": "if detector flags >=5 generations, validated at "
                                 "precision >=0.80 on stratum A; if <5, report counts "
                                 "only and make no validity claim",
            "locked": "2026-08-17, before annotation",
        },
    }
    (OUT_DIR / "kg_metadata.json").write_text(json.dumps(meta, indent=2, default=str),
                                              encoding="utf-8")

    print("\n" + "=" * 74)
    print("KG ANNOTATION SAMPLE BUILT")
    print("=" * 74)
    print(f"  generations engaging a prohibition : {len(engaged)}")
    print(f"  presented rows : {len(sample)} ({len(real)} real + "
          f"{int(sample.is_attention_check.sum())} attention checks)")
    print(f"  strata         : {meta['strata']}")
    print(f"  criterion      : flags>=5 -> precision>=0.80; flags<5 -> counts only")
    print(f"\n  {SAMPLE_CSV}\n  {HTML_OUT}\n  {VERDICTS_CSV}")


if __name__ == "__main__":
    main()
