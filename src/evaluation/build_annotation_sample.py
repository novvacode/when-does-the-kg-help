"""
src/evaluation/build_annotation_sample.py — sample + tooling for the human
hallucination-annotation check.

WHY THIS EXISTS
===============
Every hallucination number in the paper comes from two automatic heuristics
(`ehr_contradiction_score`, `unsupported_score` in run_evaluation.py). The
Limitations section already concedes they are "automatic heuristics, not
clinician-adjudicated", with no human evaluation and no agreement study. This
builds the sample and the tool for that study; annotation_agreement.py scores
it afterwards.

WHAT IT PRODUCES
----------------
  experiments/results/annotation/sample_75.csv   the sample, ready to annotate
  experiments/results/annotation/annotate.html   self-contained annotation UI
  experiments/results/annotation/sample_metadata.json

THE CONTEXT PROBLEM (and how drift is handled)
----------------------------------------------
run_evaluation.py does not persist prompt_context, so contexts must be rebuilt
through the live Retriever (retrieval only — no LLM is loaded). Rebuilt
contexts are not guaranteed identical to eval time, so this script records
THREE versions of each detector score instead of hiding the gap:

  *_stored              from per_question_results.csv (eval time, UNBUDGETED)
  *_rebuilt_unbudgeted  recomputed on the rebuilt context, unbudgeted
  *_rebuilt             recomputed on the BUDGETED context — what the model
                        actually saw, and what the annotator is shown

Comparing stored vs rebuilt_unbudgeted isolates retrieval reproduction;
comparing rebuilt_unbudgeted vs rebuilt isolates the effect of per-section
budgeting. run_evaluation.py scores the unbudgeted context while generate()
feeds the model the budgeted one, so these genuinely differ.

`*_rebuilt` is the primary comparator for the agreement study: the annotator
and the detector must judge identical evidence, or the resulting kappa
measures context drift as well as disagreement.

PRE-REGISTERED BINARIZATION (locked 2026-08-13, before any annotation)
---------------------------------------------------------------------
  EHR-contradiction : score > 0 -> yes.  mode_used == "T" -> n/a, never "no"
                      (the detector scans an EHR snapshot mode T never
                      receives — see RESEARCH_LOG 2026-08-15).
  Unsupported claim : score >= 0.5 -> yes (primary). A 0.1-0.9 sweep is
                      reported as sensitivity by annotation_agreement.py.

The threshold is fixed BEFORE annotation deliberately, so it cannot be tuned
against the human labels afterwards. Do not change it retrospectively.

Usage:
    python -m src.evaluation.build_annotation_sample     # needs Neo4j running
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("experiments/results/final_eval/per_question_results.csv")
OUT_DIR = Path("experiments/results/annotation")
SAMPLE_CSV = OUT_DIR / "sample_75.csv"
HTML_OUT = OUT_DIR / "annotate.html"

N_SAMPLE = 75
SEED = 42
UNSUPPORTED_THRESHOLD = 0.5      # PRE-REGISTERED — see module docstring
COMMON_WORDS = {"the", "a", "an", "is", "was", "are", "for", "in", "of",
                "to", "and", "or", "this", "that", "with", "has", "have"}


# ── Detector logic, copied verbatim from run_evaluation.py ────────────────────
# Duplicated rather than imported because importing run_evaluation pulls in
# torch + transformers + the generator. Any divergence here would silently
# invalidate the agreement study, so these must stay byte-equivalent.

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
    answer_words = set(str(answer).lower().split()) - COMMON_WORDS
    if not answer_words:
        return 0.0
    context_words = set(str(context).lower().split())
    return len(answer_words - context_words) / len(answer_words)


# ── Sampling ──────────────────────────────────────────────────────────────────


def stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    """75 rows spread across all 7 systems, seed-42 reproducible."""
    systems = sorted(df["system"].unique())
    base, extra = divmod(N_SAMPLE, len(systems))
    rng = np.random.default_rng(SEED)

    parts = []
    for i, s in enumerate(systems):
        n = base + (1 if i < extra else 0)
        sub = df[df["system"] == s]
        idx = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
        parts.append(sub.iloc[np.sort(idx)])
    out = pd.concat(parts, ignore_index=True)

    # Interleave so the annotator does not work through 11 rows of one system
    # in a block — order effects are a real risk in sequential annotation.
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out.insert(0, "annot_id", [f"A{i:03d}" for i in range(1, len(out) + 1)])
    return out


# ── Context rebuild ───────────────────────────────────────────────────────────


def rebuild_contexts(sample: pd.DataFrame) -> dict[tuple, tuple[str, str]]:
    """Return {(hadm_id, question, mode): (unbudgeted, budgeted)}."""
    from transformers import AutoTokenizer
    from src.retrieval.retriever import Retriever, Mode as RMode
    from src.model.context_budget import build_budgeted_context

    try:
        import src.mkg.retrieval as kg
    except Exception as e:
        raise SystemExit(
            f"[FATAL] KG module unavailable ({e}). T+E+K contexts would be "
            f"silently missing their KG facts, which would corrupt the "
            f"annotation. Start Neo4j and re-run."
        )

    # Importing the module does NOT open a connection, so an import check alone
    # is worthless: with bad credentials every KG lookup fails, Retriever
    # swallows the error, and T+E+K contexts come back silently KG-less. That
    # happened on the 2026-08-13 first run — 0/19 T+E+K rows had KG facts while
    # 13 of them had up to 23 at eval time. Actually execute a query here.
    try:
        drv = kg.get_driver()
        with drv.session() as sess:
            n_disease = sess.run("MATCH (d:Disease) RETURN count(d) AS n").single()["n"]
        drv.close()
    except Exception as e:
        raise SystemExit(
            f"[FATAL] Neo4j is reachable but the query failed: {e}\n"
            f"        Most likely NEO4J_PASSWORD is unset or wrong. In PowerShell "
            f"use SINGLE quotes so '$' and '`' in the password are not expanded:\n"
            f"            $env:NEO4J_PASSWORD='your-password'\n"
            f"        If the message mentions AuthenticationRateLimit, Neo4j has "
            f"locked out after repeated failures — wait ~30s or restart the DBMS.\n"
            f"        Refusing to build the sample: T+E+K contexts would be "
            f"silently missing their KG facts."
        )
    if n_disease == 0:
        raise SystemExit("[FATAL] Neo4j authenticated but holds 0 Disease nodes — "
                         "the MKG is not loaded. Run src.mkg.neo4j_loader first.")
    print(f"[INFO] Neo4j preflight OK — {n_disease} Disease nodes reachable.")

    tok = AutoTokenizer.from_pretrained("google/medgemma-1.5-4b-it")
    r = Retriever(kg_module=kg)
    mode_map = {"T": RMode.T, "T+E": RMode.TE, "T+E+K": RMode.TEK}

    need = sample[["hadm_id", "question", "mode_used"]].drop_duplicates()
    print(f"[INFO] Rebuilding {len(need)} unique contexts (retrieval only, no LLM)...")

    ctx: dict[tuple, tuple[str, str, int]] = {}
    failures = 0
    for i, (_, row) in enumerate(need.iterrows(), 1):
        key = (row["hadm_id"], row["question"], row["mode_used"])
        try:
            res = r.retrieve(question=str(row["question"]),
                             hadm_id=int(row["hadm_id"]),
                             mode=mode_map[row["mode_used"]])
            ctx[key] = (res.prompt_context,
                        build_budgeted_context(res.prompt_context, tok),
                        len(res.kg_facts))
        except Exception as e:
            print(f"[WARN] retrieval failed for {key}: {e}")
            ctx[key] = ("", "", 0)
            failures += 1
        if i % 20 == 0:
            print(f"       {i}/{len(need)}")
    r.close()

    if failures:
        print(f"[WARN] {failures} context rebuild(s) FAILED and are empty. "
              f"Those rows cannot be annotated meaningfully.")
    return ctx


# ── HTML tool ─────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hallucination annotation — 75 rows</title>
<style>
:root{
  --bg:#f6f7f9; --panel:#fff; --ink:#15181d; --muted:#5b6472; --line:#dfe3e9;
  --accent:#2b6cb0; --yes:#c53030; --no:#2f855a; --na:#718096; --warn:#b7791f;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#14171c; --panel:#1c2027; --ink:#e8eaed; --muted:#9aa4b2;
         --line:#2c323b; --accent:#63a4dd; --yes:#f56565; --no:#68d391;
         --na:#a0aec0; --warn:#ecc94b; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:10;background:var(--panel);
       border-bottom:1px solid var(--line);padding:10px 18px;
       display:flex;gap:14px;align-items:center;flex-wrap:wrap}
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
h2{font-size:12px;text-transform:uppercase;letter-spacing:.07em;
   color:var(--muted);margin:0 0 7px}
.q{font-size:17px;font-weight:600;margin:0}
pre.ctx{white-space:pre-wrap;word-break:break-word;background:var(--bg);
        border:1px solid var(--line);border-radius:8px;padding:12px;
        max-height:340px;overflow:auto;font:12.5px/1.5 ui-monospace,Consolas,monospace;margin:0}
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
button.sel[data-v="na"]{background:var(--na);border-color:var(--na);color:#fff}
button:disabled{opacity:.4;cursor:not-allowed}
textarea{width:100%;min-height:52px;padding:8px;border-radius:7px;
         border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit}
nav{display:flex;gap:10px;align-items:center;justify-content:space-between;
    padding:4px 0 30px}
.rev{margin-top:10px;padding:9px 12px;border:1px dashed var(--line);
     border-radius:8px;font-size:13px;color:var(--muted)}
.locked{color:var(--warn);font-size:12.5px;margin-top:6px}
.done{color:var(--no);font-weight:600}
kbd{font:11px ui-monospace,monospace;background:var(--bg);border:1px solid var(--line);
    border-radius:4px;padding:1px 5px;color:var(--muted)}
</style>
</head>
<body>
<header>
  <h1>Hallucination annotation</h1>
  <div class="bar"><i id="pbar"></i></div>
  <span class="count" id="pcount"></span>
  <button id="exp">Export CSV</button>
</header>

<main>
  <div class="card">
    <div class="meta" id="meta"></div>
    <h2>Question</h2>
    <p class="q" id="q"></p>
  </div>

  <div class="card">
    <h2>Retrieved context the model actually received (budgeted)</h2>
    <pre class="ctx" id="ctx"></pre>
  </div>

  <div class="card grid">
    <div>
      <h2>Generated answer</h2>
      <div class="ans" id="ans"></div>
    </div>
    <div>
      <h2>Gold answer</h2>
      <div class="gold" id="gold"></div>
    </div>
  </div>

  <div class="card">
    <div class="qbox">
      <div class="qtxt">1. Does the answer contradict the EHR context?
        <div class="hint">Yes = the answer asserts something the EHR context
          directly refutes. n/a = mode T (no EHR snapshot was provided, so the
          question cannot be asked).</div>
      </div>
      <div class="btns" id="b_ehr">
        <button data-v="yes">Yes <kbd>1</kbd></button>
        <button data-v="no">No <kbd>2</kbd></button>
        <button data-v="na">n/a <kbd>3</kbd></button>
      </div>
      <div class="locked" id="ehrlock" hidden>Mode T — locked to n/a by the
        pre-registered rule (the detector scans an EHR snapshot mode T never receives).</div>
    </div>

    <div class="qbox">
      <div class="qtxt">2. Does the answer contain an unsupported claim?
        <div class="hint">Yes = it states something not grounded in the context
          above, regardless of whether it happens to be true.</div>
      </div>
      <div class="btns" id="b_uns">
        <button data-v="yes">Yes <kbd>4</kbd></button>
        <button data-v="no">No <kbd>5</kbd></button>
      </div>
    </div>

    <div class="qbox">
      <div class="qtxt">Notes <span class="hint">(optional)</span></div>
      <textarea id="note" placeholder="Anything worth recording..."></textarea>
    </div>

    <div class="rev" id="rev"></div>
  </div>

  <nav>
    <button id="prev">&larr; Prev <kbd>&#8592;</kbd></button>
    <span class="count" id="status"></span>
    <button id="next">Next &rarr; <kbd>&#8594;</kbd></button>
  </nav>
</main>

<script>
const ROWS = __ROWS_JSON__;
const KEY  = "medrag_annot_v1";
let ann = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;

const $ = id => document.getElementById(id);
const save = () => localStorage.setItem(KEY, JSON.stringify(ann));
const rec  = id => (ann[id] = ann[id] || {ehr:"", uns:"", note:""});

function done(){ return ROWS.filter(r => ann[r.annot_id] && ann[r.annot_id].ehr && ann[r.annot_id].uns).length; }

function render(){
  const r = ROWS[i], a = rec(r.annot_id);
  $("meta").innerHTML =
      `<span class="tag">${r.annot_id}</span>`
    + `<span class="tag">mode ${r.mode_used}</span>`
    + `<span class="tag">${r.question_type}</span>`
    + `<span class="tag">sparsity ${r.sparsity_bucket}</span>`
    + `<span class="tag">q_idx ${r.q_idx}</span>`;
  $("q").textContent    = r.question;
  $("ctx").textContent  = r.context || "(context rebuild failed — skip this row)";
  $("ans").textContent  = r.predicted_answer;
  $("gold").textContent = r.reference;
  $("note").value       = a.note || "";

  const isT = r.mode_used === "T";
  $("ehrlock").hidden = !isT;
  if (isT && a.ehr !== "na"){ a.ehr = "na"; save(); }
  [...$("b_ehr").children].forEach(b => {
    b.disabled = isT && b.dataset.v !== "na";
    b.classList.toggle("sel", a.ehr === b.dataset.v);
  });
  [...$("b_uns").children].forEach(b =>
    b.classList.toggle("sel", a.uns === b.dataset.v));

  // Blind by default: detector labels appear only after BOTH answers are given.
  const answered = a.ehr && a.uns;
  $("rev").innerHTML = answered
    ? `<b>Detector (revealed):</b> EHR-contradiction <b>${r.det_ehr_label}</b>`
      + ` (score ${r.det_ehr_score}) &nbsp;·&nbsp; unsupported <b>${r.det_uns_label}</b>`
      + ` (score ${r.det_uns_score}, threshold ≥ ${r.threshold})`
    : `Detector labels hidden until you answer both questions (blind annotation).`;

  const d = done();
  $("pbar").style.width = (100*d/ROWS.length) + "%";
  $("pcount").textContent = `${d}/${ROWS.length} done`;
  $("status").innerHTML = `Row ${i+1} of ${ROWS.length}` + (answered ? ` · <span class="done">answered</span>` : "");
  $("prev").disabled = i === 0;
  $("next").disabled = i === ROWS.length - 1;
}

function set(field, v){
  const r = ROWS[i], a = rec(r.annot_id);
  if (field === "ehr" && r.mode_used === "T" && v !== "na") return;
  a[field] = v; save(); render();
  // Auto-advance once a row is fully answered, to keep the pass moving.
  if (a.ehr && a.uns && i < ROWS.length-1) setTimeout(()=>{ i++; render(); }, 160);
}

$("b_ehr").addEventListener("click", e => { const b=e.target.closest("button");
  if (b && !b.disabled) set("ehr", b.dataset.v); });
$("b_uns").addEventListener("click", e => { const b=e.target.closest("button");
  if (b) set("uns", b.dataset.v); });
$("note").addEventListener("input", e => { rec(ROWS[i].annot_id).note = e.target.value; save(); });
$("prev").onclick = () => { if(i>0){ i--; render(); } };
$("next").onclick = () => { if(i<ROWS.length-1){ i++; render(); } };

document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  const m = {"1":["ehr","yes"],"2":["ehr","no"],"3":["ehr","na"],
             "4":["uns","yes"],"5":["uns","no"]};
  if (m[e.key]) { set(m[e.key][0], m[e.key][1]); e.preventDefault(); }
  if (e.key === "ArrowLeft"  && i>0)            { i--; render(); }
  if (e.key === "ArrowRight" && i<ROWS.length-1){ i++; render(); }
});

$("exp").onclick = () => {
  const cols = ["annot_id","q_idx","system","mode_used","question_type",
                "human_ehr_contradiction","human_unsupported","human_note",
                "det_ehr_label","det_uns_label","det_ehr_score","det_uns_score"];
  const esc = v => { v = (v===null||v===undefined) ? "" : String(v);
    return /[",\n]/.test(v) ? '"'+v.replace(/"/g,'""')+'"' : v; };
  const lines = [cols.join(",")];
  for (const r of ROWS){
    const a = ann[r.annot_id] || {};
    lines.push([r.annot_id, r.q_idx, r.system, r.mode_used, r.question_type,
                a.ehr||"", a.uns||"", a.note||"",
                r.det_ehr_label, r.det_uns_label, r.det_ehr_score, r.det_uns_score
               ].map(esc).join(","));
  }
  const blob = new Blob([lines.join("\n")], {type:"text/csv;charset=utf-8"});
  const url  = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "annotations_filled.csv"; a.click();
  URL.revokeObjectURL(url);
};

render();
</script>
</body>
</html>
"""


def build_html(sample: pd.DataFrame) -> str:
    """Embed the sample as JSON in the self-contained annotation page."""
    rows = []
    for _, r in sample.iterrows():
        rows.append({
            "annot_id": r["annot_id"],
            "q_idx": int(r["q_idx"]),
            "system": r["system"],           # exported, NOT displayed while annotating
            "mode_used": r["mode_used"],
            "question_type": r["question_type"],
            "sparsity_bucket": r["sparsity_bucket"],
            "question": r["question"],
            "context": r["context_budgeted"],
            "predicted_answer": r["predicted_answer"],
            "reference": r["reference"],
            "det_ehr_label": r["det_ehr_label_rebuilt"],
            "det_uns_label": r["det_uns_label_rebuilt"],
            "det_ehr_score": round(float(r["ehr_contradiction_rebuilt"]), 4),
            "det_uns_score": round(float(r["unsupported_rebuilt"]), 4),
            "threshold": UNSUPPORTED_THRESHOLD,
        })
    payload = json.dumps(rows, ensure_ascii=False)
    return HTML_TEMPLATE.replace("__ROWS_JSON__", payload)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS)
    print(f"[INFO] {len(df)} result rows / {df.q_idx.nunique()} questions / "
          f"{df.system.nunique()} systems")

    sample = stratified_sample(df)
    print(f"[INFO] Sampled {len(sample)} rows: "
          f"{sample.system.value_counts().sort_index().to_dict()}")

    ctx = rebuild_contexts(sample)
    keys = list(zip(sample.hadm_id, sample.question, sample.mode_used))
    sample["context_unbudgeted"] = [ctx.get(k, ("", "", 0))[0] for k in keys]
    sample["context_budgeted"] = [ctx.get(k, ("", "", 0))[1] for k in keys]
    sample["n_kg_facts_rebuilt"] = [ctx.get(k, ("", "", 0))[2] for k in keys]

    # Integrity check: a T+E+K row that HAD KG facts at eval time must have them
    # again now. The preflight above should make this unreachable; it stays as a
    # second line of defence, because the failure it catches is silent and
    # produces a corrupt annotation sample rather than an error.
    tek = sample[sample.mode_used == "T+E+K"]
    lost = tek[(tek.n_kg_facts > 0) & (tek.n_kg_facts_rebuilt == 0)]
    if len(lost):
        raise SystemExit(
            f"[FATAL] {len(lost)} of {len(tek)} T+E+K rows had KG facts at eval "
            f"time but rebuilt with ZERO. The KG is not being reached; the "
            f"sample would be corrupt. Nothing written. Fix Neo4j auth and re-run."
        )

    # Three detector versions — see module docstring.
    sample["ehr_contradiction_stored"] = sample["ehr_contradiction"]
    sample["unsupported_stored"] = sample["unsupported_rate"]
    for suffix, col in [("rebuilt_unbudgeted", "context_unbudgeted"),
                        ("rebuilt", "context_budgeted")]:
        sample[f"ehr_contradiction_{suffix}"] = [
            ehr_contradiction_score(a, c)
            for a, c in zip(sample.predicted_answer, sample[col])]
        sample[f"unsupported_{suffix}"] = [
            unsupported_score(a, c)
            for a, c in zip(sample.predicted_answer, sample[col])]

    # Pre-registered binarization of the primary (rebuilt/budgeted) scores.
    sample["det_ehr_label_rebuilt"] = np.where(
        sample.mode_used == "T", "na",
        np.where(sample.ehr_contradiction_rebuilt > 0, "yes", "no"))
    sample["det_uns_label_rebuilt"] = np.where(
        sample.unsupported_rebuilt >= UNSUPPORTED_THRESHOLD, "yes", "no")

    # Blank columns for the annotator; filled via annotate.html's CSV export.
    sample["human_ehr_contradiction"] = ""
    sample["human_unsupported"] = ""
    sample["human_note"] = ""

    sample.to_csv(SAMPLE_CSV, index=False)
    HTML_OUT.write_text(build_html(sample), encoding="utf-8")

    # ── Drift report ──────────────────────────────────────────────────────────
    def _drift(a: str, b: str, sub: pd.DataFrame | None = None) -> dict:
        d = sample if sub is None else sub
        ea = (d[f"ehr_contradiction_{a}"] > 0)
        eb = (d[f"ehr_contradiction_{b}"] > 0)
        ua = (d[f"unsupported_{a}"] >= UNSUPPORTED_THRESHOLD)
        ub = (d[f"unsupported_{b}"] >= UNSUPPORTED_THRESHOLD)
        return {
            "n": int(len(d)),
            "ehr_label_disagreements": int((ea != eb).sum()),
            "unsupported_label_disagreements": int((ua != ub).sum()),
            "unsupported_mean_abs_score_delta": round(float(
                (d[f"unsupported_{a}"] - d[f"unsupported_{b}"]).abs().mean()), 4),
        }

    drift = {
        "stored_vs_rebuilt_unbudgeted (retrieval reproduction)":
            _drift("stored", "rebuilt_unbudgeted"),
        "rebuilt_unbudgeted_vs_rebuilt (effect of budgeting)":
            _drift("rebuilt_unbudgeted", "rebuilt"),
        "stored_vs_rebuilt (total drift vs eval-time labels)":
            _drift("stored", "rebuilt"),
        # Per mode, because an aggregate over 75 rows can hide a fault confined
        # to one mode — exactly how the first run's missing-KG corruption
        # looked like a mild 6/75 drift.
        "stored_vs_rebuilt_by_mode": {
            m: _drift("stored", "rebuilt", sample[sample.mode_used == m])
            for m in sorted(sample.mode_used.unique())
        },
    }

    n_na = int((sample.mode_used == "T").sum())
    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_sample": int(len(sample)), "seed": SEED,
        "stratification": "system (7 strata), random within stratum, then shuffled",
        "system_counts": sample.system.value_counts().sort_index().to_dict(),
        "mode_counts": sample.mode_used.value_counts().sort_index().to_dict(),
        "n_rows_na_for_ehr_contradiction": n_na,
        "n_rows_scorable_for_ehr_contradiction": int(len(sample) - n_na),
        "prereg_unsupported_threshold": UNSUPPORTED_THRESHOLD,
        "prereg_locked": "2026-08-13, before annotation. Do not retune.",
        "detector_positives_rebuilt": {
            "ehr_contradiction_yes": int((sample.det_ehr_label_rebuilt == "yes").sum()),
            "unsupported_yes": int((sample.det_uns_label_rebuilt == "yes").sum()),
        },
        "context_drift": drift,
        "n_failed_context_rebuilds": int((sample.context_budgeted == "").sum()),
        "kg_facts_eval_vs_rebuilt": {
            "tek_rows": int((sample.mode_used == "T+E+K").sum()),
            "tek_with_kg_at_eval": int(((sample.mode_used == "T+E+K") & (sample.n_kg_facts > 0)).sum()),
            "tek_with_kg_rebuilt": int(((sample.mode_used == "T+E+K") & (sample.n_kg_facts_rebuilt > 0)).sum()),
        },
    }
    (OUT_DIR / "sample_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("SAMPLE BUILT")
    print("=" * 74)
    print(f"  rows                     : {len(sample)}")
    print(f"  systems                  : {meta['system_counts']}")
    print(f"  modes                    : {meta['mode_counts']}")
    print(f"  n/a for EHR-contradiction: {n_na}  (mode T)")
    print(f"  scorable for EHR-contra. : {len(sample) - n_na}")
    print(f"  detector says YES        : EHR {meta['detector_positives_rebuilt']['ehr_contradiction_yes']}"
          f" · unsupported {meta['detector_positives_rebuilt']['unsupported_yes']}")
    print(f"  failed context rebuilds  : {meta['n_failed_context_rebuilds']}")
    kgm = meta["kg_facts_eval_vs_rebuilt"]
    print(f"  KG facts on T+E+K rows   : {kgm['tek_with_kg_rebuilt']}/{kgm['tek_rows']} rebuilt"
          f"  (eval time: {kgm['tek_with_kg_at_eval']}/{kgm['tek_rows']})")
    print("\n  CONTEXT DRIFT (reported, not buried):")
    for k, v in drift.items():
        if k.endswith("_by_mode"):
            print(f"    {k}")
            for m, mv in v.items():
                print(f"      mode {m:<6} n={mv['n']:<3}"
                      f" EHR flips {mv['ehr_label_disagreements']}"
                      f" · unsup flips {mv['unsupported_label_disagreements']}"
                      f" · mean |Δ| {mv['unsupported_mean_abs_score_delta']}")
            continue
        print(f"    {k}")
        print(f"      EHR label flips {v['ehr_label_disagreements']}/{v['n']}"
              f" · unsupported label flips {v['unsupported_label_disagreements']}/{v['n']}"
              f" · mean |Δ unsupported score| {v['unsupported_mean_abs_score_delta']}")
    print(f"\n  {SAMPLE_CSV}")
    print(f"  {HTML_OUT}")


if __name__ == "__main__":
    main()
