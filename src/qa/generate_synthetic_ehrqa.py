"""
generate_synthetic_ehrqa_split.py
================

Split-safe synthetic EHR-QA generation.

Reads splits/patient_splits.json to enforce strict patient-level separation
between fine-tune, router-train, router-val, and held-out eval sets.

Writes:
    data/qa/ehrqa_finetune.parquet      ← for SLM QLoRA fine-tuning
    data/qa/ehrqa_router_train.parquet  ← for router pseudo-label generation
    data/qa/ehrqa_router_val.parquet    ← for router hyperparameter tuning
    data/qa/ehrqa_eval.parquet          ← HELD-OUT: never used for training

NEVER cross patient boundaries. The split JSON is the source of truth.

Design principles:
- All joins and filtering stay inside DuckDB SQL.
- Only per-admission result rows enter Python memory.
- Optional lookup tables (d_icd_diagnoses, d_labitems) auto-detected.
- Schema-robust column detection handles hadm_id/hadmid variants.
- A single PatientSnapshot instance is reused across all admissions to
  compute structural routing features (n_labs, n_diag, n_meds,
  sparsity_score, sparsity_bucket) with no duplicate DB queries.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd

from src.lakehouse.patient_snapshot import PatientSnapshot

# ── Config ────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

LAKE        = Path("data/lakehouse")
SPLITS_FILE = Path("splits/patient_splits.json")
OUT_DIR     = Path("data/qa")

# Target QA pairs per split (adjust as needed)
SPLIT_TARGETS = {
    "finetune":     1000,   # 60% of patients → ~1000 QA pairs for SLM training
    "router_train":  200,   # 15% of patients → exactly 200 for router labels
    "router_val":    100,   # 10% of patients → exactly 100 for router tuning
    "eval":          300,   # 15% of patients → 300–500 for final evaluation
}

# Keys to try when loading splits JSON (handles naming variants)
SPLIT_KEY_MAP = {
    "finetune":     ["finetune_train", "finetune", "fine_tune", "train", "sft"],
    "router_train": ["router_train", "router-train", "rtrain"],
    "router_val":   ["router_val", "router-val", "rval", "val"],
    "eval":         ["held_out_eval", "eval", "evaluation", "held_out", "test"],
}

PROGRESS_EVERY = 100

# ══════════════════════════════════════════════════════════════════════════════
# DISEASE-LEVEL SPLIT for KG-derived questions (2026-08-12 redesign)
# ══════════════════════════════════════════════════════════════════════════════
# The 2026-08-11 analysis showed the fine-tuned model had MEMORISED KG facts:
# on contraindication questions, mode T (no EHR, no KG) scored 62.5% exactly
# correct, because ~180 KG-derived QA pairs were in the fine-tuning split and
# disease->drug facts are patient-independent — so the patient-level split in
# splits/patient_splits.json gave no protection at all.
#
# Fix: split the SEED_DISEASES themselves. KG questions in the fine-tuning
# split may only use FINETUNE_DISEASES; KG questions in router/eval splits may
# only use HELDOUT_DISEASES. The model therefore learns the ANSWER FORMAT for
# guideline questions (so it isn't penalised on ROUGE/BERTScore for phrasing)
# while never seeing the specific facts it is later tested on. Any KG benefit
# measured at evaluation time is then genuine retrieval, not recall.
#
# Deterministic split (sorted + seeded) so it is reproducible and auditable.
#
# Clinically-related diseases are split as a FAMILY, never individually.
# Splitting CKD Stage 3/4 into fine-tune while leaving Stage 5 held-out would
# leak: the stages share contraindications (Metformin, NSAIDs, Nitrofurantoin),
# so a model trained on stage 3/4 could generalise the same facts to stage 5
# and the "held-out" evaluation would be contaminated by near-duplicates.
DISEASE_FAMILIES = [
    # renal failure spectrum — shared contraindication profile
    ["Chronic Kidney Disease Stage 3", "Chronic Kidney Disease Stage 4",
     "Chronic Kidney Disease Stage 5", "Acute Kidney Injury"],
    # thrombo-embolic / anticoagulation
    ["Deep Vein Thrombosis", "Pulmonary Embolism", "Atrial Fibrillation"],
    # hepatic
    ["Cirrhosis of Liver", "Chronic Hepatitis C"],
    # upper-GI acid disease
    ["Gastroesophageal Reflux Disease", "Peptic Ulcer Disease"],
    # cardiac / metabolic-vascular
    ["Congestive Heart Failure", "Coronary Artery Disease", "Hyperlipidemia"],
    # infection
    ["Pneumonia", "Sepsis", "Urinary Tract Infection"],
    # psychiatric
    ["Major Depressive Disorder", "Bipolar Disorder"],
]


def _split_seed_diseases() -> tuple[set, set]:
    """Deterministic (sorted + seeded) family-aware 50/50 split of the seed
    diseases into a fine-tuning set and a held-out set."""
    from src.mkg.seed_diseases import SEED_DISEASES
    names = sorted(d["name"] for d in SEED_DISEASES)

    # Build family groups; any disease not in an explicit family is its own.
    grouped = {n for fam in DISEASE_FAMILIES for n in fam}
    units: list[list[str]] = [sorted(f) for f in DISEASE_FAMILIES]
    units += [[n] for n in names if n not in grouped]
    units.sort(key=lambda u: (-len(u), u[0]))  # deterministic order

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(units)

    # Greedy balance by disease count so neither side is starved.
    finetune: set = set()
    heldout: set = set()
    for unit in units:
        if len(finetune) <= len(heldout):
            finetune.update(unit)
        else:
            heldout.update(unit)

    # Families must never straddle the split.
    for fam in DISEASE_FAMILIES:
        present = [d for d in fam if d in names]
        if present:
            in_ft = [d for d in present if d in finetune]
            assert not in_ft or len(in_ft) == len(present), (
                f"Disease family split across sets — leakage risk: {fam}")
    return finetune, heldout


FINETUNE_DISEASES, HELDOUT_DISEASES = _split_seed_diseases()


def diseases_allowed_for(split_name: str) -> set:
    """KG questions in the fine-tune split use a disjoint disease set from
    those in router_train / router_val / eval."""
    return FINETUNE_DISEASES if split_name == "finetune" else HELDOUT_DISEASES


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION MIX per split (2026-08-12 redesign)
# ══════════════════════════════════════════════════════════════════════════════
# The 2026-08-12 router audit found the learned router was beaten by a trivial
# question_type -> mode lookup table (val acc 0.97 vs 0.87), because 6 of 7
# question types were extractive templates whose gold answer is a verbatim EHR
# field. That saturates T+E (composite exactly 1.0000 on three types), makes
# 93% of T+E/T+E+K pairs quality-identical, and leaves the optimal mode a
# function of question TYPE with no dependence on the PATIENT — so there was
# no adaptive signal for any router to learn.
#
# The redesign adds PATIENT-DEPENDENT question types (see
# make_monitoring_labs_qa / make_expected_symptoms_qa) whose optimal mode
# genuinely varies across patients for the SAME question, and rebalances the
# mix so those types are a large share of the router/eval splits.
#
# Fine-tuning keeps a majority of extractive types (it needs to learn to read
# context and produce the answer format), but the router/eval splits are
# weighted toward the patient-dependent types that actually exercise routing.
EXTRACTIVE_TYPES = ["primary_diagnosis", "diagnoses", "lab", "medication",
                    "summary", "next_step"]
KG_DEPENDENT_TYPES = ["contraindication_check", "monitoring_labs",
                      "expected_symptoms"]

QUESTION_MIX = {
    # split -> (n extractive per admission, n KG-dependent per admission)
    "finetune":     (6, 2),
    "router_train": (3, 4),
    "router_val":   (3, 4),
    "eval":         (3, 4),
}

TEMPLATES: Dict[str, List[str]] = {
    "primary_diagnosis": [
        "What is the primary diagnosis for this patient?",
        "What condition was this admission mainly for?",
        "What was the most likely main diagnosis?",
    ],
    "diagnoses": [
        "What conditions were diagnosed during this admission?",
        "List the main diagnoses recorded for this patient.",
    ],
    "lab": [
        "What is the most abnormal lab value for this patient and what does it indicate?",
        "Which lab abnormality is most concerning in this admission?",
    ],
    "medication": [
        "What medications were prescribed during this admission?",
        "Which discharge medications are relevant for the patient's condition?",
    ],
    "summary": [
        "Provide a brief clinical summary of this patient case.",
        "Summarize the main issues for this admission in one or two sentences.",
    ],
    "next_step": [
        "What is the recommended next clinical step for this patient?",
        "What follow-up plan would be appropriate after discharge?",
    ],
}


# ── Splits loading ─────────────────────────────────────────────────────────────

def load_splits() -> Dict[str, List[int]]:
    """
    Load patient_splits.json and normalise to canonical split names.
    Handles multiple naming conventions.
    Returns dict: {split_name: [subject_id, ...]}
    """
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(
            f"Patient splits file not found: {SPLITS_FILE.resolve()}\n"
            "Run src/lakehouse/create_patient_splits.py first."
        )

    with open(SPLITS_FILE, "r") as f:
        raw = json.load(f)

    print(f"[INFO] Raw split keys found: {list(raw.keys())}")

    splits: Dict[str, List[int]] = {}
    for canonical, candidates in SPLIT_KEY_MAP.items():
        found = None
        for key in candidates:
            if key in raw:
                found = key
                break
        if found is None:
            print(f"[WARN] Split '{canonical}' not found under any key: {candidates}. Skipping.")
            splits[canonical] = []
        else:
            ids = raw[found]
            # Handle both {"subject_ids": [...]} nested format and flat list format
            if isinstance(ids, dict):
                ids = ids.get("subject_ids", ids.get("ids", []))
            splits[canonical] = [int(x) for x in ids]
            print(f"[INFO] Split '{canonical}' ← key '{found}': {len(splits[canonical])} patients")

    return splits


# ── DuckDB setup ──────────────────────────────────────────────────────────────

def connect_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4;")
    return con


def parquet_sql(path: Path) -> str:
    return f"read_parquet('{path.as_posix()}')"


def create_views(con: duckdb.DuckDBPyConnection) -> Dict[str, bool]:
    """Create DuckDB views over Parquet. Returns optional table availability flags."""
    required = {
        "patients":    LAKE / "patients.parquet",
        "admissions":  LAKE / "admissions.parquet",
        "diagnoses":   LAKE / "diagnoses.parquet",
        "labs":        LAKE / "labs.parquet",
        "medications": LAKE / "medications.parquet",
    }
    optional = {
        "d_icd_diagnoses": LAKE / "d_icd_diagnoses.parquet",
        "d_labitems":      LAKE / "d_labitems.parquet",
    }

    missing = [n for n, p in required.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required lakehouse files: {missing}")

    for name, path in required.items():
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM {parquet_sql(path)}")

    available: Dict[str, bool] = {}
    for name, path in optional.items():
        if path.exists():
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM {parquet_sql(path)}")
            available[name] = True
        else:
            available[name] = False

    return available


# ── Schema detection ──────────────────────────────────────────────────────────

def detect_schema(con: duckdb.DuckDBPyConnection, table: str) -> List[str]:
    return [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]


def pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def build_schema_map(con: duckdb.DuckDBPyConnection) -> Dict[str, Dict[str, Optional[str]]]:
    pc = detect_schema(con, "patients")
    ac = detect_schema(con, "admissions")
    dc = detect_schema(con, "diagnoses")
    lc = detect_schema(con, "labs")
    mc = detect_schema(con, "medications")

    schema: Dict[str, Dict[str, Optional[str]]] = {
        "patients": {
            "subject_id": pick_col(pc, ["subject_id", "subjectid"]),
            "gender":     pick_col(pc, ["gender"]),
            "anchor_age": pick_col(pc, ["anchor_age", "anchorage", "age"]),
        },
        "admissions": {
            "hadm_id":    pick_col(ac, ["hadm_id", "hadmid"]),
            "subject_id": pick_col(ac, ["subject_id", "subjectid"]),
        },
        "diagnoses": {
            "hadm_id":     pick_col(dc, ["hadm_id", "hadmid"]),
            "icd_code":    pick_col(dc, ["icd_code", "icdcode"]),
            "icd_version": pick_col(dc, ["icd_version", "icdversion"]),
            "seq_num":     pick_col(dc, ["seq_num", "seqnum"]),
        },
        "labs": {
            "hadm_id":   pick_col(lc, ["hadm_id", "hadmid"]),
            "itemid":    pick_col(lc, ["itemid"]),
            "charttime": pick_col(lc, ["charttime"]),
            "valuenum":  pick_col(lc, ["valuenum"]),
            "valueuom":  pick_col(lc, ["valueuom"]),
            "flag":      pick_col(lc, ["flag"]),
        },
        "medications": {
            "hadm_id": pick_col(mc, ["hadm_id", "hadmid"]),
            "drug":    pick_col(mc, ["drug"]),
        },
    }

    # Optional lookup schemas
    views = {r[0].lower() for r in con.execute("SHOW TABLES").fetchall()}

    if "d_icd_diagnoses" in views:
        ic = detect_schema(con, "d_icd_diagnoses")
        schema["d_icd_diagnoses"] = {
            "icd_code":    pick_col(ic, ["icd_code", "icdcode"]),
            "icd_version": pick_col(ic, ["icd_version", "icdversion"]),
            "long_title":  pick_col(ic, ["long_title", "longtitle", "title"]),
        }
    else:
        schema["d_icd_diagnoses"] = {"icd_code": None, "icd_version": None, "long_title": None}

    if "d_labitems" in views:
        li = detect_schema(con, "d_labitems")
        schema["d_labitems"] = {
            "itemid": pick_col(li, ["itemid"]),
            "label":  pick_col(li, ["label"]),
        }
    else:
        schema["d_labitems"] = {"itemid": None, "label": None}

    return schema


# ── Admission fetching (split-restricted) ─────────────────────────────────────

def get_admissions_for_patients(
    con: duckdb.DuckDBPyConnection,
    schema: Dict[str, Dict[str, Optional[str]]],
    subject_ids: List[int],
) -> List[Tuple[int, int]]:
    """
    Return (subject_id, hadm_id) pairs for a given list of subject_ids.
    Keeps everything in DuckDB — only the ID pairs come to Python.
    """
    if not subject_ids:
        return []

    a_hadm    = schema["admissions"]["hadm_id"]
    a_subject = schema["admissions"]["subject_id"]

    if not a_hadm or not a_subject:
        raise ValueError("Admissions table missing hadm_id or subject_id column.")

    # Write subject_ids to a temp table for the IN filter
    ids_df = pd.DataFrame({"subject_id": subject_ids})
    con.register("_split_subjects", ids_df)

    query = f"""
    SELECT a.{a_subject} AS subject_id, a.{a_hadm} AS hadm_id
    FROM admissions a
    INNER JOIN _split_subjects s ON a.{a_subject} = s.subject_id
    WHERE a.{a_hadm} IS NOT NULL
    ORDER BY random()
    """
    rows = con.execute(query).fetchall()
    con.execute("DROP VIEW IF EXISTS _split_subjects")
    return [(int(r[0]), int(r[1])) for r in rows]


# ── Per-admission extractors (same as production version) ─────────────────────

def get_age_gender(
    con: duckdb.DuckDBPyConnection,
    schema: Dict[str, Dict[str, Optional[str]]],
    hadm_id: int,
) -> Tuple[Optional[int], Optional[str]]:
    a_hadm    = schema["admissions"]["hadm_id"]
    a_subject = schema["admissions"]["subject_id"]
    p_subject = schema["patients"]["subject_id"]
    p_age     = schema["patients"]["anchor_age"]
    p_gender  = schema["patients"]["gender"]

    if not all([a_hadm, a_subject, p_subject, p_age, p_gender]):
        return None, None

    row = con.execute(f"""
        SELECT p.{p_age}, p.{p_gender}
        FROM admissions a
        JOIN patients p ON a.{a_subject} = p.{p_subject}
        WHERE a.{a_hadm} = ?
        LIMIT 1
    """, [hadm_id]).fetchone()

    if row is None:
        return None, None

    age = int(row[0]) if row[0] is not None else None
    g = str(row[1]).strip().lower() if row[1] else None
    gender = "male" if g == "m" else "female" if g == "f" else g
    return age, gender


def get_top_diagnoses(
    con: duckdb.DuckDBPyConnection,
    schema: Dict[str, Dict[str, Optional[str]]],
    hadm_id: int,
    limit: int = 3,
) -> List[str]:
    d       = schema["diagnoses"]
    d_hadm  = d["hadm_id"]
    d_icd   = d["icd_code"]
    d_ver   = d["icd_version"]
    d_seq   = d["seq_num"]
    di      = schema["d_icd_diagnoses"]
    has_lkp = di["icd_code"] and di["long_title"]

    if not d_hadm or not d_icd:
        return []

    if has_lkp:
        join_on = (f"d.{d_icd} = di.{di['icd_code']} AND d.{d_ver} = di.{di['icd_version']}"
                   if d_ver and di["icd_version"]
                   else f"d.{d_icd} = di.{di['icd_code']}")
        title = f"COALESCE(di.{di['long_title']}, CAST(d.{d_icd} AS VARCHAR))"
        query = f"""
        SELECT {title} AS name
        FROM diagnoses d
        LEFT JOIN d_icd_diagnoses di ON {join_on}
        WHERE d.{d_hadm} = ?
        ORDER BY d.{d_seq} ASC NULLS LAST
        LIMIT ?
        """
    else:
        query = f"""
        SELECT CAST(d.{d_icd} AS VARCHAR) AS name
        FROM diagnoses d
        WHERE d.{d_hadm} = ?
        ORDER BY d.{d_seq} ASC NULLS LAST
        LIMIT ?
        """

    rows = con.execute(query, [hadm_id, limit]).fetchall()
    return [str(r[0]).strip() for r in rows if r[0] and str(r[0]).strip()]


def get_diagnosis_codes(
    con: duckdb.DuckDBPyConnection,
    schema: Dict[str, Dict[str, Optional[str]]],
    hadm_id: int,
    limit: int = 10,
) -> List[Tuple[str, str]]:
    """Raw (icd_code, icd_version) pairs for MKG seed-disease prefix matching
    (see load_mkg_facts / match_seed_diseases below) — get_top_diagnoses()
    returns human-readable titles, not codes, so this is a separate query."""
    d = schema["diagnoses"]
    d_hadm, d_icd, d_ver, d_seq = d["hadm_id"], d["icd_code"], d["icd_version"], d["seq_num"]
    if not d_hadm or not d_icd:
        return []
    ver_expr = f"CAST(d.{d_ver} AS VARCHAR)" if d_ver else "'10'"
    query = f"""
        SELECT CAST(d.{d_icd} AS VARCHAR) AS icd_code, {ver_expr} AS icd_version
        FROM diagnoses d
        WHERE d.{d_hadm} = ?
        ORDER BY d.{d_seq} ASC NULLS LAST
        LIMIT ?
    """
    rows = con.execute(query, [hadm_id, limit]).fetchall()
    return [(str(r[0]).strip(), str(r[1]).strip()) for r in rows if r[0]]


# ── MKG-grounded reasoning questions ───────────────────────────────────────────
#
# The template question types above (primary_diagnosis, diagnoses, lab,
# medication) derive their reference answer directly from the same
# structured fields (diagnoses[0], labs[:3], medications[:5]) that are also
# shown verbatim in the T+E/T+E+K prompt's EHR snapshot. That makes those
# comparisons partly mechanical (the T mode's prompt doesn't contain the
# EHR snapshot at all, so it structurally can't produce the literal answer,
# while T+E/T+E+K can — regardless of any genuine reasoning benefit).
# See RESEARCH_LOG.md, 2026-08-06 audit, finding #5.
#
# contraindication_check questions below are designed so the correct answer
# is NOT recoverable from the EHR snapshot alone: knowing a patient has
# "Chronic Kidney Disease Stage 3" does not by itself tell you Metformin is
# contraindicated at that stage — that fact only lives in the MKG. This
# gives the T vs. T+E vs. T+E+K comparison at least one question type where
# T+E+K's advantage (if any) reflects genuine use of KG facts, not context
# containing the literal answer string.

def load_mkg_facts() -> Dict[str, Dict[str, list]]:
    """Load the hand-curated ontology edges (mkg/edges/ontology_edges.csv —
    the same file src/mkg/neo4j_loader.py loads into Neo4j, read directly so
    QA generation doesn't require a running Neo4j instance).

    Returns {edge_type: {disease: [...]}} for the four edge types used to
    build KG-dependent questions.
    """
    path = Path("mkg/edges/ontology_edges.csv")
    facts: Dict[str, Dict[str, list]] = {
        "CONTRAINDICATED_WITH": {},
        "FIRST_LINE_TREATMENT": {},
        "INDICATES_LAB": {},
        "HAS_SYMPTOM": {},
    }
    if not path.exists():
        print(f"[WARN] {path} not found — KG-dependent questions will be skipped.")
        return facts

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        etype = str(row["edge_type"]).strip()
        if etype not in facts:
            continue
        disease = str(row["disease"]).strip()
        target = str(row["target"]).strip()
        if etype == "CONTRAINDICATED_WITH":
            note = str(row["notes"]).strip() if pd.notna(row.get("notes")) else ""
            facts[etype].setdefault(disease, []).append((target, note))
        else:
            facts[etype].setdefault(disease, []).append(target)
    return facts


def match_seed_diseases(icd_pairs: List[Tuple[str, str]]) -> List[str]:
    """Match an admission's (icd_code, icd_version) pairs against
    SEED_DISEASES' icd9/icd10 prefixes — the same prefix-matching approach
    src/mkg/cooccurrence.py uses to build the co-occurrence edges, applied
    per-admission instead of per-disease-cohort."""
    from src.mkg.seed_diseases import SEED_DISEASES

    matched: List[str] = []
    for icd_code, icd_version in icd_pairs:
        code = icd_code.upper()
        for d in SEED_DISEASES:
            if code.startswith(d["icd9_prefix"]) or code.startswith(d["icd10_prefix"]):
                if d["name"] not in matched:
                    matched.append(d["name"])
    return matched


def make_monitoring_labs_qa(
    hadm_id: int,
    disease: str,
    facts: Dict[str, Dict[str, list]],
    patient_lab_names: List[str],
    struct: Dict,
) -> Optional[Dict]:
    """PATIENT-DEPENDENT question type — the core of the 2026-08-12 redesign
    and the direct operationalisation of H2.

    Question: "Which laboratory tests should be monitored for this patient's
    {disease}?"  Gold answer: the guideline-recommended labs for that disease
    (KG INDICATES_LAB edges).

    Why the optimal retrieval mode varies BY PATIENT for this same question:

      * If the patient's EHR already contains those labs, the T+E snapshot
        shows them and T+E can answer without any KG access.
      * If the patient's EHR does NOT contain them (sparse EHR), the answer
        exists only in the knowledge graph, so T+E+K is required.

    Crucially the gold answer does NOT depend on which mode can reach it —
    it is the clinically-expected monitoring panel either way. So this type
    creates genuine per-patient variation in the best mode, which is exactly
    the adaptive signal the previous template-only QA design lacked (see
    RESEARCH_LOG.md, 2026-08-12 router audit).

    `ehr_covers_answer` is recorded so H2 can be analysed directly rather
    than inferred: it is the ground-truth indicator of whether the EHR alone
    was sufficient for this patient.
    """
    labs = facts["INDICATES_LAB"].get(disease, [])
    if not labs:
        return None

    patient_labs_lower = {l.lower() for l in patient_lab_names}
    covered = [l for l in labs if any(l.lower() in pl or pl in l.lower()
                                       for pl in patient_labs_lower)]
    # Continuous coverage fraction is stored alongside the binary flag so H2
    # can be analysed on a graded variable rather than a hard threshold.
    coverage = len(covered) / len(labs) if labs else 0.0
    # "EHR alone is sufficient" = the guideline panel is essentially all
    # present in this patient's recorded labs. 0.8 rather than 1.0 so a
    # single unusual assay (e.g. eGFR recorded as a derived value) does not
    # flip an otherwise fully-covered patient to "needs KG".
    ehr_covers = coverage >= 0.8

    question = f"Which laboratory tests should be monitored for this patient's {disease}?"
    answer = ", ".join(labs) + "."
    return {
        "hadm_id":            hadm_id,
        "question_type":      "monitoring_labs",
        "question":           question,
        "answer":             answer,
        "age":                None,
        "gender":             None,
        "diagnoses":          disease,
        "labs":               "",
        "medications":        "",
        "source":             "synthetic_ehrqa_kg",
        "kg_disease":         disease,
        "ehr_covers_answer":  int(ehr_covers),
        "ehr_lab_coverage":   round(coverage, 3),
        "n_labs":             struct["n_labs"],
        "n_diag":             struct["n_diag"],
        "n_meds":             struct["n_meds"],
        "sparsity_score":     struct["sparsity_score"],
        "sparsity_bucket":    struct["sparsity_bucket"],
    }


def make_expected_symptoms_qa(
    hadm_id: int,
    disease: str,
    facts: Dict[str, Dict[str, list]],
    struct: Dict,
) -> Optional[Dict]:
    """Second patient-dependent KG type. Same logic as monitoring_labs: the
    expected symptom set for a disease is guideline knowledge, but for a
    patient whose notes already describe those symptoms, text retrieval (T)
    or the EHR snapshot may suffice, whereas for a patient with thin
    documentation only the KG supplies them."""
    symptoms = facts["HAS_SYMPTOM"].get(disease, [])
    if not symptoms:
        return None
    question = f"What symptoms are typically associated with this patient's {disease}?"
    answer = ", ".join(symptoms) + "."
    return {
        "hadm_id":           hadm_id,
        "question_type":     "expected_symptoms",
        "question":          question,
        "answer":            answer,
        "age":               None,
        "gender":            None,
        "diagnoses":         disease,
        "labs":              "",
        "medications":       "",
        "source":            "synthetic_ehrqa_kg",
        "kg_disease":        disease,
        "ehr_covers_answer": 0,
        "ehr_lab_coverage":  0.0,
        "n_labs":            struct["n_labs"],
        "n_diag":            struct["n_diag"],
        "n_meds":            struct["n_meds"],
        "sparsity_score":    struct["sparsity_score"],
        "sparsity_bucket":   struct["sparsity_bucket"],
    }


def make_contraindication_qa(
    hadm_id: int,
    disease: str,
    contraindicated: Dict[str, List[Tuple[str, str]]],
    first_line: Dict[str, List[str]],
    n_labs: int, n_diag: int, n_meds: int,
    sparsity_score: Optional[float], sparsity_bucket: str,
    kg_disease: Optional[str] = None,
) -> Optional[Dict]:
    """Build one contraindication-check QA pair for a matched seed disease.
    Roughly half the time asks about a genuinely contraindicated drug
    (answer: unsafe), half the time about a first-line drug for the same
    disease (answer: safe) — so the question set isn't trivially answerable
    by always saying "contraindicated"."""
    bad_options = contraindicated.get(disease, [])
    good_options = first_line.get(disease, [])
    if not bad_options and not good_options:
        return None

    ask_unsafe = bool(bad_options) and (not good_options or random.random() < 0.5)
    if ask_unsafe:
        drug, note = random.choice(bad_options)
        reason = f" ({note})" if note else ""
        answer = f"No, {drug} is contraindicated in {disease}{reason}."
    else:
        drug = random.choice(good_options)
        answer = f"Yes, {drug} is a standard first-line treatment for {disease}."

    question = f"Would prescribing {drug} be appropriate for this patient's {disease}?"
    return {
        "hadm_id":         hadm_id,
        "question_type":   "contraindication_check",
        "question":        question,
        "answer":          answer,
        "age":             None,
        "gender":          None,
        "diagnoses":       disease,
        "labs":            "",
        "medications":     "",
        "source":          "synthetic_ehrqa_kg",
        "kg_disease":        kg_disease or disease,
        "ehr_covers_answer": 0,
        "ehr_lab_coverage":  0.0,
        "n_labs":          n_labs,
        "n_diag":          n_diag,
        "n_meds":          n_meds,
        "sparsity_score":  sparsity_score,
        "sparsity_bucket": sparsity_bucket,
    }


def get_all_lab_names(
    con: duckdb.DuckDBPyConnection,
    schema: Dict[str, Dict[str, Optional[str]]],
    hadm_id: int,
) -> List[str]:
    """Every DISTINCT lab name recorded for the admission.

    Needed by make_monitoring_labs_qa to decide whether the EHR already
    covers the guideline panel. An earlier version reused
    get_abnormal_labs(limit=3), which returns only the top three ABNORMAL
    labs — against a mean of ~30 distinct labs per admission — so
    `ehr_covers_answer` came out 0 for 100% of rows and the question type
    produced no patient-dependent routing variation at all. See
    RESEARCH_LOG.md, 2026-08-12 regeneration check.
    """
    l = schema["labs"]
    li = schema["d_labitems"]
    if not l["hadm_id"] or not l["itemid"]:
        return []
    has_lkp = li["itemid"] and li["label"]
    label_expr = (f"COALESCE(li.{li['label']}, CAST(l.{l['itemid']} AS VARCHAR))"
                  if has_lkp else f"CAST(l.{l['itemid']} AS VARCHAR)")
    join_clause = (f"LEFT JOIN d_labitems li ON l.{l['itemid']} = li.{li['itemid']}"
                   if has_lkp else "")
    rows = con.execute(f"""
        SELECT DISTINCT {label_expr} AS label
        FROM labs l
        {join_clause}
        WHERE l.{l['hadm_id']} = ?
          AND {label_expr} IS NOT NULL
    """, [hadm_id]).fetchall()
    return [str(r[0]).strip() for r in rows if r[0] and str(r[0]).strip()]


def get_abnormal_labs(
    con: duckdb.DuckDBPyConnection,
    schema: Dict[str, Dict[str, Optional[str]]],
    hadm_id: int,
    limit: int = 3,
) -> List[str]:
    l       = schema["labs"]
    l_hadm  = l["hadm_id"]
    l_item  = l["itemid"]
    l_time  = l["charttime"]
    l_val   = l["valuenum"]
    l_uom   = l["valueuom"]
    l_flag  = l["flag"]
    li      = schema["d_labitems"]
    has_lkp = li["itemid"] and li["label"]

    if not l_hadm or not l_item or not l_val:
        return []

    label_expr = (f"COALESCE(li.{li['label']}, CAST(l.{l_item} AS VARCHAR))"
                  if has_lkp else f"CAST(l.{l_item} AS VARCHAR)")
    join_clause = (f"LEFT JOIN d_labitems li ON l.{l_item} = li.{li['itemid']}"
                   if has_lkp else "")
    time_expr = f"l.{l_time}" if l_time else "NULL"
    flag_expr = f"LOWER(COALESCE(CAST(l.{l_flag} AS VARCHAR), ''))" if l_flag else "''"
    uom_expr  = f"COALESCE(CAST(l.{l_uom} AS VARCHAR), '')"         if l_uom  else "''"

    query = f"""
    WITH ranked AS (
        SELECT
            {label_expr}   AS label,
            l.{l_val}      AS valuenum,
            {uom_expr}     AS valueuom,
            {flag_expr}    AS flag_text,
            ROW_NUMBER() OVER (
                PARTITION BY {label_expr}
                ORDER BY
                    CASE WHEN {flag_expr} = 'abnormal' THEN 0 ELSE 1 END,
                    {time_expr} DESC NULLS LAST
            ) AS rn
        FROM labs l
        {join_clause}
        WHERE l.{l_hadm} = ?
          AND l.{l_val} IS NOT NULL
    )
    SELECT label, valuenum, valueuom, flag_text
    FROM ranked
    WHERE rn = 1
      AND label IS NOT NULL
      AND LENGTH(TRIM(CAST(label AS VARCHAR))) > 0
    ORDER BY CASE WHEN flag_text = 'abnormal' THEN 0 ELSE 1 END, label ASC
    LIMIT ?
    """
    rows = con.execute(query, [hadm_id, limit]).fetchall()
    results = []
    for label, valuenum, valueuom, flag_text in rows:
        if label is None or valuenum is None:
            continue
        flag = "abnormal" if str(flag_text).strip().lower() == "abnormal" else "normal"
        unit = str(valueuom).strip() if valueuom else ""
        results.append(f"{str(label).strip()} {valuenum}{unit} ({flag})")
    return results


def get_medications(
    con: duckdb.DuckDBPyConnection,
    schema: Dict[str, Dict[str, Optional[str]]],
    hadm_id: int,
    limit: int = 5,
) -> List[str]:
    m      = schema["medications"]
    m_hadm = m["hadm_id"]
    m_drug = m["drug"]

    if not m_hadm or not m_drug:
        return []

    rows = con.execute(f"""
        SELECT DISTINCT CAST({m_drug} AS VARCHAR) AS drug_name
        FROM medications
        WHERE {m_hadm} = ?
          AND {m_drug} IS NOT NULL
          AND LENGTH(TRIM(CAST({m_drug} AS VARCHAR))) > 0
        ORDER BY drug_name
        LIMIT ?
    """, [hadm_id, limit]).fetchall()
    return [str(r[0]).strip() for r in rows if r[0]]


# ── Structural routing features (PatientSnapshot) ─────────────────────────────

def get_structural_features(
    snapshot_api: PatientSnapshot,
    hadm_id: int,
    diagnoses: List[str],
    labs: List[str],
    medications: List[str],
) -> Dict:
    """
    Single PatientSnapshot lookup per admission to derive structural
    routing features: n_labs, n_diag, n_meds, sparsity_score, sparsity_bucket.

    On any failure, falls back to counts derived from the already-fetched
    diagnoses/labs/medications lists so generation never crashes.
    """
    try:
        snapshot = snapshot_api.get(hadm_id)
        n_labs = len(snapshot.get("labs", []))
        n_diag = len(snapshot.get("diagnoses", []))
        n_meds = len(snapshot.get("medications", []))
        sparsity_score = snapshot.get("sparsity_score", None)
        sparsity_bucket = snapshot.get("sparsity_bucket", "unknown")
        
        return {
            "n_labs": n_labs,
            "n_diag": n_diag,
            "n_meds": n_meds,
            "sparsity_score": sparsity_score,
            "sparsity_bucket": sparsity_bucket,
        }
    except (KeyError, TypeError, ValueError) as exc:
        logging.warning(f"[WARN] PatientSnapshot lookup failed for hadm_id={hadm_id}: {exc}. "
              f"Falling back to extracted counts.")
        return {
            "n_labs": len(labs),
            "n_diag": len(diagnoses),
            "n_meds": len(medications),
            "sparsity_score": None,
            "sparsity_bucket": "unknown",
        }


# ── QA generation ─────────────────────────────────────────────────────────────

def make_qa(
    hadm_id: int,
    mode: str,
    diagnoses: List[str],
    labs: List[str],
    medications: List[str],
    age: Optional[int],
    gender: Optional[str],
    n_labs: int,
    n_diag: int,
    n_meds: int,
    sparsity_score: Optional[float],
    sparsity_bucket: str,
) -> Dict:
    question = random.choice(TEMPLATES[mode])

    if mode == "primary_diagnosis":
        answer = diagnoses[0] if diagnoses else "Unknown"
    elif mode == "diagnoses":
        answer = "; ".join(diagnoses[:3]) if diagnoses else "No diagnoses available"
    elif mode == "lab":
        answer = "; ".join(labs[:3]) if labs else "No abnormal labs available"
    elif mode == "medication":
        answer = "; ".join(medications[:5]) if medications else "No medications found"
    elif mode == "summary":
        parts = []
        if age and gender:
            parts.append(f"{age}-year-old {gender}")
        if diagnoses:
            parts.append(f"with {diagnoses[0]}")
        if labs:
            parts.append(f"notable labs: {labs[0]}")
        if medications:
            parts.append(f"medications: {medications[0]}")
        answer = ", ".join(parts) if parts else "Clinical admission with limited structured detail."
    elif mode == "next_step":
        answer = (f"Continue outpatient follow-up and monitor: {diagnoses[0]}."
                  if diagnoses else "Continue outpatient follow-up and reassess.")
    else:
        answer = "Unknown"

    return {
        "hadm_id":         hadm_id,
        "question_type": mode,
        "question":      question,
        "answer":        answer,
        "age":           age,
        "gender":        gender,
        "diagnoses":     " | ".join(diagnoses[:3]),
        "labs":          " | ".join(labs[:3]),
        "medications":   " | ".join(medications[:5]),
        "source":        "synthetic_ehrqa",
        # ── New structural routing features ──────────────────────────────
        "n_labs":           n_labs,
        "n_diag":           n_diag,
        "n_meds":           n_meds,
        "sparsity_score":   sparsity_score,
        "sparsity_bucket":  sparsity_bucket,
    }


# ── Per-split generation ───────────────────────────────────────────────────────

def generate_for_split(
    con: duckdb.DuckDBPyConnection,
    schema: Dict[str, Dict[str, Optional[str]]],
    snapshot_api: PatientSnapshot,
    split_name: str,
    subject_ids: List[int],
    target: int,
) -> pd.DataFrame:
    if not subject_ids:
        print(f"[WARN] Split '{split_name}' has no patients — skipping.")
        return pd.DataFrame()

    print(f"\n[INFO] ── Generating split: {split_name} ──────────────────")
    print(f"[INFO] Patients: {len(subject_ids)} | Target QA pairs: {target}")

    admissions = get_admissions_for_patients(con, schema, subject_ids)
    print(f"[INFO] Admissions found: {len(admissions)}")

    if not admissions:
        print(f"[WARN] No admissions found for split '{split_name}'.")
        return pd.DataFrame()

    records: List[Dict] = []
    seen_keys: set = set()
    scanned = 0
    facts = load_mkg_facts()
    contraindicated_edges = facts["CONTRAINDICATED_WITH"]
    first_line_edges = facts["FIRST_LINE_TREATMENT"]

    # Disease-level split: KG questions in the fine-tune split draw from a
    # disjoint disease set to the router/eval splits, so the model cannot
    # memorise the specific facts it is later evaluated on. See
    # diseases_allowed_for() and RESEARCH_LOG.md 2026-08-12.
    allowed_diseases = diseases_allowed_for(split_name)
    n_extractive, n_kg = QUESTION_MIX.get(split_name, (6, 2))
    modes = EXTRACTIVE_TYPES[:]
    print(f"[INFO] question mix for '{split_name}': "
          f"{n_extractive} extractive + up to {n_kg} KG-dependent per admission")
    print(f"[INFO] KG questions restricted to {len(allowed_diseases)} diseases "
          f"({'FINETUNE' if split_name == 'finetune' else 'HELD-OUT'} disease set)")
    t0 = time.time()

    for subject_id, hadm_id in admissions:
        scanned += 1
        diagnoses   = get_top_diagnoses(con, schema, hadm_id)
        labs        = get_abnormal_labs(con, schema, hadm_id)
        medications = get_medications(con, schema, hadm_id)
        age, gender = get_age_gender(con, schema, hadm_id)

        if not diagnoses and not labs and not medications:
            continue

        # Single PatientSnapshot lookup per admission — reused across all
        # question types generated for this hadm_id (no duplicate queries).
        struct_features = get_structural_features(
            snapshot_api, hadm_id, diagnoses, labs, medications
        )

        def _add(rec) -> bool:
            """Append a QA record if new; return True when the split target
            has been reached."""
            if rec is None:
                return len(records) >= target
            key = (hadm_id, rec["question"], rec["answer"])
            if key in seen_keys:
                return len(records) >= target
            seen_keys.add(key)
            # Every record must carry the same columns, otherwise the parquet
            # schema differs between extractive and KG rows.
            rec.setdefault("kg_disease", "")
            rec.setdefault("ehr_covers_answer", 0)
            rec.setdefault("ehr_lab_coverage", 0.0)
            records.append(rec)
            if len(records) % PROGRESS_EVERY == 0:
                print(f"[INFO] {split_name}: {len(records)}/{target} QA pairs | "
                      f"admissions scanned: {scanned} | {time.time()-t0:.1f}s")
            return len(records) >= target

        # ── extractive template questions ────────────────────────────────
        for mode in modes[:n_extractive]:
            rec = make_qa(
                hadm_id, mode, diagnoses, labs, medications, age, gender,
                n_labs=struct_features["n_labs"],
                n_diag=struct_features["n_diag"],
                n_meds=struct_features["n_meds"],
                sparsity_score=struct_features["sparsity_score"],
                sparsity_bucket=struct_features["sparsity_bucket"],
            )
            if _add(rec):
                break

        # ── KG-dependent questions (disease-split enforced) ───────────────
        if len(records) < target:
            icd_pairs = get_diagnosis_codes(con, schema, hadm_id)
            matched = [d for d in match_seed_diseases(icd_pairs) if d in allowed_diseases]
            patient_lab_names = get_all_lab_names(con, schema, hadm_id)

            kg_added = 0
            for disease in matched:
                if kg_added >= n_kg or len(records) >= target:
                    break
                # monitoring_labs and expected_symptoms are the patient-
                # dependent types that create real routing variation;
                # contraindication_check is the guideline-reasoning type.
                for builder in (
                    lambda d: make_monitoring_labs_qa(
                        hadm_id, d, facts, patient_lab_names, struct_features),
                    lambda d: make_contraindication_qa(
                        hadm_id, d, contraindicated_edges, first_line_edges,
                        n_labs=struct_features["n_labs"],
                        n_diag=struct_features["n_diag"],
                        n_meds=struct_features["n_meds"],
                        sparsity_score=struct_features["sparsity_score"],
                        sparsity_bucket=struct_features["sparsity_bucket"],
                        kg_disease=d),
                    lambda d: make_expected_symptoms_qa(
                        hadm_id, d, facts, struct_features),
                ):
                    if kg_added >= n_kg or len(records) >= target:
                        break
                    rec = builder(disease)
                    if rec is not None:
                        before = len(records)
                        _add(rec)
                        if len(records) > before:
                            kg_added += 1

        if len(records) >= target:
            break

    df = pd.DataFrame(records)
    elapsed = time.time() - t0
    print(f"[INFO] {split_name} done: {len(df)} QA pairs | "
          f"admissions scanned: {scanned} | {elapsed:.1f}s")
    return df


# ── Save + summary ─────────────────────────────────────────────────────────────

def save_split(df: pd.DataFrame, split_name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = OUT_DIR / f"ehrqa_{split_name}.parquet"
    out_csv     = OUT_DIR / f"ehrqa_{split_name}.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved: {out_parquet}  ({len(df)} rows)")


def verify_no_leakage(splits: Dict[str, List[int]]) -> None:
    """Hard check: no patient appears in more than one split."""
    all_sets = {name: set(ids) for name, ids in splits.items() if ids}
    names    = list(all_sets.keys())
    clean    = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = all_sets[names[i]] & all_sets[names[j]]
            if overlap:
                print(f"[ERROR] LEAKAGE DETECTED: {names[i]} ∩ {names[j]} = "
                      f"{len(overlap)} patients!")
                clean = False
    if clean:
        print("[INFO] ✅ No patient leakage detected across splits.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()

    print("[INFO] Loading patient splits...")
    splits = load_splits()
    verify_no_leakage(splits)

    print("\n[INFO] Connecting to DuckDB and creating Parquet views...")
    con = connect_duckdb()
    optional_flags = create_views(con)
    print(f"[INFO] d_icd_diagnoses: {optional_flags['d_icd_diagnoses']} | "
          f"d_labitems: {optional_flags['d_labitems']}")

    print("[INFO] Detecting schema...")
    schema = build_schema_map(con)

    # ── Single PatientSnapshot instance for the whole generation process ──
    # Created once here and reused across all splits/admissions; never
    # instantiated per-admission. Closed at the end in a finally block.
    print("[INFO] Initialising PatientSnapshot API (single instance)...")
    snapshot_api = PatientSnapshot()

    try:
        results: Dict[str, pd.DataFrame] = {}
        for split_name, target in SPLIT_TARGETS.items():
            subject_ids = splits.get(split_name, [])
            df = generate_for_split(con, schema, snapshot_api, split_name, subject_ids, target)
            if not df.empty:
                save_split(df, split_name)
                results[split_name] = df

        # ── Final summary ──────────────────────────────────────────────────
        total_elapsed = time.time() - t_start
        print("\n" + "═" * 60)
        print("SPLIT-SAFE GENERATION COMPLETE")
        print("═" * 60)
        for split_name, df in results.items():
            qtypes = df["question_type"].value_counts().to_dict()
            print(f"  {split_name:15s} | {len(df):5d} QA pairs | {qtypes}")
        print(f"\nTotal time: {total_elapsed:.1f}s")
        print("Output dir:", OUT_DIR.resolve())
        print("\n⚠️  REMINDER: ehrqa_eval.parquet is HELD-OUT.")
        print("   Do NOT use it for training, router labelling, or tuning.")
    finally:
        # Ensure the PatientSnapshot resource is always released.
        close_fn = getattr(snapshot_api, "close", None)
        if callable(close_fn):
            close_fn()
            print("[INFO] PatientSnapshot closed.")


if __name__ == "__main__":
    main()