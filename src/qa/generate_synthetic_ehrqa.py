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

def load_mkg_facts() -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, List[str]]]:
    """Load CONTRAINDICATED_WITH and FIRST_LINE_TREATMENT edges from the
    hand-curated ontology CSV (mkg/edges/ontology_edges.csv) — the same file
    src/mkg/neo4j_loader.py loads into Neo4j, read directly here so QA
    generation doesn't require a running Neo4j instance."""
    path = Path("mkg/edges/ontology_edges.csv")
    contraindicated: Dict[str, List[Tuple[str, str]]] = {}
    first_line: Dict[str, List[str]] = {}
    if not path.exists():
        print(f"[WARN] {path} not found — contraindication_check questions will be skipped.")
        return contraindicated, first_line

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        disease = str(row["disease"]).strip()
        drug = str(row["target"]).strip()
        if row["edge_type"] == "CONTRAINDICATED_WITH":
            note = str(row["notes"]).strip() if pd.notna(row.get("notes")) else ""
            contraindicated.setdefault(disease, []).append((drug, note))
        elif row["edge_type"] == "FIRST_LINE_TREATMENT":
            first_line.setdefault(disease, []).append(drug)
    return contraindicated, first_line


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


def make_contraindication_qa(
    hadm_id: int,
    disease: str,
    contraindicated: Dict[str, List[Tuple[str, str]]],
    first_line: Dict[str, List[str]],
    n_labs: int, n_diag: int, n_meds: int,
    sparsity_score: Optional[float], sparsity_bucket: str,
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
        "n_labs":          n_labs,
        "n_diag":          n_diag,
        "n_meds":          n_meds,
        "sparsity_score":  sparsity_score,
        "sparsity_bucket": sparsity_bucket,
    }


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
    modes = list(TEMPLATES.keys())
    contraindicated_edges, first_line_edges = load_mkg_facts()
    max_kg_qa_per_admission = 2
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

        for mode in modes:
            rec = make_qa(
                hadm_id, mode, diagnoses, labs, medications, age, gender,
                n_labs=struct_features["n_labs"],
                n_diag=struct_features["n_diag"],
                n_meds=struct_features["n_meds"],
                sparsity_score=struct_features["sparsity_score"],
                sparsity_bucket=struct_features["sparsity_bucket"],
            )
            key  = (hadm_id, rec["question"], rec["answer"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            records.append(rec)

            if len(records) % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                print(f"[INFO] {split_name}: {len(records)}/{target} QA pairs | "
                      f"admissions scanned: {scanned} | {elapsed:.1f}s")

            if len(records) >= target:
                break

        # KG-grounded contraindication_check questions (see load_mkg_facts /
        # make_contraindication_qa docstrings) — only generated for
        # admissions whose diagnoses match a seed disease with ontology
        # coverage, so this is a bonus type layered on top of, not a
        # replacement for, the template types above.
        if len(records) < target and (contraindicated_edges or first_line_edges):
            icd_pairs = get_diagnosis_codes(con, schema, hadm_id)
            matched_diseases = match_seed_diseases(icd_pairs)
            for disease in matched_diseases[:max_kg_qa_per_admission]:
                rec = make_contraindication_qa(
                    hadm_id, disease, contraindicated_edges, first_line_edges,
                    n_labs=struct_features["n_labs"],
                    n_diag=struct_features["n_diag"],
                    n_meds=struct_features["n_meds"],
                    sparsity_score=struct_features["sparsity_score"],
                    sparsity_bucket=struct_features["sparsity_bucket"],
                )
                if rec is None:
                    continue
                key = (hadm_id, rec["question"], rec["answer"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                records.append(rec)
                if len(records) >= target:
                    break

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