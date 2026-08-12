"""
src/lakehouse/sparsity.py

Computes EHR sparsity scores for every admission in the QA splits.

Sparsity score S = alpha_1 * I(n_labs < tau_labs)
                 + alpha_2 * I(n_diag < tau_diag)
                 + alpha_3 * I(d_note > tau_days)

Buckets:
    high   → S >= 2
    medium → S == 1
    low    → S == 0

Outputs:
    data/lakehouse/sparsity.parquet

Usage:
    python src/lakehouse/sparsity.py
    python src/lakehouse/sparsity.py --tau-labs 10 --tau-diag 5 --tau-days 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import pandas as pd

# ── Defaults ──────────────────────────────────────────────────────────────────
LAKE        = Path("data/lakehouse")
QA_DIR      = Path("data/qa")
OUT_PARQUET = LAKE / "sparsity.parquet"

TAU_LABS = 5
TAU_DIAG = 3
TAU_DAYS = 7

ALPHA_1 = 1.0
ALPHA_2 = 1.0
ALPHA_3 = 1.0

BUCKET_HIGH   = 2
BUCKET_MEDIUM = 1


# ── DuckDB ────────────────────────────────────────────────────────────────────

def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=4;")
    return con


def create_views(con: duckdb.DuckDBPyConnection) -> dict[str, bool]:
    required = {
        "admissions": LAKE / "admissions.parquet",
        "labs":       LAKE / "labs.parquet",
        "diagnoses":  LAKE / "diagnoses.parquet",
    }
    optional = {
        "notes": LAKE / "notes.parquet",
    }

    missing = [n for n, p in required.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required lakehouse files: {missing}")

    for name, path in required.items():
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_parquet('{path.as_posix()}')"
        )

    flags: dict[str, bool] = {}
    for name, path in optional.items():
        if path.exists():
            con.execute(
                f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM read_parquet('{path.as_posix()}')"
            )
            flags[name] = True
        else:
            flags[name] = False
    return flags


# ── Schema detection ──────────────────────────────────────────────────────────

def detect_cols(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]


def pick(cols: list[str], candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def detect_schema(con: duckdb.DuckDBPyConnection, has_notes: bool) -> dict:
    ac = detect_cols(con, "admissions")
    lc = detect_cols(con, "labs")
    dc = detect_cols(con, "diagnoses")

    schema: dict = {
        "admissions": {
            "hadm_id":   pick(ac, ["hadm_id", "hadmid"]),
            "admittime": pick(ac, ["admittime", "admit_time"]),
            "dischtime": pick(ac, ["dischtime", "disch_time", "dischargetime"]),
        },
        "labs": {
            "hadm_id":   pick(lc, ["hadm_id", "hadmid"]),
            "itemid":    pick(lc, ["itemid"]),
            "charttime": pick(lc, ["charttime", "chart_time"]),
            "valuenum":  pick(lc, ["valuenum"]),
        },
        "diagnoses": {
            "hadm_id":  pick(dc, ["hadm_id", "hadmid"]),
            "icd_code": pick(dc, ["icd_code", "icdcode"]),
        },
    }

    if has_notes:
        nc = detect_cols(con, "notes")
        schema["notes"] = {
            "hadm_id":   pick(nc, ["hadm_id", "hadmid"]),
            "chartdate": pick(nc, ["chartdate", "chart_date", "charttime"]),
        }

    return schema


# ── Component SQL builders ────────────────────────────────────────────────────

def sql_n_labs(schema: dict, ids_view: str) -> str:
    l = schema["labs"]
    if not l["hadm_id"] or not l["itemid"]:
        return f"SELECT hadm_id, 0 AS n_labs FROM {ids_view}"

    val_filter = f"AND l.{l['valuenum']} IS NOT NULL" if l["valuenum"] else ""
    return f"""
    SELECT h.hadm_id, COUNT(DISTINCT l.{l['itemid']}) AS n_labs
    FROM {ids_view} h
    LEFT JOIN labs l ON h.hadm_id = l.{l['hadm_id']}
    WHERE 1=1 {val_filter}
    GROUP BY h.hadm_id
    """


def sql_n_diag(schema: dict, ids_view: str) -> str:
    d = schema["diagnoses"]
    if not d["hadm_id"] or not d["icd_code"]:
        return f"SELECT hadm_id, 0 AS n_diag FROM {ids_view}"

    return f"""
    SELECT h.hadm_id, COUNT(DISTINCT d.{d['icd_code']}) AS n_diag
    FROM {ids_view} h
    LEFT JOIN diagnoses d ON h.hadm_id = d.{d['hadm_id']}
    GROUP BY h.hadm_id
    """


def sql_d_note(schema: dict, ids_view: str, has_notes: bool) -> str:
    """
    d_note = days between the LAST clinical note and DISCHARGE.

    BUG FIX (2026-08-12): this previously measured from ADMISSION
    (`admittime - last_note`). Clinical notes are written during and after
    the admission, so that expression was almost always NEGATIVE, and the
    `d_note > tau_days` sparsity term could then only ever fire on the 999
    "no notes at all" sentinel. Measured on the real data: of 266
    admissions, 111 were exactly 999, 150 were negative, and **zero** fell
    anywhere in between — i.e. the third sparsity component had silently
    degenerated into a has-notes/no-notes binary rather than the
    note-staleness measure the design doc specifies ("days since last
    clinical note before question/discharge time"). Since sparsity_bucket
    is the independent variable for H2, this materially affected the
    headline hypothesis. See RESEARCH_LOG.md, 2026-08-12.

    Now measured from dischtime (falling back to admittime only if the
    column is absent) and clamped at 0, so d_note is a genuine
    non-negative staleness value: 0 = documented right up to discharge,
    larger = documentation stopped earlier in the stay, 999 = no notes.
    """
    if not has_notes:
        return f"SELECT hadm_id, 999.0 AS d_note FROM {ids_view}"

    n = schema["notes"]
    a = schema["admissions"]
    n_hadm      = n.get("hadm_id")
    n_chartdate = n.get("chartdate")
    a_hadm      = a.get("hadm_id")
    a_ref       = a.get("dischtime") or a.get("admittime")

    if not n_hadm or not n_chartdate:
        return f"SELECT hadm_id, 999.0 AS d_note FROM {ids_view}"

    # ANY_VALUE so DuckDB does not require the column in GROUP BY.
    ref_expr = (
        f"CAST(ANY_VALUE(a.{a_ref}) AS DATE)" if a_ref else "CURRENT_DATE"
    )

    # NOTE the explicit NULL test rather than COALESCE(GREATEST(...), 999):
    # DuckDB's GREATEST(NULL, 0) evaluates to 0, so wrapping the clamp in
    # COALESCE silently destroyed the 999 "no notes at all" sentinel and
    # collapsed every admission to d_note = 0 (high-sparsity bucket fell from
    # 20.3% to 4.5%). Test for the NULL first, clamp second.
    return f"""
    SELECT
        h.hadm_id,
        CASE
            WHEN MAX(CAST(n.{n_chartdate} AS DATE)) IS NULL THEN 999
            ELSE GREATEST(
                DATEDIFF('day',
                    MAX(CAST(n.{n_chartdate} AS DATE)),
                    {ref_expr}
                ),
                0
            )
        END AS d_note
    FROM {ids_view} h
    LEFT JOIN notes n       ON h.hadm_id = n.{n_hadm}
    LEFT JOIN admissions a  ON h.hadm_id = a.{a_hadm}
    GROUP BY h.hadm_id
    """


# ── Main computation ──────────────────────────────────────────────────────────

def compute_sparsity(
    tau_labs: int   = TAU_LABS,
    tau_diag: int   = TAU_DIAG,
    tau_days: int   = TAU_DAYS,
    alpha_1:  float = ALPHA_1,
    alpha_2:  float = ALPHA_2,
    alpha_3:  float = ALPHA_3,
) -> pd.DataFrame:
    t0 = time.time()

    print("[INFO] Connecting to DuckDB...")
    con = connect()

    print("[INFO] Creating Parquet views...")
    flags     = create_views(con)
    has_notes = flags.get("notes", False)
    print(f"[INFO] notes table available: {has_notes}")

    print("[INFO] Detecting schema...")
    schema = detect_schema(con, has_notes)

    # ── Collect hadm_ids from all QA splits ───────────────────────────────────
    print("[INFO] Collecting hadm_ids from QA splits...")
    all_ids: set[int] = set()
    qa_files = sorted(QA_DIR.glob("ehrqa_*.parquet"))

    if not qa_files:
        raise FileNotFoundError(
            f"No QA parquet files found in {QA_DIR.resolve()}. "
            "Run generate_synthetic_ehrqa_split.py first."
        )

    for f in qa_files:
        df_qa = pd.read_parquet(f, columns=["hadm_id"])
        all_ids.update(df_qa["hadm_id"].dropna().astype(int).tolist())
        print(f"[INFO]   {f.name}: {len(df_qa)} rows")

    print(f"[INFO] Total unique hadm_ids: {len(all_ids)}")

    # Register as DuckDB view
    ids_df = pd.DataFrame({"hadm_id": sorted(all_ids)})
    con.register("_target_ids_df", ids_df)
    con.execute(
        "CREATE OR REPLACE VIEW target_ids AS "
        "SELECT hadm_id FROM _target_ids_df"
    )

    # ── Build component SQL ───────────────────────────────────────────────────
    print("[INFO] Computing n_labs...")
    q_labs = sql_n_labs(schema, "target_ids")

    print("[INFO] Computing n_diag...")
    q_diag = sql_n_diag(schema, "target_ids")

    print("[INFO] Computing d_note...")
    q_dnote = sql_d_note(schema, "target_ids", has_notes)

    # ── Assemble final query ──────────────────────────────────────────────────
    print("[INFO] Assembling sparsity scores and buckets...")

    score_expr = f"""
        {alpha_1} * CASE WHEN COALESCE(nl.n_labs, 0) < {tau_labs}  THEN 1.0 ELSE 0.0 END
      + {alpha_2} * CASE WHEN COALESCE(nd.n_diag, 0) < {tau_diag}  THEN 1.0 ELSE 0.0 END
      + {alpha_3} * CASE WHEN COALESCE(dn.d_note, 999) > {tau_days} THEN 1.0 ELSE 0.0 END
    """

    final_sql = f"""
    WITH
        nlabs AS ({q_labs}),
        ndiag AS ({q_diag}),
        dnote AS ({q_dnote})
    SELECT
        t.hadm_id,

        COALESCE(nl.n_labs, 0)   AS n_labs,
        COALESCE(nd.n_diag, 0)   AS n_diag,
        COALESCE(dn.d_note, 999) AS d_note,

        CASE WHEN COALESCE(nl.n_labs, 0)   < {tau_labs}  THEN 1 ELSE 0 END AS sparse_labs,
        CASE WHEN COALESCE(nd.n_diag, 0)   < {tau_diag}  THEN 1 ELSE 0 END AS sparse_diag,
        CASE WHEN COALESCE(dn.d_note, 999) > {tau_days}  THEN 1 ELSE 0 END AS sparse_note,

        ROUND({score_expr}, 2) AS sparsity_score,

        CASE
            WHEN ({score_expr}) >= {BUCKET_HIGH}   THEN 'high'
            WHEN ({score_expr}) >= {BUCKET_MEDIUM} THEN 'medium'
            ELSE 'low'
        END AS sparsity_bucket,

        {tau_labs} AS tau_labs_used,
        {tau_diag} AS tau_diag_used,
        {tau_days} AS tau_days_used

    FROM target_ids t
    LEFT JOIN nlabs nl ON t.hadm_id = nl.hadm_id
    LEFT JOIN ndiag nd ON t.hadm_id = nd.hadm_id
    LEFT JOIN dnote dn ON t.hadm_id = dn.hadm_id
    ORDER BY t.hadm_id
    """

    df = con.execute(final_sql).df()
    elapsed = time.time() - t0
    print(f"[INFO] Sparsity computed for {len(df)} admissions in {elapsed:.1f}s")
    return df


# ── Summary + save ────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "═" * 60)
    print("SPARSITY SUMMARY")
    print("═" * 60)

    counts = df["sparsity_bucket"].value_counts()
    total  = len(df)

    for bucket in ["high", "medium", "low"]:
        n   = counts.get(bucket, 0)
        pct = 100 * n / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket:8s}: {n:5d} ({pct:5.1f}%)  {bar}")

    tau_l = df["tau_labs_used"].iloc[0]
    tau_d = df["tau_diag_used"].iloc[0]
    tau_n = df["tau_days_used"].iloc[0]

    print(f"\n  n_labs  → mean: {df['n_labs'].mean():.1f} | "
          f"median: {df['n_labs'].median():.0f} | "
          f"< tau({tau_l}): {(df['n_labs'] < tau_l).sum()}")
    print(f"  n_diag  → mean: {df['n_diag'].mean():.1f} | "
          f"median: {df['n_diag'].median():.0f} | "
          f"< tau({tau_d}): {(df['n_diag'] < tau_d).sum()}")
    print(f"  d_note  → mean: {df['d_note'].mean():.1f} | "
          f"median: {df['d_note'].median():.0f} | "
          f"> tau({tau_n}): {(df['d_note'] > tau_n).sum()}")
    print(f"\n  Thresholds: tau_labs={tau_l}, tau_diag={tau_d}, tau_days={tau_n}")
    print(f"  Output: {OUT_PARQUET.resolve()}")
    print("═" * 60)

    for bucket in ["high", "medium", "low"]:
        n   = counts.get(bucket, 0)
        pct = 100 * n / total if total > 0 else 0
        if pct < 10:
            print(f"\n  ⚠️  WARNING: '{bucket}' bucket only {pct:.1f}% — consider adjusting thresholds.")
            print(f"     Re-run: python src/lakehouse/sparsity.py --tau-labs X --tau-diag Y --tau-days Z")


def save(df: pd.DataFrame) -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[INFO] Saved: {OUT_PARQUET}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute EHR sparsity scores for all QA admissions."
    )
    parser.add_argument("--tau-labs", type=int,   default=TAU_LABS)
    parser.add_argument("--tau-diag", type=int,   default=TAU_DIAG)
    parser.add_argument("--tau-days", type=int,   default=TAU_DAYS)
    parser.add_argument("--alpha1",   type=float, default=ALPHA_1)
    parser.add_argument("--alpha2",   type=float, default=ALPHA_2)
    parser.add_argument("--alpha3",   type=float, default=ALPHA_3)
    args = parser.parse_args()

    df = compute_sparsity(
        tau_labs=args.tau_labs,
        tau_diag=args.tau_diag,
        tau_days=args.tau_days,
        alpha_1=args.alpha1,
        alpha_2=args.alpha2,
        alpha_3=args.alpha3,
    )

    print_summary(df)
    save(df)

    print("\n[INFO] Sample rows:")
    print(df[["hadm_id", "n_labs", "n_diag", "d_note",
              "sparsity_score", "sparsity_bucket"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()