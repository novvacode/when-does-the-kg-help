"""
cooccurrence.py — Compute disease-lab co-occurrence edges from MIMIC-IV.

For each seed disease, find admissions where that disease was diagnosed,
then compute what fraction of those admissions also had each lab test.
Add an edge disease -> lab_test (CO_OCCURS_WITH_LAB) if the co-occurrence
frequency exceeds 5% of admissions for that disease (Section 7.3 spec).
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lakehouse.query import get_connection
from mkg.seed_diseases import SEED_DISEASES

COOCCURRENCE_THRESHOLD = 0.05  # >5% of admissions
TOP_N_LABS_PER_DISEASE = 15    # cap to keep graph focused (raised from 10 as part
                                # of the 2026-08-06 MKG expansion — see RESEARCH_LOG.md)


def find_disease_hadm_ids(con, icd9_prefix: str, icd10_prefix: str) -> list:
    """Find all hadm_ids where diagnosis ICD code matches disease prefix."""
    q = f"""
        SELECT DISTINCT hadm_id
        FROM diagnoses
        WHERE icd_code LIKE '{icd9_prefix}%' OR icd_code LIKE '{icd10_prefix}%'
    """
    return con.execute(q).fetchdf()["hadm_id"].tolist()


def compute_cooccurrence_for_disease(con, disease_name: str, hadm_ids: list) -> pd.DataFrame:
    """For a set of admission IDs, compute lab test frequency across those admissions."""
    if not hadm_ids:
        return pd.DataFrame()

    n_admissions = len(hadm_ids)
    ids_str = ",".join(str(int(h)) for h in hadm_ids)

    q = f"""
        SELECT li.label AS lab_test, COUNT(DISTINCT l.hadm_id) AS n_admissions_with_lab
        FROM labs l
        LEFT JOIN d_labitems li ON l.itemid = li.itemid
        WHERE l.hadm_id IN ({ids_str}) AND li.label IS NOT NULL
        GROUP BY li.label
        ORDER BY n_admissions_with_lab DESC
    """
    df = con.execute(q).fetchdf()
    df["disease"] = disease_name
    df["total_admissions"] = n_admissions
    df["frequency"] = df["n_admissions_with_lab"] / n_admissions
    return df[df["frequency"] > COOCCURRENCE_THRESHOLD].head(TOP_N_LABS_PER_DISEASE)


def main():
    con = get_connection()
    all_edges = []

    for disease in SEED_DISEASES:
        name = disease["name"]
        hadm_ids = find_disease_hadm_ids(con, disease["icd9_prefix"], disease["icd10_prefix"])
        print(f"{name}: {len(hadm_ids)} admissions found")

        if len(hadm_ids) < 5:
            print(f"  -> Skipping (too few admissions for reliable stats)")
            continue

        cooc_df = compute_cooccurrence_for_disease(con, name, hadm_ids)
        all_edges.append(cooc_df)

    result = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    result = result[["disease", "lab_test", "n_admissions_with_lab", "total_admissions", "frequency"]]
    result = result.sort_values(["disease", "frequency"], ascending=[True, False])

    out_path = Path("mkg/edges/cooccurrence_edges.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"\nTotal co-occurrence edges: {len(result)}")
    print(f"Saved to {out_path}")
    print("\nSample:")
    print(result.head(15))


if __name__ == "__main__":
    main()