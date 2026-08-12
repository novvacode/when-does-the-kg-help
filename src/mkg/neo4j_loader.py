"""
neo4j_loader.py — Load MKG nodes and edges into Neo4j.

Builds four node types (Disease, Symptom, LabTest, Drug) and loads:
  1. Ontology-based edges (HAS_SYMPTOM, INDICATES_LAB, FIRST_LINE_TREATMENT, CONTRAINDICATED_WITH)
  2. EHR co-occurrence edges (CO_OCCURS_WITH_LAB)

Uses MERGE throughout so the script is idempotent -- safe to re-run.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mkg.seed_diseases import SEED_DISEASES

# Read from the environment so no credential is committed. See README setup.
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")

ONTOLOGY_EDGES_PATH = Path("mkg/edges/ontology_edges.csv")
COOCCURRENCE_EDGES_PATH = Path("mkg/edges/cooccurrence_edges.csv")


def clear_graph(session):
    session.run("MATCH (n) DETACH DELETE n")


def load_disease_nodes(session):
    for d in SEED_DISEASES:
        session.run(
            "MERGE (d:Disease {name: $name})",
            name=d["name"]
        )


def load_ontology_edges(session, df: pd.DataFrame):
    edge_type_map = {
        "HAS_SYMPTOM": ("Symptom", "HAS_SYMPTOM"),
        "INDICATES_LAB": ("LabTest", "INDICATES_LAB"),
        "FIRST_LINE_TREATMENT": ("Drug", "FIRST_LINE_TREATMENT"),
        "CONTRAINDICATED_WITH": ("Drug", "CONTRAINDICATED_WITH"),
    }

    for _, row in df.iterrows():
        target_label, rel_type = edge_type_map[row["edge_type"]]
        notes = row["notes"] if pd.notna(row["notes"]) else ""

        query = f"""
            MERGE (d:Disease {{name: $disease}})
            MERGE (t:{target_label} {{name: $target}})
            MERGE (d)-[r:{rel_type}]->(t)
            SET r.notes = $notes, r.source = 'ontology'
        """
        session.run(query, disease=row["disease"], target=row["target"], notes=notes)


def load_cooccurrence_edges(session, df: pd.DataFrame):
    for _, row in df.iterrows():
        query = """
            MERGE (d:Disease {name: $disease})
            MERGE (l:LabTest {name: $lab_test})
            MERGE (d)-[r:CO_OCCURS_WITH_LAB]->(l)
            SET r.frequency = $frequency,
                r.n_admissions_with_lab = $n_admissions,
                r.total_admissions = $total_admissions,
                r.source = 'ehr_cooccurrence'
        """
        session.run(
            query,
            disease=row["disease"],
            lab_test=row["lab_test"],
            frequency=float(row["frequency"]),
            n_admissions=int(row["n_admissions_with_lab"]),
            total_admissions=int(row["total_admissions"]),
        )


def get_graph_stats(session):
    node_counts = session.run("""
        MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count
    """).data()
    edge_counts = session.run("""
        MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS count
    """).data()
    return node_counts, edge_counts


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    ontology_df = pd.read_csv(ONTOLOGY_EDGES_PATH)
    cooccurrence_df = pd.read_csv(COOCCURRENCE_EDGES_PATH)

    with driver.session() as session:
        print("Clearing existing graph...")
        clear_graph(session)

        print("Loading disease nodes...")
        load_disease_nodes(session)

        print(f"Loading {len(ontology_df)} ontology edges...")
        load_ontology_edges(session, ontology_df)

        print(f"Loading {len(cooccurrence_df)} co-occurrence edges...")
        load_cooccurrence_edges(session, cooccurrence_df)

        print("\n--- Graph Stats ---")
        node_counts, edge_counts = get_graph_stats(session)
        print("Nodes:")
        for row in node_counts:
            print(f"  {row['label']:12s}: {row['count']}")
        print("Edges:")
        for row in edge_counts:
            print(f"  {row['rel_type']:24s}: {row['count']}")

    driver.close()
    print("\nNeo4j load complete.")


if __name__ == "__main__":
    main()