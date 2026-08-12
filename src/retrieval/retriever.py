"""
src/retrieval/retriever.py

Retrieval Layer — implements all 3 retrieval modes.

    T     → Text-only RAG (FAISS over clinical notes)
    T+E   → Text + structured EHR snapshot
    T+E+K → Text + EHR snapshot + MKG subgraph facts
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.lakehouse.patient_snapshot import PatientSnapshot

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDINGS_DIR = Path("embeddings")
INDEX_FILE     = EMBEDDINGS_DIR / "notes_index.faiss"
CHUNKS_FILE    = EMBEDDINGS_DIR / "notes_chunks.parquet"

DEFAULT_MODEL  = "BAAI/bge-small-en-v1.5"
TOP_K          = 5       # chunks retrieved per query
MAX_KG_FACTS   = 15      # max MKG triples per query


# ── Mode enum ─────────────────────────────────────────────────────────────────

class Mode(str, Enum):
    T   = "T"     # Text-only
    TE  = "T+E"   # Text + EHR
    TEK = "T+E+K" # Text + EHR + KG


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    mode:             Mode
    question:         str
    hadm_id:          Optional[int]

    retrieved_chunks: list[dict]  = field(default_factory=list)
    ehr_snapshot:     str         = ""
    kg_facts:         list[str]   = field(default_factory=list)

    latency_ms:       float       = 0.0
    n_tokens_approx:  int         = 0

    @property
    def prompt_context(self) -> str:
        """
        Assemble the full context string to inject into the LLM prompt.
        Format varies by mode.
        """
        parts: list[str] = []

        if self.retrieved_chunks:
            passage_lines = []
            for i, chunk in enumerate(self.retrieved_chunks, 1):
                text = chunk.get("text", "").strip()
                src  = chunk.get("category", "note")
                passage_lines.append(f"[Passage {i} | {src}]\n{text}")
            parts.append("## Retrieved Clinical Passages\n" +
                         "\n\n".join(passage_lines))

        if self.ehr_snapshot:
            parts.append("## Patient EHR Snapshot\n" + self.ehr_snapshot)

        if self.kg_facts:
            facts_str = "\n".join(f"- {f}" for f in self.kg_facts)
            parts.append("## Relevant Medical Knowledge\n" + facts_str)

        return "\n\n".join(parts) if parts else "No context available."

    @property
    def stats(self) -> dict:
        return {
            "mode":            self.mode.value,
            "n_chunks":        len(self.retrieved_chunks),
            "has_ehr":         bool(self.ehr_snapshot),
            "n_kg_facts":      len(self.kg_facts),
            "latency_ms":      round(self.latency_ms, 1),
            "n_tokens_approx": self.n_tokens_approx,
        }


# ── Retriever class ───────────────────────────────────────────────────────────

class Retriever:
    """
    Unified retriever implementing all 3 modes.
    Keeps FAISS index and embedding model loaded in memory across calls.
    """

    def __init__(
        self,
        model_name:    str  = DEFAULT_MODEL,
        top_k:         int  = TOP_K,
        max_kg_facts:  int  = MAX_KG_FACTS,
        kg_module=None,     # Injected Neo4j retrieval module
    ) -> None:
        self.top_k        = top_k
        self.max_kg_facts = max_kg_facts
        self._kg          = kg_module

        self._model      = self._load_model(model_name)
        self._index      = self._load_index()
        self._df_chunks  = self._load_chunks()
        self._snapshot   = PatientSnapshot()

        self._model_name = model_name
        use_prefix       = "bge" in model_name.lower()
        self._q_prefix   = ("Represent this clinical question for retrieval: "
                            if use_prefix else "")

        print(f"[Retriever] Ready — index: {self._index.ntotal:,} vectors, "
              f"top_k={top_k}, KG={'enabled' if kg_module else 'disabled'}")

    @staticmethod
    def _load_model(model_name: str) -> SentenceTransformer:
        # MEDRAG_EMBED_DEVICE lets a caller force the query encoder onto CPU.
        # Motivation: run_evaluation.py keeps 4-bit MedGemma AND this encoder
        # resident on the GPU simultaneously, whereas oracle_labels.py reads
        # pre-built contexts from parquet and loads no encoder at all — and
        # oracle_labels completed 300 questions cleanly (3.58 GB peak) while
        # run_evaluation hit "CUDA error: an illegal memory access" twice.
        # Moving this small encoder to CPU is therefore the cheapest lever on
        # the one structural difference between the two scripts. Query
        # encoding is a single short string, so CPU cost is minor.
        # NOTE: CPU vs GPU embeddings differ by ~1e-6, which can in principle
        # flip a near-tied FAISS neighbour. Do not mix devices within one
        # evaluation run — start such a run from a clean checkpoint.
        override = os.environ.get("MEDRAG_EMBED_DEVICE", "").strip().lower()
        if override in {"cpu", "cuda"}:
            device = override
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Retriever] Loading embedding model: {model_name} on {device}")
        return SentenceTransformer(model_name, device=device)

    @staticmethod
    def _load_index() -> faiss.Index:
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_FILE.resolve()}\n"
                "Run: python src/retrieval/embedder.py"
            )
        index = faiss.read_index(str(INDEX_FILE))
        print(f"[Retriever] FAISS index loaded: {index.ntotal:,} vectors, dim={index.d}")
        return index

    @staticmethod
    def _load_chunks() -> pd.DataFrame:
        if not CHUNKS_FILE.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found: {CHUNKS_FILE.resolve()}\n"
                "Run: python src/retrieval/embedder.py"
            )
        df = pd.read_parquet(CHUNKS_FILE)
        print(f"[Retriever] Chunk metadata loaded: {len(df):,} rows")
        return df

    def _retrieve_text(self, question: str) -> list[dict]:
        query = self._q_prefix + question
        vec   = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        D, I = self._index.search(vec, k=self.top_k)

        chunks: list[dict] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._df_chunks):
                continue
            row = self._df_chunks.iloc[idx]
            chunks.append({
                "text":      row.get("text", ""),
                "score":     float(score),
                "hadm_id":   row.get("hadm_id"),
                "category":  row.get("category", "note"),
                "chartdate": row.get("chartdate"),
                "chunk_id":  int(row.get("chunk_id", idx)),
            })
        return chunks

    def _retrieve_ehr(self, hadm_id: int) -> str:
        snap = self._snapshot.get(hadm_id)
        return snap.get("snapshot_text", "")

    def _retrieve_kg(self, question: str, hadm_id: int) -> list[str]:
        """
        Retrieve relevant MKG facts using the imported Phase 2 module.
        """
        if self._kg is None:
            return []

        try:
            snap = self._snapshot.get(hadm_id)
            raw_dx = snap.get("diagnoses", [])
            
            # Robust extraction of diagnosis strings
            dx_texts = []
            if isinstance(raw_dx, str):
                for x in raw_dx.replace(";", "|").split("|"):
                    if x.strip():
                        dx_texts.append(x.strip())
            else:
                for d in raw_dx:
                    if isinstance(d, dict):
                        # Priority: description -> long_title -> name -> icd_code
                        text = (d.get("description") or 
                                d.get("long_title") or 
                                d.get("name") or 
                                d.get("icd_code") or "")
                        
                        if text and str(text).strip():
                            dx_texts.append(str(text).strip())
                    else:
                        if str(d).strip():
                            dx_texts.append(str(d).strip())
                            
            # The QUESTION is itself a retrieval signal and was previously
            # ignored: this method accepted `question` but matched only on
            # the patient's ICD diagnosis strings. Measured consequence —
            # 18/74 (24.3%) of KG-dependent questions retrieved ZERO KG
            # facts, because a diagnosis rendered as e.g. "Hypertensive
            # chronic kidney disease with stage 1 through stage 4..." does
            # not clear find_matching_diseases()'s 0.6 token-overlap bar
            # against the node "Essential Hypertension" — even though the
            # question explicitly names that disease. All 18 are recovered
            # by also matching on the question text. Diagnoses stay first so
            # patient context keeps priority; the question is an additional
            # candidate, not a replacement. See RESEARCH_LOG.md 2026-08-13.
            dx_texts = dx_texts + [str(question)]

            logger.debug(f"RAW SNAPSHOT DIAGNOSES: {raw_dx}")
            logger.debug(f"DX TEXTS SENT TO KG (incl. question): {dx_texts}")

            # Try to log matched diseases by looking up the driver locally
            if hasattr(self._kg, 'get_driver') and hasattr(self._kg, 'find_matching_diseases'):
                try:
                    driver = self._kg.get_driver()
                    with driver.session() as session:
                        matched = self._kg.find_matching_diseases(session, dx_texts)[:3]
                    logger.debug(f"MATCHED KG DISEASES: {matched}")
                except Exception:
                    pass

            # Call the Phase 2 functional entry point
            context_str = self._kg.retrieve_kg_context(dx_texts, max_diseases=3)

            if "No relevant" in context_str or not context_str.strip():
                logger.debug(f"retrieve_kg_context returned no facts for dx_texts={dx_texts}")
                return []

            # Split the paragraph back into individual facts for cleaner formatting.
            # NOTE: naive '.' splitting will incorrectly break on decimal values
            # inside a fact (e.g. "eGFR<30.5") — acceptable for now since KG facts
            # are template-generated without decimals, but flagged for anyone
            # changing retrieve_kg_context's fact templates.
            facts = [f.strip() + "." for f in context_str.split('.') if f.strip()]
            logger.debug(f"NUMBER OF KG FACTS: {len(facts)}")

            return facts

        except Exception as e:
            logger.warning(f"KG retrieval failed for hadm_id={hadm_id}: {e}")
            return []

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text) // 4

    def retrieve(
        self,
        question: str,
        hadm_id:  Optional[int] = None,
        mode:     Mode          = Mode.T,
    ) -> RetrievalResult:
        t0 = time.time()

        result = RetrievalResult(
            mode=mode,
            question=question,
            hadm_id=hadm_id,
        )

        result.retrieved_chunks = self._retrieve_text(question)

        if mode in (Mode.TE, Mode.TEK):
            if hadm_id is None:
                print(f"[Retriever] WARNING: mode={mode.value} requires hadm_id. Falling back to text-only.")
            else:
                result.ehr_snapshot = self._retrieve_ehr(hadm_id)

        if mode == Mode.TEK:
            if hadm_id is None:
                print("[Retriever] WARNING: mode=T+E+K requires hadm_id for KG. Skipping KG.")
            else:
                result.kg_facts = self._retrieve_kg(question, hadm_id)

        result.latency_ms       = (time.time() - t0) * 1000
        result.n_tokens_approx  = self._estimate_tokens(result.prompt_context)

        return result

    def retrieve_all_modes(
        self,
        question: str,
        hadm_id:  Optional[int] = None,
    ) -> dict[str, RetrievalResult]:
        return {
            Mode.T.value:   self.retrieve(question, hadm_id, Mode.T),
            Mode.TE.value:  self.retrieve(question, hadm_id, Mode.TE),
            Mode.TEK.value: self.retrieve(question, hadm_id, Mode.TEK),
        }

    def close(self) -> None:
        self._snapshot.close()


# ── Standalone test ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Test all 3 retrieval modes on a sample question."
    )
    parser.add_argument("--hadm-id",  type=int, default=None,
                        help="Admission ID to test (picked from QA file if omitted)")
    parser.add_argument("--question", type=str,
                        default="What is the primary diagnosis for this patient?")
    parser.add_argument("--model",    type=str, default=DEFAULT_MODEL)
    parser.add_argument("--top-k",   type=int, default=TOP_K)
    args = parser.parse_args()

    # Resolve hadm_id
    hadm_id = args.hadm_id
    if hadm_id is None:
        qa_file = Path("data/qa/ehrqa_router_train.parquet")  # canonical — RESEARCH_LOG.md, 2026-08-09
        if qa_file.exists():
            df = pd.read_parquet(qa_file, columns=["hadm_id"])
            hadm_id = int(df["hadm_id"].dropna().iloc[0])
            print(f"[INFO] No --hadm-id given. Using first from QA file: {hadm_id}")

    # ── Neo4j MKG Module Import ──
    kg_module = None
    try:
        print("[INFO] Attempting to load Neo4j KG module from src.mkg.retrieval...")
        import src.mkg.retrieval as kg_module
        print("[INFO] Neo4j KG module loaded successfully.")
    except Exception as e:
        print(f"[WARN] Could not initialize KG module from src.mkg.retrieval: {e}")
        print("[WARN] Proceeding with KG component disabled.")

    retriever = Retriever(model_name=args.model, top_k=args.top_k, kg_module=kg_module)

    print(f"\n[INFO] Question: '{args.question}'")
    print(f"[INFO] hadm_id : {hadm_id}")

    for mode in [Mode.T, Mode.TE, Mode.TEK]:
        print(f"\n{'═'*60}")
        print(f"MODE: {mode.value}")
        print("═" * 60)

        result = retriever.retrieve(
            question=args.question,
            hadm_id=hadm_id,
            mode=mode,
        )

        stats = result.stats
        print(f"  chunks    : {stats['n_chunks']}")
        print(f"  has_ehr   : {stats['has_ehr']}")
        print(f"  kg_facts  : {stats['n_kg_facts']}")
        print(f"  latency   : {stats['latency_ms']:.1f}ms")
        print(f"  ~tokens   : {stats['n_tokens_approx']}")
        print(f"\n── Prompt Context Preview (first 600 chars) ──")
        print(result.prompt_context[:600])
        print("...")

    retriever.close()
    print("\n✅ Retriever test complete.")


if __name__ == "__main__":
    main()