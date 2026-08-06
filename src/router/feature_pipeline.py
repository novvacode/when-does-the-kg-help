"""
src/router/feature_pipeline.py
================================
Stable home for RouterConfig and HybridFeaturePipeline so pickled
router artifacts can always be deserialized, regardless of which
script trains or loads them, and regardless of whether that script
is executed directly (as __main__) or imported as a module.

IMPORTANT: Both classes must live here (not in train_router.py) because
HybridFeaturePipeline stores a RouterConfig instance as an attribute
(self.cfg). If RouterConfig were defined in train_router.py, pickling
HybridFeaturePipeline would also try to pickle a RouterConfig reference
pointing at whatever module was active as __main__ at save time — which
breaks the moment a different script (like run_evaluation.py) tries to
unpickle it. Keeping both classes in one stable, always-importable
module eliminates this failure mode permanently.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════


class RouterConfig:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Train the Advanced Adaptive RAG Router")
        parser.add_argument("--train-path", type=str, default="data/router/router_train_oracle.parquet")
        parser.add_argument("--val-path", type=str, default="data/router/router_val_oracle.parquet")
        parser.add_argument("--out-dir", type=str, default="models/router")
        parser.add_argument("--embed-model", type=str, default="BAAI/bge-small-en-v1.5")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--tune-hyperparams", action="store_true", help="Execute comprehensive 5-fold CV search")
        args = parser.parse_args()

        self.train_path = Path(args.train_path)
        self.val_path = Path(args.val_path)
        self.out_dir = Path(args.out_dir)
        self.embed_model = args.embed_model
        self.seed = args.seed
        self.tune_hyperparams = args.tune_hyperparams

        # Structural Patient Columns
        self.ehr_feature_cols = ["n_labs", "n_diag", "n_meds", "sparsity_score", "sparsity_bucket"]

        # Heuristic compute matrix aligned roughly with relative latency weights (T=1x, T+E=3x, T+E+K=10x)
        self.cost_matrix = {"T": 1.0, "T+E": 3.0, "T+E+K": 10.0}
        self.misclassification_penalty = 15.0


# ══════════════════════════════════════════════════════════════════════════════
# Hybrid Feature Pipeline
# ══════════════════════════════════════════════════════════════════════════════


class HybridFeaturePipeline:
    def __init__(self, config: RouterConfig, logger: logging.Logger):
        self.cfg = config
        self.logger = logger
        self.logger.info(f"Initialising Semantic Encoder: {config.embed_model}")
        self.embedder = SentenceTransformer(config.embed_model, device='cpu')
        self.scaler = StandardScaler()
        # NOTE: must match the exact bucket vocabulary produced by
        # src/lakehouse/sparsity.py (canonical sparsity source — see its
        # module docstring). Values are lowercase "low" / "medium" / "high".
        # A previous version of this map used a different vocabulary
        # ("very_sparse"/"sparse"/"dense") that silently never matched any
        # real bucket value, collapsing "low" and "high" to the same -1
        # fallback and destroying this feature's H2 signal. See
        # RESEARCH_LOG.md, 2026-08-06 audit, finding #3.
        self.bucket_map = {
            "low": 0,
            "medium": 1,
            "high": 2,
            "unknown": -1
        }

    def _extract_tabular(self, df: pd.DataFrame) -> np.ndarray:
        """Helper to securely extract, encode, and impute structural tabular features."""
        encoded_cols = []
        for col in self.cfg.ehr_feature_cols:
            if col == "sparsity_bucket":
                # Create a temporary numeric column for scaling to preserve string in CSV
                temp_col_name = "sparsity_bucket_encoded"
                if "sparsity_bucket" in df.columns:
                    df[temp_col_name] = df["sparsity_bucket"].map(self.bucket_map).fillna(-1)
                else:
                    df[temp_col_name] = -1
                encoded_cols.append(temp_col_name)
            else:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].fillna(0.0)
                encoded_cols.append(col)

        return df[encoded_cols].to_numpy().astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts text embeddings, fits the scaler on training EHR data, and returns concatenated vectors."""
        self.logger.info(f"Extracting hybrid features and fitting scaler on {len(df)} samples...")

        # 1. Generate text embeddings
        embeddings = self.embedder.encode(df["question"].tolist(), show_progress_bar=True)

        # 2. Extract and scale actual structural patient features
        tabular_features = self._extract_tabular(df)
        scaled_tabular = self.scaler.fit_transform(tabular_features)

        # 3. Securely concatenate features along the column axis
        return np.hstack((embeddings, scaled_tabular))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms validation or test datasets using the pre-fit training distribution rules."""
        self.logger.info(f"Transforming validation hybrid features for {len(df)} samples...")

        embeddings = self.embedder.encode(df["question"].tolist(), show_progress_bar=True)
        tabular_features = self._extract_tabular(df)
        scaled_tabular = self.scaler.transform(tabular_features)

        return np.hstack((embeddings, scaled_tabular))