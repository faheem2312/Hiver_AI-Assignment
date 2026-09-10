"""
Retrieval Index for Historical Amazon Support Resolutions
=========================================================
Builds an in-memory vector index (FAISS or vectorized NumPy cosine similarity)
over historical (customer_query, brand_resolution) pairs from `amazon_grounding_corpus.csv`.

Features:
1. Filters exclusively for informative resolutions (is_informative_resolution == True).
2. Computes embeddings locally using sentence-transformers (all-MiniLM-L6-v2).
3. Caches precomputed embeddings to disk (grounding_embeddings.npy) for instant reloads.
4. Intent-aware boosting: rewards historical pairs that share the predicted intent.
5. Dual engine: Uses FAISS if available, with automatic vectorized NumPy fallback.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intents import Intent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "amazon_grounding_corpus.csv"
EMBEDDINGS_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "grounding_embeddings.npy"

# Try importing FAISS; fall back seamlessly to NumPy if not available
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class ResolutionRetriever:
    """Retrieves top-k historical brand resolutions for a given customer inquiry."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        intent_boost: float = 0.15,
        force_rebuild: bool = False
    ):
        self.model_name = model_name
        self.intent_boost = intent_boost
        self.force_rebuild = force_rebuild

        # 1. Load Grounding Corpus
        if not CORPUS_PATH.exists():
            raise FileNotFoundError(
                f"Grounding corpus missing at {CORPUS_PATH}. "
                f"Run 'python eval/sample_golden_set.py' first to create isolated corpus."
            )

        full_df = pd.read_csv(CORPUS_PATH)
        # Strictly keep informative resolutions only
        if "is_informative_resolution" in full_df.columns:
            self.corpus_df = full_df[full_df["is_informative_resolution"] == True].reset_index(drop=True)
        else:
            self.corpus_df = full_df.reset_index(drop=True)

        logger.info(f"Loaded {len(self.corpus_df):,} informative historical resolution pairs.")

        # 2. Load Local Embedding Model
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(self.model_name)

        # 3. Load or Compute Embeddings
        self.embeddings = self._get_or_compute_embeddings()

        # 4. Build Index (FAISS or NumPy)
        self.faiss_index = None
        if HAS_FAISS:
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings.astype(np.float32))
            logger.info("Built in-memory FAISS IndexFlatIP vector index.")
        else:
            logger.info("FAISS not detected; using high-speed vectorized NumPy cosine similarity.")

    def _get_or_compute_embeddings(self) -> np.ndarray:
        """Loads cached embeddings from disk or computes and persists them."""
        if not self.force_rebuild and EMBEDDINGS_CACHE_PATH.exists():
            try:
                cached_emb = np.load(EMBEDDINGS_CACHE_PATH)
                if len(cached_emb) == len(self.corpus_df):
                    logger.info(f"Loaded precomputed embeddings from {EMBEDDINGS_CACHE_PATH.name} ({cached_emb.shape}).")
                    return cached_emb
            except Exception as e:
                logger.warning(f"Could not load cached embeddings: {e}")

        logger.info(f"Computing embeddings for {len(self.corpus_df):,} grounding pairs...")
        texts = self.corpus_df["customer_text"].tolist()
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64
        ).astype(np.float32)

        try:
            EMBEDDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.save(EMBEDDINGS_CACHE_PATH, embeddings)
            logger.info(f"Saved grounding embeddings to {EMBEDDINGS_CACHE_PATH.name}.")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")

        return embeddings

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        predicted_intent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k historical brand resolutions semantically closest to the query,
        optionally boosted by predicted intent match.

        Returns:
            List of dicts: [
                {
                    "customer_text": str,
                    "brand_resolution": str,
                    "similarity": float,
                    "category": str,
                    "intent_matched": bool
                }, ...
            ]
        """
        # Encode query and normalize for cosine similarity
        query_vec = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)

        # Compute raw cosine similarities across entire corpus
        # (normalized dot product = cosine similarity)
        raw_scores = np.dot(self.embeddings, query_vec.T).flatten()

        # Apply Intent Boosting
        final_scores = raw_scores.copy()
        if predicted_intent and "category_tag" in self.corpus_df.columns:
            intent_mask = (self.corpus_df["category_tag"] == predicted_intent).values
            final_scores[intent_mask] += self.intent_boost

        # Select top-k indices
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.corpus_df.iloc[idx]
            cat = row.get("category_tag", "UNKNOWN")
            is_matched = bool(predicted_intent and cat == predicted_intent)
            results.append({
                "customer_text": row["customer_text"],
                "brand_resolution": row["brand_text"],
                "similarity": float(raw_scores[idx]),
                "boosted_similarity": float(final_scores[idx]),
                "category": cat,
                "intent_matched": is_matched,
            })

        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test retrieval of historical Amazon support resolutions.")
    parser.add_argument("--query", type=str, default=None, help="Customer message to retrieve resolutions for.")
    parser.add_argument("--intent", type=str, default=None, help="Optional intent to boost.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of examples to retrieve.")
    args = parser.parse_args()

    test_q = args.query or "I returned my shoes 4 days ago via UPS, when will the refund show up in my account?"
    test_intent = args.intent or Intent.REFUND_RETURN_INQUIRY.value

    print("==================================================")
    print("      @AmazonHelp Resolution Retrieval Test       ")
    print("==================================================")
    print(f"Query  : \"{test_q}\"")
    print(f"Intent : {test_intent}\n")

    retriever = ResolutionRetriever()
    hits = retriever.retrieve(test_q, top_k=args.top_k, predicted_intent=test_intent)

    for i, hit in enumerate(hits, 1):
        print(f"--- Top Match #{i} (Cosine Sim: {hit['similarity']:.3f}, Category: {hit['category']}) ---")
        print(f"  Historical Query : {hit['customer_text']}")
        print(f"  Brand Resolution : {hit['brand_resolution']}\n")
    print("==================================================")
