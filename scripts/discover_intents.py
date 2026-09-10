"""
Standalone CLI Runner for Intent Discovery
Runs unsupervised semantic clustering on customer tweets to empirically validate the intent taxonomy.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intents import Intent


def run_discovery(sample_size: int = 250, k: int = 7):
    data_path = PROJECT_ROOT / "data" / "processed" / "amazon_threads_subsample.csv"
    cache_path = PROJECT_ROOT / "data" / "processed" / f"discovery_embeddings_{sample_size}.npy"

    if not data_path.exists():
        print(f"[ERROR] Processed data not found at {data_path}. Run 'python src/data_prep.py' first.")
        sys.exit(1)

    print("==================================================")
    print("      @AmazonHelp Intent Taxonomy Discovery       ")
    print("==================================================")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} customer-agent threads.")

    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    customer_texts = sample_df["customer_text"].tolist()

    # Check for cached embeddings
    if cache_path.exists():
        print(f"\n[CACHE HIT] Loading precomputed embeddings from {cache_path.name}...")
        embeddings = np.load(cache_path)
    else:
        print(f"\nEmbedding {len(customer_texts)} customer inquiries (first run loads ~90MB local model)...")
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = embedder.encode(customer_texts, show_progress_bar=True, normalize_embeddings=True)
            print(f"[OK] Generated local embeddings: {embeddings.shape}")
            np.save(cache_path, embeddings)
        except Exception as e:
            print(f"[NOTE] sentence-transformers fallback to TF-IDF ({e})")
            vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
            embeddings = vectorizer.fit_transform(customer_texts).toarray()

    print(f"\nClustering into k={k} customer problem domains via KMeans...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    sample_df["cluster"] = kmeans.fit_predict(embeddings)

    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = tfidf.fit_transform(customer_texts)
    feature_names = np.array(tfidf.get_feature_names_out())

    print("\n--- Discovered Cluster Breakdown ---")
    for c in range(k):
        c_indices = sample_df[sample_df["cluster"] == c].index
        c_centroid = kmeans.cluster_centers_[c]
        c_embeddings = embeddings[c_indices]
        distances = np.linalg.norm(c_embeddings - c_centroid, axis=1)
        closest_idx = c_indices[np.argmin(distances)]

        c_tfidf = tfidf_matrix[c_indices].mean(axis=0)
        top_word_indices = np.argsort(np.asarray(c_tfidf).flatten())[::-1][:4]
        top_words = feature_names[top_word_indices]

        print(f"\n[Cluster {c+1}] ({len(c_indices)} tweets)")
        print(f"  Keywords : {', '.join(top_words)}")
        print(f"  Exemplar : \"{customer_texts[closest_idx][:110]}...\"")

    print("\n" + "=" * 50)
    print("Formalized Intent Enum in src/intents.py:")
    for intent in Intent:
        print(f"  * {intent.value:<26} -> Default Action: {intent.default_action}")
    print("=" * 50)


if __name__ == "__main__":
    run_discovery()
