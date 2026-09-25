"""Checks the embedding service and the cache."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np

from src.embeddings import EmbeddingClient


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    client = EmbeddingClient()

    texts = [
        "The company's net profit increased by 27 percent.",
        "Firm revenue showed significant growth this year.",
        "The cat is sleeping on the sofa.",
    ]

    start = time.time()
    vectors = client.embed(texts, show_progress=True)
    print(f"First call: {time.time() - start:.2f} s")
    print(f"Matrix shape: {vectors.shape}")

    start = time.time()
    client.embed(texts)
    print(f"Second call (cache): {time.time() - start:.2f} s")

    print("\nSimilarity check:")
    print(f"  finance-finance : {cosine(vectors[0], vectors[1]):.4f}")
    print(f"  finance-cat     : {cosine(vectors[0], vectors[2]):.4f}")


if __name__ == "__main__":
    main()
