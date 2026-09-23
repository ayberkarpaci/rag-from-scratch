"""Embedding servisini ve cache davranisini dogrular."""

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
    print(f"Ilk cagri: {time.time() - start:.2f} saniye")
    print(f"Matris sekli: {vectors.shape}")

    start = time.time()
    client.embed(texts)
    print(f"Ikinci cagri (cache): {time.time() - start:.2f} saniye")

    print("\nBenzerlik kontrolu:")
    print(f"  finans-finans : {cosine(vectors[0], vectors[1]):.4f}")
    print(f"  finans-kedi   : {cosine(vectors[0], vectors[2]):.4f}")


if __name__ == "__main__":
    main()
