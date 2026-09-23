"""NumPy tabanli depo ile ChromaDB'nin arama performansini karsilastirir.

Bu script chromadb paketini gerektirir (requirements.txt'de yer almaz):
    pip install chromadb
"""

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import chromadb
import numpy as np

from src import config
from src.embeddings import EmbeddingClient
from src.vectorstore import VectorStore


CHROMA_DIR = ROOT / "data" / "chroma_benchmark"


def load_questions() -> list:
    with open(config.DATA_RAW / "questions.json", "r", encoding="utf-8") as f:
        return json.load(f)


def benchmark_numpy(store: VectorStore, query_vectors: np.ndarray, k: int) -> float:
    start = time.perf_counter()
    for vector in query_vectors:
        store.search(vector, k=k)
    return (time.perf_counter() - start) / len(query_vectors) * 1000


def benchmark_chroma(collection, query_vectors: np.ndarray, k: int) -> float:
    start = time.perf_counter()
    for vector in query_vectors:
        collection.query(query_embeddings=[vector.tolist()], n_results=k)
    return (time.perf_counter() - start) / len(query_vectors) * 1000


def compare_results(store: VectorStore, collection, query_vectors: np.ndarray,
                    k: int) -> float:
    """Iki yontemin dondurdugu chunk kumelerinin ortusme oranini olcer."""
    overlaps = []

    for vector in query_vectors:
        numpy_ids = {c["chunk_id"] for c, _ in store.search(vector, k=k)}
        chroma_result = collection.query(query_embeddings=[vector.tolist()], n_results=k)
        chroma_ids = set(chroma_result["ids"][0])
        overlaps.append(len(numpy_ids & chroma_ids) / k)

    return sum(overlaps) / len(overlaps)


def main():
    k = config.RETRIEVE_K

    store = VectorStore()
    store.load()
    print(f"Chunk sayisi: {len(store)}")

    questions = load_questions()
    embedder = EmbeddingClient()
    query_vectors = embedder.embed([q["question"] for q in questions])

    # ChromaDB kolleksiyonu ayni vektorlerle kuruluyor
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.create_collection(
        name="benchmark",
        metadata={"hnsw:space": "cosine"},
    )

    start = time.perf_counter()
    collection.add(
        ids=[c["chunk_id"] for c in store.chunks],
        embeddings=[v.tolist() for v in store.matrix],
        documents=[c["text"] for c in store.chunks],
    )
    chroma_index_time = time.perf_counter() - start

    numpy_ms = benchmark_numpy(store, query_vectors, k)
    chroma_ms = benchmark_chroma(collection, query_vectors, k)
    overlap = compare_results(store, collection, query_vectors, k)

    print()
    print("=" * 52)
    print(f"{'Yontem':<20} {'Sorgu (ms)':>14}")
    print("-" * 52)
    print(f"{'NumPy (exact)':<20} {numpy_ms:>14.3f}")
    print(f"{'ChromaDB (HNSW)':<20} {chroma_ms:>14.3f}")
    print("=" * 52)
    print(f"ChromaDB indeksleme suresi : {chroma_index_time:.3f} saniye")
    print(f"Sonuc ortusme orani        : {overlap:.1%}")

    try:
        shutil.rmtree(CHROMA_DIR)
    except PermissionError:
        print(f"\nNot: {CHROMA_DIR} elle silinebilir (dosya kilidi).")


if __name__ == "__main__":
    main()
