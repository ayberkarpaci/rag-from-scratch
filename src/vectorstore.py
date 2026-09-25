"""Vector store. The corpus is small, so exact search is used."""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src import config


class VectorStore:
    """Holds chunk vectors and searches them by cosine similarity."""

    def __init__(self):
        self.chunks: List[dict] = []
        self.matrix: np.ndarray = None

    def build(self, chunks: List[dict], vectors: np.ndarray):
        """Builds the store from a chunk list and the matching vector matrix."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"number of chunks ({len(chunks)}) does not match number of vectors ({len(vectors)})"
            )

        self.chunks = chunks
        self.matrix = self._normalize(vectors)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """Scales vectors to unit length, so cosine similarity becomes a dot product."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[dict, float]]:
        """Returns the k chunks closest to the query vector, with their similarity."""
        if self.matrix is None:
            raise RuntimeError("Store is empty. Call build() or load() first.")

        query = query_vector / (np.linalg.norm(query_vector) or 1.0)
        scores = self.matrix @ query

        k = min(k, len(self.chunks))
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def save(self, directory: Path = None):
        directory = directory or config.VECTORDB_DIR
        directory.mkdir(parents=True, exist_ok=True)

        np.save(directory / "vectors.npy", self.matrix)
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    def load(self, directory: Path = None):
        directory = directory or config.VECTORDB_DIR

        self.matrix = np.load(directory / "vectors.npy")
        with open(directory / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def __len__(self) -> int:
        return len(self.chunks)
