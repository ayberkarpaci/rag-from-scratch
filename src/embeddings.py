"""Embedding servisi ve disk cache."""

import hashlib
import json
from pathlib import Path
from typing import List

import certifi
import httpx
import numpy as np
from openai import OpenAI

from src import config


class EmbeddingClient:
    """Metinleri vektore cevirir. Ayni metin tekrar istenirse cache'ten doner."""

    def __init__(self, cache_dir: Path = None, batch_size: int = 16):
        self.client = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            http_client=httpx.Client(verify=certifi.where(), timeout=120.0),
        )
        self.model = config.EMBEDDING_MODEL
        self.batch_size = batch_size

        self.cache_dir = cache_dir or (config.ROOT / "data" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "embeddings.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        with open(self.cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)

    def _key(self, text: str) -> str:
        """Model adi + metin icerigi uzerinden cache anahtari."""
        raw = f"{self.model}:{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _fetch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Metin listesini vektor matrisine cevirir. Sonuc (n, boyut) seklindedir."""
        keys = [self._key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]

        if missing and show_progress:
            print(f"  {len(missing)}/{len(texts)} metin hesaplanacak "
                  f"({len(texts) - len(missing)} cache'te)")

        for start in range(0, len(missing), self.batch_size):
            batch_idx = missing[start:start + self.batch_size]
            batch_texts = [texts[i] for i in batch_idx]

            vectors = self._fetch(batch_texts)
            for i, vector in zip(batch_idx, vectors):
                self.cache[keys[i]] = vector

            if show_progress:
                done = min(start + self.batch_size, len(missing))
                print(f"  {done}/{len(missing)}")

        if missing:
            self._save_cache()

        return np.array([self.cache[k] for k in keys], dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
