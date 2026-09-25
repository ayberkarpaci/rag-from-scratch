"""Cross-encoder reranking."""

import time
from typing import List

import requests

from src import config


class Reranker:
    """
    Reorders retrieval results by scoring question-document pairs.

    Vector search encodes the question and the document separately
    (bi-encoder); the reranker reads them together (cross-encoder) and gives a
    more accurate relevance score.
    """

    def __init__(self, model: str = None, max_retries: int = 5):
        self.model = model or config.RERANKER_MODEL
        self.url = config.LLM_BASE_URL.rstrip("/") + "/rerank"
        self.headers = {
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        self.max_retries = max_retries

    def rerank(self, query: str, chunks: List[dict], top_k: int,
               score_threshold: float = None) -> List[dict]:
        """
        Sorts the chunks by relevance score and returns the first top_k.

        With score_threshold set, chunks below it are dropped, so the number of
        contexts follows relevance instead of staying fixed.
        """
        if not chunks:
            return []

        payload = {
            "model": self.model,
            "query": query,
            "documents": [c["text"] for c in chunks],
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.url, headers=self.headers,
                                         json=payload, timeout=120)

                if response.status_code == 429:
                    wait = 30 * (attempt + 1)
                    print(f"    rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                # Most servers return results sorted by score, but that is not
                # guaranteed; sort before cutting.
                results = sorted(response.json()["results"],
                                 key=lambda r: -r["relevance_score"])[:top_k]

                if score_threshold is not None:
                    filtered = [r for r in results
                                if r["relevance_score"] >= score_threshold]
                    # If no chunk passes the threshold, keep the best candidate
                    results = filtered if filtered else results[:1]

                return [
                    {**chunks[item["index"]], "score": item["relevance_score"]}
                    for item in results
                ]

            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(10 * (attempt + 1))

        # Silently falling back to unranked candidates would run an experiment
        # labelled "rerank" without a reranker and corrupt its results.
        raise RuntimeError(
            f"Reranker was still rate-limited after {self.max_retries} attempts"
        )
