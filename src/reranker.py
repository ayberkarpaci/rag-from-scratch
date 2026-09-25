"""Cross-encoder tabanli yeniden siralama."""

import time
from typing import List

import requests

from src import config


class Reranker:
    """
    Retrieval sonuclarini soru-dokuman ciftleri uzerinden yeniden siralar.

    Vektor aramasi soru ve dokumani ayri ayri kodlar (bi-encoder); reranker
    ikisini birlikte degerlendirir (cross-encoder) ve daha isabetli bir alaka
    skoru uretir.
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
        Chunk listesini alaka skoruna gore siralar ve ilk top_k tanesini dondurur.

        score_threshold verilirse, esigin altinda kalan chunk'lar elenir. Bu,
        baglam sayisini sabit tutmak yerine alaka duzeyine gore degistirir.
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
                    print(f"    rate limit, {wait}s bekleniyor...")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                # Servislerin cogu sonuclari skora gore sirali dondurur ama
                # bu garanti degil; kesmeden once kendimiz siralariz.
                results = sorted(response.json()["results"],
                                 key=lambda r: -r["relevance_score"])[:top_k]

                if score_threshold is not None:
                    filtered = [r for r in results
                                if r["relevance_score"] >= score_threshold]
                    # Hicbir chunk esigi gecemezse en iyi adayi koru
                    results = filtered if filtered else results[:1]

                return [
                    {**chunks[item["index"]], "score": item["relevance_score"]}
                    for item in results
                ]

            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(10 * (attempt + 1))

        # Sessizce siralanmamis adaylara donmek, "rerank" etiketli bir deneyi
        # reranker olmadan calistirip sonuclari bozardi.
        raise RuntimeError(
            f"Reranker {self.max_retries} denemede de rate limit'e takildi"
        )
