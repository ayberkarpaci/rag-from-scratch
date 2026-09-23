"""Reranker modellerini karsilastirir.

Qwen3-Reranker-8B ve bge-reranker-v2-m3'un ayni sorgu uzerindeki
siralamalari karsilastirilir.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import requests

from src import config

MODELS = ["Qwen3-Reranker-8B", "bge-reranker-v2-m3"]

QUERY = "How do I deposit a third party cheque?"
DOCUMENTS = [
    "Just have the associate sign the back and then deposit it.",
    "The weather in Ankara is cold in winter.",
    "A third party cheque requires endorsement before deposit.",
]

HEADERS = {
    "Authorization": f"Bearer {config.LLM_API_KEY}",
    "Content-Type": "application/json",
}


def rerank(model: str):
    url = config.LLM_BASE_URL.rstrip("/") + "/rerank"
    payload = {"model": model, "query": QUERY, "documents": DOCUMENTS}

    response = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["results"]


def main():
    print(f"Sorgu: {QUERY}\n")

    for model in MODELS:
        print("=" * 70)
        print(model)
        print("=" * 70)

        try:
            for rank, item in enumerate(rerank(model), 1):
                text = item["document"]["text"]
                print(f"  {rank}. [{item['relevance_score']:.4f}] {text}")
        except Exception as e:
            print(f"  HATA: {e}")

        print()


if __name__ == "__main__":
    main()
