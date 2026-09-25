"""Prints retrieval results for a manual quality check."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src import config
from src.embeddings import EmbeddingClient
from src.vectorstore import VectorStore


def load_questions() -> list:
    with open(config.DATA_RAW / "questions.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    store = VectorStore()
    store.load()
    print(f"Store loaded: {len(store)} chunks\n")

    client = EmbeddingClient()
    questions = load_questions()

    for question in questions[:3]:
        query_vector = client.embed_one(question["question"])
        results = store.search(query_vector, k=3)

        print("=" * 70)
        print(f"QUESTION: {question['question']}")
        print(f"Expected source: doc_{question['question_id'][2:]}_*")
        print()

        for chunk, score in results:
            preview = chunk["text"][:150].replace("\n", " ")
            print(f"  [{score:.4f}] {chunk['chunk_id']}")
            print(f"           {preview}...")
        print()


if __name__ == "__main__":
    main()
