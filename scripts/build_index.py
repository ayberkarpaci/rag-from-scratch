"""Corpus'u parcalara boler, vektore cevirir ve vektor deposunu olusturur."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src import config
from src.chunking import chunk_documents
from src.embeddings import EmbeddingClient
from src.preprocessing import clean_documents
from src.vectorstore import VectorStore


def load_corpus() -> list:
    with open(config.DATA_RAW / "corpus.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--no-clean", action="store_true",
                        help="Metin temizligini atla (varsayilan: uygulanir)")
    args = parser.parse_args()

    chunk_size = args.chunk_size if args.chunk_size else config.CHUNK_SIZE
    overlap = args.overlap if args.overlap is not None else config.CHUNK_OVERLAP

    start = time.time()

    documents = load_corpus()
    if not args.no_clean:
        documents = clean_documents(documents)
        print("Metin temizligi uygulandi")

    chunks = chunk_documents(documents, chunk_size, overlap)
    print(f"{len(documents)} dokuman -> {len(chunks)} chunk "
          f"(size={chunk_size}, overlap={overlap})")

    client = EmbeddingClient()
    vectors = client.embed([c["text"] for c in chunks], show_progress=True)

    store = VectorStore()
    store.build(chunks, vectors)
    store.save()

    print(f"Depo kaydedildi: {config.VECTORDB_DIR}")
    print(f"Sure: {time.time() - start:.1f} saniye")


if __name__ == "__main__":
    main()
