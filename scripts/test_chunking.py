"""Shows how the chunking parameters affect the corpus."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking import chunk_documents
from src.preprocessing import clean_documents


def load_corpus():
    with open(ROOT / "data" / "raw" / "corpus.json", "r", encoding="utf-8") as f:
        return json.load(f)


def show_samples(documents, chunk_size, overlap, count=3):
    chunks = chunk_documents(documents[:1], chunk_size, overlap)

    print(f"\nchunk_size={chunk_size}, overlap={overlap} "
          f"-> first document gives {len(chunks)} chunks")

    for chunk in chunks[:count]:
        print(f"\n[{chunk['chunk_id']}] {len(chunk['text'])} characters")
        print(chunk["text"])


def compare_parameters(documents):
    configs = [(200, 20), (300, 30), (500, 50), (800, 80), (1000, 100)]

    print(f"\n{'size':>6} {'overlap':>8} {'chunks':>8} {'mean':>8}")

    for chunk_size, overlap in configs:
        chunks = chunk_documents(documents, chunk_size, overlap)
        avg = sum(len(c["text"]) for c in chunks) // len(chunks)
        print(f"{chunk_size:>6} {overlap:>8} {len(chunks):>8} {avg:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-clean", action="store_true",
                        help="Skip text cleaning (applied by default)")
    args = parser.parse_args()

    documents = load_corpus()
    if not args.no_clean:
        documents = clean_documents(documents)

    show_samples(documents, chunk_size=800, overlap=80)
    compare_parameters(documents)


if __name__ == "__main__":
    main()
