"""Builds the corpus and question files from the FiQA dataset."""

import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


def build_corpus():
    print("Loading dataset...")
    data = load_dataset("vibrantlabsai/fiqa", "ragas_eval_v3")["baseline"]

    documents = []
    questions = []

    for idx, row in enumerate(data):
        for ctx_idx, ctx in enumerate(row["retrieved_contexts"]):
            documents.append({
                "doc_id": f"doc_{idx:03d}_{ctx_idx}",
                "source_row": idx,
                "text": ctx,
            })

        reference = row["reference"]
        if isinstance(reference, list):
            reference = reference[0] if reference else ""

        questions.append({
            "question_id": f"q_{idx:03d}",
            "question": row["user_input"],
            "ground_truth": reference,
        })

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    corpus_path = RAW_DIR / "corpus.json"
    questions_path = RAW_DIR / "questions.json"

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    total_chars = sum(len(d["text"]) for d in documents)
    print(f"Documents: {len(documents)}  Questions: {len(questions)}  "
          f"Total: {total_chars:,} characters")
    print(f"-> {corpus_path}")
    print(f"-> {questions_path}")


if __name__ == "__main__":
    build_corpus()
