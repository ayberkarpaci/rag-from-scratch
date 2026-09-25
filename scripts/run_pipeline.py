"""Runs every question through the pipeline and saves the output for scoring."""

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
from src.pipeline import RAGPipeline


def load_questions() -> list:
    with open(config.DATA_RAW / "questions.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None,
                        help="Number of chunks to retrieve")
    parser.add_argument("--retrieve-k", type=int, default=None,
                        help="Candidates before reranking")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Chunks given to the LLM")
    parser.add_argument("--name", type=str, default="baseline")
    # The index is built by build_index.py; these two values are only
    # recorded in the output file's configuration.
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Chunk size of the current index (for the record)")
    parser.add_argument("--overlap", type=int, default=None,
                        help="Overlap of the current index (for the record)")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--prompt", type=str, default="baseline",
                        choices=["baseline", "strict", "cited"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="Reranker score threshold; chunks below it are dropped")
    args = parser.parse_args()

    # A single --k applies to both
    retrieve_k = args.retrieve_k or args.k or config.RETRIEVE_K
    top_k = args.top_k or args.k or config.TOP_K

    questions = load_questions()
    pipeline = RAGPipeline(use_hybrid=args.hybrid, use_reranker=args.rerank,
                           prompt_variant=args.prompt,
                           score_threshold=args.threshold)

    run_config = {
        **config.describe_config(),
        "retrieve_k": retrieve_k,
        "top_k": top_k,
        "chunk_size": args.chunk_size or config.CHUNK_SIZE,
        "chunk_overlap": args.overlap if args.overlap is not None else config.CHUNK_OVERLAP,
        "use_hybrid_search": args.hybrid,
        "use_reranker": args.rerank,
        "prompt_variant": args.prompt,
        "score_threshold": args.threshold,
    }

    print(f"Experiment: {args.name}")
    print(f"{len(questions)} questions, retrieve_k={retrieve_k}, top_k={top_k}, "
          f"rerank={args.rerank}\n")

    records = []
    start = time.time()

    for i, item in enumerate(questions, 1):
        result = pipeline.answer(item["question"], retrieve_k=retrieve_k, top_k=top_k)

        records.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "retrieved_chunk_ids": [c["chunk_id"] for c in result["chunks"]],
            "scores": [c["score"] for c in result["chunks"]],
        })

        print(f"  {i}/{len(questions)}  {item['question_id']}")

    elapsed = time.time() - start

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULTS_DIR / f"pipeline_{args.name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": args.name,
            "config": run_config,
            "records": records,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nTime: {elapsed:.1f} s")
    print(f"-> {output_path}")


if __name__ == "__main__":
    main()
