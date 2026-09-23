"""Tum sorulari calistirir ve sonuclari degerlendirme icin kaydeder."""

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
                        help="Getirilecek chunk sayisi")
    parser.add_argument("--retrieve-k", type=int, default=None,
                        help="Reranker oncesi aday sayisi")
    parser.add_argument("--top-k", type=int, default=None,
                        help="LLM'e verilecek chunk sayisi")
    parser.add_argument("--name", type=str, default="baseline")
    # Indeks build_index.py ile kurulur; bu iki deger yalnizca sonuc
    # dosyasindaki konfigurasyon kaydina yazilir.
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Mevcut indeksin chunk boyutu (kayit icin)")
    parser.add_argument("--overlap", type=int, default=None,
                        help="Mevcut indeksin overlap degeri (kayit icin)")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--prompt", type=str, default="baseline",
                        choices=["baseline", "strict", "cited"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="Reranker skor esigi; altinda kalanlar elenir")
    args = parser.parse_args()

    # --k tek deger olarak verilirse her ikisine de uygulanir
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

    print(f"Deney: {args.name}")
    print(f"{len(questions)} soru, retrieve_k={retrieve_k}, top_k={top_k}, "
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

    print(f"\nSure: {elapsed:.1f} saniye")
    print(f"-> {output_path}")


if __name__ == "__main__":
    main()
