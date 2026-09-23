"""Pipeline ciktisini Ragas ile degerlendirir."""

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
from src.evaluation import run_evaluation


def load_output(name: str) -> dict:
    path = config.RESULTS_DIR / f"pipeline_{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Ilk N soruyu degerlendir (test icin)")
    parser.add_argument("--name", type=str, default="baseline",
                        help="Deney adi; pipeline_<name>.json okunur")
    args = parser.parse_args()

    data = load_output(args.name)
    records = data["records"]

    if args.limit:
        records = records[:args.limit]

    print(f"{len(records)} kayit degerlendirilecek")
    print(f"Hakem modeli: {config.JUDGE_MODEL}\n")

    start = time.time()
    result = run_evaluation(records)
    elapsed = time.time() - start

    scores = {k: float(v) for k, v in result._repr_dict.items()}

    precision = scores.get("llm_context_precision_with_reference")
    recall = scores.get("context_recall")
    if precision and recall:
        scores["retrieval_f1"] = 2 * precision * recall / (precision + recall)

    print("\n" + "=" * 50)
    for name, value in scores.items():
        print(f"  {name:<28} {value:.4f}")
    print("=" * 50)
    print(f"Sure: {elapsed:.1f} saniye ({elapsed / len(records):.1f} sn/soru)")

    df = result.to_pandas()
    nan_counts = {}
    print("\nNaN sayilari:")
    for column in ["faithfulness", "answer_relevancy",
                   "llm_context_precision_with_reference", "context_recall"]:
        if column in df.columns:
            count = int(df[column].isna().sum())
            nan_counts[column] = count
            print(f"  {column:<38} {count}/{len(df)}")

    output_path = config.RESULTS_DIR / f"eval_{args.name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": args.name,
            "config": data["config"],
            "record_count": len(records),
            "scores": scores,
            "nan_counts": nan_counts,
            "elapsed_seconds": round(elapsed, 1),
        }, f, ensure_ascii=False, indent=2)

    print(f"-> {output_path}")


if __name__ == "__main__":
    main()
