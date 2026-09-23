"""Pipeline ciktisinin saglikli olup olmadigini kontrol eder."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="baseline",
                        help="Deney adi; pipeline_<name>.json okunur")
    args = parser.parse_args()

    path = config.RESULTS_DIR / f"pipeline_{args.name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]

    empty = [r["question_id"] for r in records if not r["answer"]]
    lengths = [len(r["answer"]) for r in records]
    context_counts = [len(r["contexts"]) for r in records]

    print(f"Deney            : {args.name}")
    print(f"Kayit sayisi     : {len(records)}")
    print(f"Bos cevap        : {len(empty)} {empty if empty else ''}")
    print(f"Cevap uzunlugu   : ort {sum(lengths) // len(lengths)}, "
          f"min {min(lengths)}, max {max(lengths)}")
    print(f"Baglam sayisi    : ort {sum(context_counts) / len(context_counts):.1f}, "
          f"min {min(context_counts)}, max {max(context_counts)}")

    refusals = [r["question_id"] for r in records
                if "does not contain enough information" in r["answer"]]
    print(f"'Bilgi yok' yaniti: {len(refusals)} {refusals if refusals else ''}")


if __name__ == "__main__":
    main()
