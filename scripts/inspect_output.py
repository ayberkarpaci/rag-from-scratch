"""Sanity-checks the output of a pipeline run."""

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
                        help="Experiment name; reads pipeline_<name>.json")
    args = parser.parse_args()

    path = config.RESULTS_DIR / f"pipeline_{args.name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]

    empty = [r["question_id"] for r in records if not r["answer"]]
    lengths = [len(r["answer"]) for r in records]
    context_counts = [len(r["contexts"]) for r in records]

    print(f"Experiment       : {args.name}")
    print(f"Records          : {len(records)}")
    print(f"Empty answers    : {len(empty)} {empty if empty else ''}")
    print(f"Answer length    : mean {sum(lengths) // len(lengths)}, "
          f"min {min(lengths)}, max {max(lengths)}")
    print(f"Contexts         : mean {sum(context_counts) / len(context_counts):.1f}, "
          f"min {min(context_counts)}, max {max(context_counts)}")

    refusals = [r["question_id"] for r in records
                if "does not contain enough information" in r["answer"]]
    print(f"'No info' answers: {len(refusals)} {refusals if refusals else ''}")


if __name__ == "__main__":
    main()
