"""Finds text defects in the corpus."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config


def main():
    with open(config.DATA_RAW / "corpus.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    text = " ".join(d["text"] for d in documents)

    patterns = {
        "sentence end without space (.X)": r"[.!?][A-Z]",
        "doubled quote": r'""',
        "letter after quote": r'"[A-Za-z]',
        "repeated spaces": r"  +",
        "comma without space": r",[A-Za-z]",
    }

    print(f"{len(documents)} documents, {len(text):,} characters\n")

    for name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        print(f"{name:<32} {len(matches):>5}")

    print("\n--- Examples ---")
    for match in re.finditer(r"[.!?][A-Z]", text):
        start = max(0, match.start() - 40)
        print(f"  ...{text[start:match.end() + 40]}...")
        if match.start() > 20000:
            break


if __name__ == "__main__":
    main()
