"""Corpus'taki metin bozukluklarini tespit eder."""

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
        "bosluksuz cumle sonu (.X)": r"[.!?][A-Z]",
        "cift tirnak": r'""',
        "tirnak sonrasi harf": r'"[A-Za-z]',
        "coklu bosluk": r"  +",
        "bosluksuz virgul": r",[A-Za-z]",
    }

    print(f"Toplam {len(documents)} dokuman, {len(text):,} karakter\n")

    for name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        print(f"{name:<32} {len(matches):>5}")

    print("\n--- Ornekler ---")
    for match in re.finditer(r"[.!?][A-Z]", text):
        start = max(0, match.start() - 40)
        print(f"  ...{text[start:match.end() + 40]}...")
        if match.start() > 20000:
            break


if __name__ == "__main__":
    main()
