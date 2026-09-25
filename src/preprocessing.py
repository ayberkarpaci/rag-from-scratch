"""Corpus text cleaning.

The dataset's context blocks are several forum answers joined together. The
joins have broken punctuation and spacing, which disables the natural
sentence separators of the recursive chunker.
"""

import re


def clean_text(text: str) -> str:
    """Normalizes text before chunking."""
    # Collapse escaped doubled quotes to one
    text = text.replace('""', '"')

    # Add a space after sentence-ending punctuation if missing
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

    # Separate a letter that directly follows a quote
    text = re.sub(r'(")([A-Za-z])', r"\1 \2", text)

    # Collapse repeated spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def clean_documents(documents: list) -> list:
    """Cleans the text of every document in the list."""
    return [{**doc, "text": clean_text(doc["text"])} for doc in documents]
