"""BM25 kelime tabanli arama."""

import math
import re
from collections import Counter
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    """Metni kucuk harfli kelime listesine cevirir."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25:
    """
    Okapi BM25 siralamasi.

    Skor, terim sikligi (TF) ve ters dokuman sikligi (IDF) uzerinden
    hesaplanir. k1 terim sikliginin doyum noktasini, b ise dokuman uzunlugu
    normalizasyonunun agirligini belirler.
    """

    def __init__(self, chunks: List[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.documents = [tokenize(c["text"]) for c in chunks]
        self.doc_lengths = [len(d) for d in self.documents]
        self.avg_length = sum(self.doc_lengths) / len(self.documents)

        self.term_frequencies = [Counter(d) for d in self.documents]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict:
        """Her terim icin ters dokuman sikligi."""
        document_count = len(self.documents)
        containing = Counter()

        for tf in self.term_frequencies:
            for term in tf:
                containing[term] += 1

        return {
            term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in containing.items()
        }

    def search(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        """Sorguya en yuksek BM25 skoruna sahip k chunk'i dondurur."""
        query_terms = tokenize(query)
        scores = []

        for i, tf in enumerate(self.term_frequencies):
            score = 0.0
            length_ratio = self.doc_lengths[i] / self.avg_length

            for term in query_terms:
                if term not in tf:
                    continue

                frequency = tf[term]
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += self.idf.get(term, 0.0) * numerator / denominator

            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:k]
        return [(self.chunks[i], score) for i, score in ranked]
