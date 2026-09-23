"""Corpus metin temizligi.

Veri setindeki context bloklari birden fazla forum cevabinin birlestirilmis
hali. Birlesme noktalarinda noktalama ve bosluk bozukluklari var; bu
bozukluklar recursive chunking'in dogal cumle ayiricilarini devre disi
birakiyor.
"""

import re


def clean_text(text: str) -> str:
    """Chunking oncesi metin normalizasyonu."""
    # Kacirilmis cift tirnaklari tek tirnaga indir
    text = text.replace('""', '"')

    # Cumle sonu noktalamasindan sonra bosluk yoksa ekle
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

    # Tirnak isaretinden hemen sonra harf geliyorsa ayir
    text = re.sub(r'(")([A-Za-z])', r"\1 \2", text)

    # Coklu bosluklari teke indir
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def clean_documents(documents: list) -> list:
    """Dokuman listesindeki metinleri temizler."""
    return [{**doc, "text": clean_text(doc["text"])} for doc in documents]
