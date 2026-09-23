"""Dokuman parcalama."""

from typing import List


SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def _split_by_separator(text: str, separator: str) -> List[str]:
    if separator == "":
        return list(text)

    parts = text.split(separator)
    result = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            result.append(part + separator)
        elif part:
            result.append(part)
    return result


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces = _split_by_separator(text, separators[0])
    remaining = separators[1:]

    chunks = []
    buffer = ""

    for piece in pieces:
        if len(piece) > chunk_size:
            if buffer.strip():
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_recursive_split(piece, chunk_size, remaining))
            continue

        if len(buffer) + len(piece) <= chunk_size:
            buffer += piece
        else:
            if buffer.strip():
                chunks.append(buffer)
            buffer = piece

    if buffer.strip():
        chunks.append(buffer)

    return chunks


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        result.append(chunks[i - 1][-overlap:] + chunks[i])
    return result


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Metni parcalara boler. Ayiricilar dogaldan zorlamaya dogru denenir."""
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) < chunk_size ({chunk_size}) olmali")

    chunks = _recursive_split(text, chunk_size, SEPARATORS)
    chunks = [c.strip() for c in chunks if c.strip()]
    return _apply_overlap(chunks, overlap)


def chunk_documents(documents: List[dict], chunk_size: int = 500,
                    overlap: int = 50) -> List[dict]:
    """Dokuman listesini parcalar, kaynak bilgisini korur."""
    result = []

    for doc in documents:
        for i, piece in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            result.append({
                "chunk_id": f"{doc['doc_id']}_c{i:03d}",
                "doc_id": doc["doc_id"],
                "source_row": doc.get("source_row"),
                "chunk_index": i,
                "text": piece,
            })

    return result
