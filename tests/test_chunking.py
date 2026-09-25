from src.chunking import chunk_documents, chunk_text

import pytest


def test_short_text_is_one_chunk():
    assert chunk_text("Kisa bir metin.", chunk_size=100, overlap=10) == ["Kisa bir metin."]


def test_empty_text_has_no_chunks():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []


def test_chunks_respect_size_before_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, chunk_size=120, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)


def test_splits_on_sentence_boundaries_when_possible():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = chunk_text(text, chunk_size=45, overlap=0)
    assert all(c.endswith(".") for c in chunks)


def test_overlap_prefixes_previous_tail():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=60, overlap=10)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.startswith(prev[-10:])


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_chunk_documents_keeps_source_ids():
    docs = [{"doc_id": "d1", "source_row": 3, "text": "a. " * 100}]
    chunks = chunk_documents(docs, chunk_size=50, overlap=5)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c["doc_id"] == "d1" and c["source_row"] == 3 for c in chunks)
    assert chunks[0]["chunk_id"] == "d1_c000"
