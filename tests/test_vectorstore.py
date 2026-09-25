import numpy as np
import pytest

from src.vectorstore import VectorStore


def make_store():
    chunks = [{"chunk_id": str(i)} for i in range(4)]
    vectors = np.array([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=np.float32)
    store = VectorStore()
    store.build(chunks, vectors)
    return store


def test_search_orders_by_cosine_similarity():
    results = make_store().search(np.array([2.0, 0.1]), k=4)
    ids = [c["chunk_id"] for c, _ in results]
    assert ids == ["0", "2", "1", "3"]
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(0.99875, abs=1e-4)


def test_k_larger_than_store_returns_everything():
    assert len(make_store().search(np.array([1.0, 0.0]), k=10)) == 4


def test_save_and_load_round_trip(tmp_path):
    store = make_store()
    store.save(tmp_path)
    loaded = VectorStore()
    loaded.load(tmp_path)
    assert loaded.chunks == store.chunks
    np.testing.assert_allclose(loaded.matrix, store.matrix)


def test_build_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        VectorStore().build([{"chunk_id": "x"}], np.zeros((2, 3)))


def test_search_before_build_fails():
    with pytest.raises(RuntimeError):
        VectorStore().search(np.array([1.0]), k=1)
