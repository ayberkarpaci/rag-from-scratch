from src.bm25 import BM25, tokenize


CHUNKS = [
    {"chunk_id": "a", "text": "Index funds have low expense ratios."},
    {"chunk_id": "b", "text": "A mortgage is a loan secured by a house."},
    {"chunk_id": "c", "text": "Expense ratios reduce the returns of a fund."},
]


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Low-cost Index, FUNDS!") == ["low", "cost", "index", "funds"]


def test_relevant_chunk_ranks_first():
    results = BM25(CHUNKS).search("mortgage loan", k=3)
    assert results[0][0]["chunk_id"] == "b"
    assert results[0][1] > results[1][1]


def test_unknown_terms_score_zero():
    results = BM25(CHUNKS).search("cryptocurrency", k=3)
    assert all(score == 0.0 for _, score in results)


def test_k_limits_results():
    assert len(BM25(CHUNKS).search("expense ratios", k=2)) == 2
