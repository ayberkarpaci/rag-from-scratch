from src.pipeline import reciprocal_rank_fusion


def ranking(*ids):
    return [{"chunk_id": i, "text": i} for i in ids]


def test_chunk_ranked_high_in_both_lists_wins():
    fused = reciprocal_rank_fusion([ranking("a", "b", "c"), ranking("b", "a", "d")])
    assert [c["chunk_id"] for c in fused][:2] in (["a", "b"], ["b", "a"])
    assert {c["chunk_id"] for c in fused} == {"a", "b", "c", "d"}


def test_scores_follow_rrf_formula():
    fused = reciprocal_rank_fusion([ranking("a", "b"), ranking("b")], k_constant=60)
    scores = {c["chunk_id"]: c["score"] for c in fused}
    assert scores["a"] == 1 / 61
    assert scores["b"] == 1 / 62 + 1 / 61
    assert fused[0]["chunk_id"] == "b"
