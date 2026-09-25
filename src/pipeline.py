"""End-to-end RAG flow: question -> retrieval -> generation."""

from typing import List

from src import config
from src.bm25 import BM25
from src.embeddings import EmbeddingClient
from src.generation import Generator
from src.reranker import Reranker
from src.vectorstore import VectorStore


def reciprocal_rank_fusion(rankings: List[List[dict]], k_constant: int = 60) -> List[dict]:
    """
    Merges several rankings into one list.

    Each chunk gets 1/(k + rank) points for its rank in every list, so chunks
    that rank high in both methods come first.
    """
    scores = {}
    lookup = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_constant + rank + 1)
            lookup[chunk_id] = chunk

    ordered = sorted(scores.items(), key=lambda x: -x[1])
    return [{**lookup[chunk_id], "score": score} for chunk_id, score in ordered]


class RAGPipeline:
    """Retrieves context from the vector store and generates the LLM answer."""

    def __init__(self, store: VectorStore = None, use_hybrid: bool = None,
                 use_reranker: bool = None, prompt_variant: str = "baseline",
                 score_threshold: float = None):
        self.store = store or VectorStore()
        if store is None:
            self.store.load()

        self.embedder = EmbeddingClient()
        self.generator = Generator(prompt_variant=prompt_variant)

        self.use_hybrid = (use_hybrid if use_hybrid is not None
                           else config.USE_HYBRID_SEARCH)
        self.bm25 = BM25(self.store.chunks) if self.use_hybrid else None

        self.use_reranker = (use_reranker if use_reranker is not None
                             else config.USE_RERANKER)
        self.reranker = Reranker() if self.use_reranker else None

        self.score_threshold = score_threshold

    def retrieve(self, question: str, retrieve_k: int = None,
                 top_k: int = None) -> List[dict]:
        """
        Returns the chunks closest to the question.

        With the reranker on, retrieve_k candidates are fetched first and then
        cut down to top_k: a wide pool keeps Recall up, the cut keeps Precision.
        """
        retrieve_k = retrieve_k if retrieve_k is not None else config.RETRIEVE_K
        top_k = top_k if top_k is not None else config.TOP_K

        query_vector = self.embedder.embed_one(question)
        fetch_k = retrieve_k * 2 if self.use_hybrid else retrieve_k

        candidates = [{**chunk, "score": score}
                      for chunk, score in self.store.search(query_vector, k=fetch_k)]

        if self.use_hybrid:
            sparse = [{**chunk, "score": score}
                      for chunk, score in self.bm25.search(question, k=fetch_k)]
            candidates = reciprocal_rank_fusion([candidates, sparse])[:retrieve_k]

        if self.use_reranker:
            return self.reranker.rerank(question, candidates, top_k,
                                        score_threshold=self.score_threshold)

        return candidates[:top_k]

    def answer(self, question: str, retrieve_k: int = None,
               top_k: int = None) -> dict:
        """Answers the question and also returns the context used."""
        chunks = self.retrieve(question, retrieve_k, top_k)
        contexts = [c["text"] for c in chunks]
        answer = self.generator.generate(question, contexts)

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "chunks": chunks,
        }
