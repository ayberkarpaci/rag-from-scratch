"""Ragas ile RAG cikti kalitesini olcer."""

import certifi
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig
from datasets import Dataset

from src import config


def _http_client() -> httpx.Client:
    return httpx.Client(verify=certifi.where(), timeout=180.0)


def build_judge_llm() -> LangchainLLMWrapper:
    """Ragas'in hakem modeli. Reasoning cikti urettigi icin max_tokens yuksek tutulur."""
    llm = ChatOpenAI(
        model=config.JUDGE_MODEL,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        temperature=0.0,
        max_tokens=8192,
        http_client=_http_client(),
    )
    return LangchainLLMWrapper(llm)


def build_judge_embeddings() -> LangchainEmbeddingsWrapper:
    """ResponseRelevancy metrigi embedding gerektirir."""
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        http_client=_http_client(),
        check_embedding_ctx_length=False,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def build_dataset(records: list) -> Dataset:
    """Pipeline ciktisini Ragas'in bekledigi alan adlarina cevirir."""
    return Dataset.from_dict({
        "user_input": [r["question"] for r in records],
        "response": [r["answer"] for r in records],
        "retrieved_contexts": [r["contexts"] for r in records],
        "reference": [r["ground_truth"] for r in records],
    })


def run_evaluation(records: list) -> dict:
    llm = build_judge_llm()
    embeddings = build_judge_embeddings()

    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]

    # Hakem modeli reasoning urettigi icin cagirilar yavas; zaman asimi
    # yukseltildi. Servis 15 istek/pencere siniri koydugu icin es zamanli
    # istek sayisi dusuk tutuldu.
    run_config = RunConfig(timeout=900, max_workers=2, max_retries=10)

    result = evaluate(
        dataset=build_dataset(records),
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    return result
