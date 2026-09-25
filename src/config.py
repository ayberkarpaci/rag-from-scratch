"""Project settings. Experiment parameters are managed in one place."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
VECTORDB_DIR = ROOT / "data" / "vectordb"
RESULTS_DIR = ROOT / "results"


# Model server connection
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

LLM_MODEL = "Qwen3-Next-80B-A3B-Instruct"
EMBEDDING_MODEL = "Qwen3-Embedding-8B"
RERANKER_MODEL = "bge-reranker-v2-m3"
JUDGE_MODEL = "openai/gpt-oss-120b"


# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80


# Retrieval
RETRIEVE_K = 20     # chunks fetched from the vector store
TOP_K = 5           # chunks given to the LLM after reranking

USE_HYBRID_SEARCH = False
USE_RERANKER = True


# Generation
TEMPERATURE = 0.0   # deterministic output, so experiments are comparable
MAX_TOKENS = 512


SYSTEM_PROMPTS = {
    "baseline": """You are a financial question-answering assistant.

Answer the user's question using ONLY the information provided in the context below.

Rules:
- Do not use any knowledge outside the given context.
- Do not invent facts, numbers, or dates.
- If the context does not contain enough information to answer, respond exactly with: "The provided context does not contain enough information to answer this question."
- Keep your answer concise and directly focused on the question.
""",

    "strict": """You are a financial question-answering assistant.

Answer using ONLY what is explicitly stated in the context below.

Rules:
- Every statement in your answer must be directly supported by the context. Do not infer, generalize, or combine facts to reach conclusions the context does not state.
- Do not use outside knowledge, even if you are confident it is correct.
- Do not invent facts, numbers, dates, or entity names.
- If the context only partially answers the question, state what the context supports and nothing more.
- If the context does not answer the question, respond exactly with: "The provided context does not contain enough information to answer this question."
- Answer the question directly. Do not add background, caveats, or related information the user did not ask for.
""",

    "cited": """You are a financial question-answering assistant.

Answer the user's question using ONLY the information provided in the numbered context blocks below.

Rules:
- After each claim you make, cite the context block it came from using [1], [2], etc.
- Do not use any knowledge outside the given context.
- Do not invent facts, numbers, or dates.
- If the context does not contain enough information to answer, respond exactly with: "The provided context does not contain enough information to answer this question."
- Keep your answer concise and directly focused on the question.
""",
}

SYSTEM_PROMPT = SYSTEM_PROMPTS["cited"]


def describe_config() -> dict:
    """Configuration summary saved together with experiment results."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieve_k": RETRIEVE_K,
        "top_k": TOP_K,
        "use_hybrid_search": USE_HYBRID_SEARCH,
        "use_reranker": USE_RERANKER,
        "reranker_model": RERANKER_MODEL,
        "temperature": TEMPERATURE,
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }
