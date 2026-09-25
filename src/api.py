"""RAG sistemi icin HTTP arayuzu."""

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import config
from src.pipeline import RAGPipeline

app = FastAPI(title="RAG Question Answering")

STATIC_DIR = ROOT / "static"
pipeline = RAGPipeline(prompt_variant="cited")


class Query(BaseModel):
    question: str = Field(min_length=1)
    # k, reranker'a giden aday sayisini (RETRIEVE_K) gecemez
    k: int = Field(default=config.TOP_K, ge=1, le=config.RETRIEVE_K)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def get_config():
    return config.describe_config()


@app.post("/api/ask")
def ask(query: Query):
    result = pipeline.answer(query.question,
                             retrieve_k=config.RETRIEVE_K,
                             top_k=query.k)

    return {
        "answer": result["answer"],
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "score": round(c["score"], 4),
                "text": c["text"],
            }
            for c in result["chunks"]
        ],
    }
