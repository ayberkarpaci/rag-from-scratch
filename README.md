# RAG Question Answering with Ragas-Based Tuning

[![tests](https://github.com/ayberkarpaci/rag-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/ayberkarpaci/rag-from-scratch/actions/workflows/tests.yml)

An end-to-end Retrieval-Augmented Generation (RAG) question answering system
written in plain Python, without a RAG framework (no LangChain or LlamaIndex
in the pipeline), plus a hyperparameter study scored with
[Ragas](https://github.com/explodinggradients/ragas).

## Results

17 measurements over 16 configurations, from the baseline to the final setup.
Every measurement finished with zero NaN scores.

| Metric | Baseline | Final | Change |
|---|---|---|---|
| Faithfulness | 0.7377 | **0.8785** | +0.141 |
| Answer Relevancy | 0.6555 | **0.7659** | +0.110 |
| Context Precision | 0.8048 | 0.7866 | -0.018 |
| Context Recall | 0.6426 | 0.6598 | +0.017 |

The full measurement log and the reasoning behind each decision are in
[`results/benchmark_report.md`](results/benchmark_report.md); notes taken
during the experiments are in [`results/experiment_notes.md`](results/experiment_notes.md).

## Final configuration

| Parameter | Value |
|---|---|
| Chunk size | 800 characters |
| Overlap | 80 characters |
| Text cleaning | On |
| Retrieval | Dense (vector), exact search |
| `retrieve_k` | 20 |
| `top_k` | 5 |
| Reranker | bge-reranker-v2-m3 |
| Hybrid search | Off (measured; it lowered the scores) |
| Prompt | With citations (`cited`) |
| Temperature | 0.0 |

## Models

| Component | Model |
|---|---|
| LLM | Qwen3-Next-80B-A3B-Instruct |
| Embeddings | Qwen3-Embedding-8B (4096 dimensions) |
| Reranker | bge-reranker-v2-m3 |
| Ragas judge | openai/gpt-oss-120b |

The models are served through an OpenAI-compatible API (for example vLLM).
The reranker needs a Cohere-compatible `/rerank` endpoint on the same server.

## Dataset

Hugging Face [`vibrantlabsai/fiqa`](https://huggingface.co/datasets/vibrantlabsai/fiqa),
config `ragas_eval_v3`: 30 financial questions with reference answers and a
corpus of about 88,400 characters.

## How it works

```
question ─► embed ─► exact cosine search (retrieve_k=20)
                          │   (optional: BM25 + reciprocal rank fusion)
                          ▼
                  cross-encoder rerank ─► top_k=5 chunks
                          ▼
             numbered context + cited prompt ─► LLM ─► answer
```

| Module | Role |
|---|---|
| `src/preprocessing.py` | Repairs punctuation where forum answers were concatenated |
| `src/chunking.py` | Recursive character splitter with overlap (natural separators first) |
| `src/embeddings.py` | Batched embedding client with a SHA-256 keyed disk cache |
| `src/vectorstore.py` | NumPy exact search over L2-normalised vectors |
| `src/bm25.py` | Okapi BM25 for the hybrid search experiment |
| `src/reranker.py` | Cross-encoder reranking with retries on rate limits |
| `src/pipeline.py` | Retrieval, rank fusion, reranking and generation |
| `src/evaluation.py` | Ragas metrics with a separate judge model |
| `src/api.py`, `static/` | FastAPI backend and a small web UI |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

or run `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1`.
Then copy `.env.example` to `.env` and fill in the API address and token:

```
LLM_BASE_URL=<api-url>
LLM_API_KEY=<token>
```

Notes:
- Networks that inspect TLS traffic break certificate checks in Python.
  `pip-system-certs` makes Python use the Windows certificate store.
- Ragas and LangChain versions depend on each other; the combination pinned
  in `requirements.txt` is the one that was tested.

## Usage

Build the data and the index once:

```bash
python scripts/build_corpus.py      # download the dataset
python scripts/build_index.py       # clean + chunk + embed + index
python scripts/test_connection.py   # check the API connection
```

Run an experiment and score it:

```bash
python scripts/run_pipeline.py --retrieve-k 20 --top-k 5 --rerank --prompt cited --name my_run
python scripts/evaluate.py --name my_run
```

Results are written to `results/pipeline_<name>.json` and
`results/eval_<name>.json`; each file records the configuration it used.

Web UI:

```bash
python -m uvicorn src.api:app --port 8000
```

Open `http://localhost:8000` and ask a question. The page shows the answer
together with the retrieved chunks and their relevance scores.

Diagnostic scripts in `scripts/`: `test_chunking.py`, `test_embeddings.py`,
`test_retrieval.py`, `test_reranker.py`, `inspect_corpus.py`,
`inspect_output.py`, and `compare_vectordb.py` (NumPy vs ChromaDB; needs
`chromadb`).

## Tests

Unit tests cover the parts that run offline: chunking, text cleaning, BM25,
the vector store and reciprocal rank fusion. They run in CI on every push.

```bash
pip install pytest
pytest
```

## Design decisions

**Plain Python.** Every layer is written by hand instead of using a RAG
framework, so each parameter can be measured directly and there is no
abstraction layer to debug through.

**Exact search.** With under 140 chunks an ANN index is unnecessary. Compared with
ChromaDB, NumPy exact search was 18 to 21 times faster and returned identical
results.

**Text cleaning.** The dataset's context blocks are several forum answers
glued together. Broken punctuation at the joins disabled the chunker's natural
sentence separators; cleaning alone added +0.054 Faithfulness.

**Citation prompt.** Asking the model to cite the source block of every claim
raised Faithfulness by 0.039. A stricter "do not infer" prompt lowered it by
0.186 instead.

**Reranker choice.** The first candidate, Qwen3-Reranker-8B, produced
irrelevant orderings in testing (see `results/experiment_notes.md`), so
bge-reranker-v2-m3 is used.
