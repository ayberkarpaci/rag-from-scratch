# Experiment Notes

Technical notes kept throughout the experiments, in the order they were
written. The condensed version with all results is in
[`benchmark_report.md`](benchmark_report.md).

---

## Dataset

- `vibrantlabsai/fiqa` (`ragas_eval_v3`) has no separate document corpus. It
  only has 30 questions, each with a `retrieved_contexts` field. The knowledge
  base to index was built from these contexts. This was the first
  architectural decision.
- 30 documents in total, about 88,400 characters. Exactly one context per row,
  2,946 characters on average.
- The context blocks are several forum answers joined together. There is no
  separator between answers; at some joins there is not even a space after the
  full stop (`...to be there.Anybody can deposit...`). This keeps the natural
  sentence separator from applying.

## Environment

- On networks that inspect TLS traffic, Python HTTPS requests fail certificate
  checks; solved with `pip-system-certs` + `certifi.where()`.
- The `truststore` package is incompatible with Python 3.13
  (`RecursionError`). It was replaced with `certifi`.

## Model infrastructure

- The models are served by vLLM with an OpenAI-compatible API.
- Embedding dimension: 4096.
- The judge model (`openai/gpt-oss-120b`) returns a `reasoning` field next to
  the answer. The thinking step uses tokens; if `max_tokens` is too low,
  `content` comes back empty. This matters for the Ragas calls.
- `Qwen3-Next-80B-A3B-Instruct` is used as the generation model. The larger
  `Qwen3.5-397B-A17B-FP8` is also available on the server.

## Embedding check

A measurement showing how semantic search differs from keyword search:

| Pair | Similarity |
|---|---|
| "net profit increased by 27 percent" - "revenue showed significant growth" | 0.6149 |
| "net profit increased by 27 percent" - "the cat is sleeping on the sofa" | 0.2272 |

The two financial sentences share no words; classic keyword search would find
no match. The embedding captures the semantic closeness.

## Chunking approach

Recursive splitting is used. The separators form a hierarchy ordered from the
most natural to the most forced:

```
["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
```

The algorithm first tries to split on paragraph boundaries (`\n\n`). If a
resulting piece is still larger than the target size, the function calls
itself on that piece with the next separator. Splits therefore happen at the
most natural point available; forced character splits are only a last resort.

The aim is to keep meaning intact: at a paragraph boundary a thought is
already complete, while cutting mid-sentence breaks the context.

This is the same approach as LangChain's `RecursiveCharacterTextSplitter`,
rewritten in plain Python for this project.

### A limitation caused by the corpus

The dataset's context blocks are several forum answers joined together. At the
joins there is no space after the full stop:

```
...you don't even technically have to be there.Anybody can deposit money...
...every bank will say the same thing.To do this, I need a state-issued...
```

Here the `". "` separator does not match and the algorithm falls back a level
(comma, space). Answers from different people can then end up mixed in the
same chunk, which is one of the things that lowers Context Precision.

This observation later led to the corpus cleaning experiment (see experiment
series 6).

## Retrieval observations (baseline: chunk=500, overlap=50, k=3)

For the first three questions, the chunk from the right document came first.
However:

- **The scores for q_002 are very close:** the right chunk 0.5776, the second
  0.5763, the third (from a wrong document) 0.5620. The gaps are in the third
  decimal place. The ranking is fragile and could easily flip with a parameter
  change; a typical case where a reranker can help.
- **q_000 got chunks on the right topic from the wrong source:** positions 2
  and 3 are chunks from `doc_008`. They cover the same topic (cheques / bank
  accounts) but belong to a different question. This will lower Context
  Precision.

## Expected limits

The corpus has 30 independent blocks and each question's answer is in its own
block. The retrieval task is relatively easy: one needle in 30 haystacks. The
baseline scores are expected to start high and optimization effects to show
up as small differences. Keep this in mind when reading the results.

## Reproducibility

### temperature = 0.0

All LLM calls use `temperature=0`. Reason: in hyperparameter optimization
the source of a score change must not be ambiguous.

With temperature > 0 the model can pick different words on every call. A score
change when going from k=5 to k=10 could then be either the parameter or the
model's randomness, and there would be no way to tell. Deterministic output
ensures that a measured difference comes only from the parameter that was
changed.

The cost: generated answers may be a little more mechanical. In an
evaluation-focused project that is an acceptable trade-off.

### Embedding cache

The vector for a given text is computed once and written to disk
(`data/cache/embeddings.json`). The cache key is a SHA-256 hash of the model
name plus the text.

Including the model name in the key is essential: after switching to a
different embedding model, the cache will not return the old vectors.
Otherwise a silent measurement error that is hard to detect would occur.

Effect: experiments with fixed chunk parameters (k, prompt, reranker) skip
the embedding step completely. The first indexing takes 9.7 seconds; later
runs are instant.

### Prompt structure

The context blocks are numbered (`[1]`, `[2]`, ...). Because the dataset's
contexts are several forum answers joined together, numbering helps the model
keep separate sources apart.

The context comes first and the question last. Models tend to miss
information in the middle of long inputs ("lost in the middle"), so the
question is kept closest to where the answer is generated.

## First end-to-end query (baseline: chunk=500, overlap=50, k=5)

The answer generated for q_000 largely matches the ground truth. The model
kept the nuance of the context: it said that some banks such as Bank of
America *may claim* this is a federal regulation, rather than stating it as
fact. The source text was indeed a forum user's claim. The system prompt's
rule about staying faithful to the context works.

However, 2 of the 5 retrieved chunks came from `doc_008` (scores 0.7739 and
0.7526). Same topic (cheques / business accounts), but a document belonging to
a different question. Chunks like this, on the right topic from the wrong
source, will lower Context Precision. They are a priority target for the
reranker and the k optimization.

## Baseline pipeline run (chunk=500, overlap=50, k=5)

- 30 questions, 186 seconds (6.2 s/question). The pipeline costs about 3
  minutes per experiment; the number of experiments should be planned around
  this.
- No empty answers. Mean answer length 509 characters (min 81, max 1182).
- **6 questions got a "not enough information" answer**: q_002, q_011, q_015,
  q_020, q_027, q_028 (20%).

There are two possible explanations for this rate:

1. The model is being honest; when retrieval brings the wrong context it holds
   back instead of making something up. This protects Faithfulness.
2. Retrieval really fails; the right information is in the corpus but is not
   found. Context Recall and Answer Relevancy come out low.

The Ragas measurement will settle this. q_002 stands out: its scores were also
very close in the retrieval test (0.5776 / 0.5763 / 0.5620). The weak match
signal is consistent across both measurements.

## Ragas version compatibility

`pip install ragas` installed the latest version (0.4.3), but that version
expects the old layout of `langchain-community`. The newly installed versions
(langchain-community 0.4.2, langchain-openai 1.1.9) had removed those modules,
which caused import errors:

```
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
ImportError: cannot import name 'ContextOverflowError' from 'langchain_core.exceptions'
```

The fix was to pin a compatible combination of versions explicitly:

| Package | Version |
|---|---|
| ragas | 0.2.14 |
| langchain-community | 0.3.14 |
| langchain-openai | 0.2.14 |
| langchain-core | 0.3.63 |

Lesson: in fast-moving ecosystems, dependency versions must be pinned in
`requirements.txt`. Otherwise the same code may not work when installed on a
different date.

## Ragas run settings

The first attempts hit two kinds of errors:

- `RateLimitError`: the service allows 15 requests per window. `max_workers=4`
  exceeded this, so it was lowered to 2.
- `LLMDidNotFinishException`: because the judge model produces reasoning,
  `max_tokens` was too low. It was raised step by step: 2048 -> 4096 -> 8192.

In faulty runs, failed jobs come back as NaN and are left out of the mean.
This biases the scores in an unknown direction. Three faulty runs on the same
data gave Faithfulness values as different as 0.61 / 0.64 / 0.67. For this
reason every measurement reports its NaN count, and any measurement with a
non-zero count is treated as invalid.

More parallelism did not change the run time (no difference between
max_workers 2 and 4). The bottleneck is the service's response time, which
slows down noticeably at busy hours.

## Baseline results (chunk=500, overlap=50, k=5)

| Metric | Score |
|---|---|
| Context Precision | 0.8056 |
| Faithfulness | 0.7349 |
| Answer Relevancy | 0.6595 |
| Context Recall | 0.6486 |

The measurement is clean: all 120 jobs completed, no NaN on any metric.
Time: 47 minutes (95 s/question).

### Interpretation

Context Recall is the weakest link. About a third of the needed information is
never retrieved. The 6 "not enough information" answers in the pipeline output
are consistent with this score: when retrieval fails, the model holds back
instead of making things up.

The low Answer Relevancy is largely a consequence of Recall. When the answer is
not in the context, the model either answers incompletely or declines; both
lower this metric.

Precision is high (0.8056) and Recall is low (0.6486). The system retrieves
"little but accurate". k=5 is probably too narrow. The first optimization
target is Context Recall.

## Measurement noise

The same pipeline output was evaluated twice (same data, same settings,
temperature=0). The aim: measure how consistently the judge model scores.

| Metric | Run 1 | Run 2 | Difference |
|---|---|---|---|
| Faithfulness | 0.7349 | 0.7405 | +0.0056 |
| Answer Relevancy | 0.6595 | 0.6515 | -0.0080 |
| Context Precision | 0.8056 | 0.8040 | -0.0016 |
| Context Recall | 0.6486 | 0.6365 | -0.0121 |

**Noise level: about ±0.012.**

Even with `temperature=0` the scores are not exactly the same. On the vLLM
side, batching and the order of floating-point operations do not guarantee
full determinism; the judge model also produces reasoning, and that step is
prone to variation.

**Rule for reading experiments:** differences above 0.03 are treated as
meaningful. The 0.01-0.02 range is within the noise band; such differences
need a repeat measurement.

Context Precision is the most stable metric (±0.0016) and Context Recall the
most variable (±0.0121). Recall involves more interpretation, because it
splits the ground truth into claims and looks for each one in the context.

## Hallucination check

A question outside the corpus was asked: "What is the capital of France?"

System answer: "The provided context does not contain enough information to
answer this question."

The model knows the answer from its training data but did not use it, because
of the system prompt's "use only the given context" rule.

Retrieval still returned 5 chunks (it always returns the nearest k), but their
similarity scores were in the 0.19-0.23 range. For comparison, the top score
for a question inside the corpus was 0.7986. The score range clearly shows
that there was no match.

This observation confirms two things:

1. The system prompt's hallucination control works.
2. The similarity score is a usable signal for detecting retrieval failure.
   Dropping chunks below a threshold might improve Context Precision. (This
   idea was tested later; see "Reranker score threshold".)

---

# Experiment series 1: sweeping k

Fixed settings: chunk_size=500, overlap=50, temperature=0, no reranker, no
hybrid search. Only the number of retrieved chunks (k) was changed.

| Metric | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| Context Precision | **0.8444** | 0.8048 | 0.7406 | 0.7184 |
| Context Recall | 0.5479 | 0.6426 | 0.7360 | **0.7535** |
| Faithfulness | 0.7079 | 0.7377 | 0.7245 | **0.7475** |
| Answer Relevancy | 0.6550 | 0.6555 | **0.6812** | 0.6291 |
| Retrieval F1 | 0.6647 | 0.7137 | **0.7383** | 0.7353 |
| Evaluation time | 39 min | 47 min | 68 min | 111 min |

(The k=5 values are the mean of two independent measurements. NaN count 0 in
every measurement. Noise band ±0.012.)

## Retrieval F1

Context Precision and Context Recall trade off directly: as k grows, one rises
and the other falls. To summarize both in one number, the harmonic mean (F1)
is used:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The harmonic mean was chosen over the arithmetic mean because the arithmetic
mean does not penalize lopsided configurations. For Precision = 0.90 and
Recall = 0.30, the arithmetic mean gives 0.60 and the harmonic mean 0.45. The
second value reflects the situation better, because a retrieval system that is
weak on one side is bad in practice.

This is the standard way to summarize precision and recall in the information
retrieval literature.

No similar combination was made for Faithfulness and Answer Relevancy; there is
no structural trade-off between those two metrics, so a harmonic mean would
have no theoretical basis.

## Observations

**The precision-recall trade-off is clearly visible.** As k grows, Precision
falls monotonically (0.8444 -> 0.7184) and Recall rises monotonically
(0.5479 -> 0.7535). Textbook behaviour.

**k=3 is the worst option.** Precision reaches its highest value, but Recall
collapses (0.5479) and drags Faithfulness down with it (0.7079, the lowest in
the series). Interpretation: when the context lacks information, the model
tries to fill the gaps. A "little but accurate" strategy does not work on this
dataset.

**Answer Relevancy peaks at k=10 and drops clearly at k=20**
(0.6812 -> 0.6291, -0.052). That loss is well above the noise band. Likely
explanation: models tend to miss information in the middle of a long context
("lost in the middle"). With 20 chunks of context the model loses focus and
the answer drifts.

**No clear trend in Faithfulness** (0.7079 / 0.7377 / 0.7245 / 0.7475). Most
differences are close to the noise band. k had no clear effect on
hallucination.

## Decision: k = 10

Three reasons:

1. **Retrieval F1 peaks.** F1 is highest at k=10 (0.7383). At k=20 it stopped
   rising and slipped slightly (0.7353). Going from k=10 to k=20 gains +0.018
   Recall (close to the noise band) and loses -0.022 Precision, so the extra
   cost buys nothing.

2. **Answer Relevancy is highest at k=10.** The -0.052 drop at k=20 is a
   meaningful degradation. For generation quality, k=10 is clearly better.

3. **Cost.** The k=20 measurement took 111 minutes, 1.6 times as long as
   k=10. With no marginal benefit, that increase is not acceptable. It also
   means twice the tokens on every request.

After this decision the `config.py` defaults were set to `RETRIEVE_K = 10`,
`TOP_K = 10`, and the next experiments kept this value fixed.

## Methodological note

k was swept over 3, 5, 10 and 20; not every integer in between was tried.
Reason: measurement noise is about ±0.012, and differences between
neighbouring values of k are expected to stay below that band. For example,
the F1 difference between k=5 and k=10 is 0.025; single-step differences would
be about a fifth of that and could not be told apart. A coarse sweep that shows
the shape of the curve gives the same information in a third of the time.

---

# Experiment series 2: chunk size

Fixed settings: k=10, temperature=0, no reranker, no hybrid search. Overlap was
kept proportional at 10% of chunk_size in every experiment.

## chunk_size = 200 (overlap 20)

| Metric | chunk=500 | chunk=200 | Difference |
|---|---|---|---|
| Context Precision | 0.7406 | 0.6717 | -0.069 |
| Context Recall | 0.7360 | 0.4881 | -0.248 |
| Faithfulness | 0.7245 | 0.6656 | -0.059 |
| Answer Relevancy | 0.6812 | 0.6335 | -0.048 |
| Retrieval F1 | 0.7383 | 0.5653 | -0.173 |

All four metrics fell; nothing improved. This was unexpected: smaller chunks
should at least have raised Precision.

**Interpretation.** 200 characters is below the threshold of semantic
completeness. A chunk can no longer hold a whole idea; it starts and ends
mid-sentence. When the judge model decides "is this chunk relevant to the
question?", it finds the fragment meaningless on its own and counts it as
irrelevant. That explains why Precision fell as well.

The collapse in Recall (-0.248) is more direct: with k=10 fixed, the model
still sees 10 chunks, but each is very small. The total context is less than
half of the 500-character configuration. The information is both fragmented
and insufficient.

**Lesson:** chunking is not only about size; there is a threshold of semantic
completeness. On this dataset, 200 characters is below it.

chunk=300 was not tried. The curve clearly falls towards 200, so 300 is
expected to stay below 500 as well; the measurement budget went to values
above 500.

## chunk_size = 800 (overlap 80)

| Metric | chunk=200 | chunk=500 | chunk=800 |
|---|---|---|---|
| Context Precision | 0.6717 | 0.7406 | **0.8240** |
| Context Recall | 0.4881 | **0.7360** | 0.7227 |
| Faithfulness | 0.6656 | **0.7245** | 0.7232 |
| Answer Relevancy | 0.6335 | **0.6812** | 0.6290 |
| Retrieval F1 | 0.5653 | 0.7383 | **0.7701** |

**Precision rose clearly (+0.083).** Interpretation: 800 characters is large
enough to hold a typical forum answer from the dataset in one piece. The judge
model sees a self-contained text when assessing the chunk, and the relevance
decision becomes clearer.

**Recall barely changed (-0.013, within the noise band).** The expectation was
that Recall would fall, since with k=10 larger chunks cover fewer documents.
That did not happen. Because the corpus has 30 independent blocks and each
question's answer is in its own block, larger chunks may capture the right
block more completely.

**Retrieval F1 of 0.7701 is the best in the series** (previous best: 0.7383
at k=10, chunk=500).

**Answer Relevancy fell (-0.052).** That is the same size as the drop seen in
the k=20 experiment. What they share is the total context length: k=10 × 800
characters gives about 8000 characters of context. The model losing focus in a
long context ("lost in the middle") is a consistent explanation.

This shows that chunk size and k are not independent: what matters is their
product, the total context length. The next experiment tests this hypothesis
(chunk=800, k=5).

## chunk_size = 800, k = 5

The hypothesis from the previous experiment: what matters is not chunk size
or k, but their product, the total context length given to the model. This
experiment tests it: chunks stay large while k is halved.

| Metric | 500/k5 | 500/k10 | 800/k10 | **800/k5** |
|---|---|---|---|---|
| Context Precision | 0.8048 | 0.7406 | 0.8240 | **0.8528** |
| Context Recall | 0.6426 | **0.7360** | 0.7227 | 0.7073 |
| Faithfulness | 0.7377 | 0.7245 | 0.7232 | **0.7600** |
| Answer Relevancy | 0.6555 | 0.6812 | 0.6290 | **0.6977** |
| Retrieval F1 | 0.7137 | 0.7383 | 0.7701 | **0.7733** |

**Hypothesis confirmed.** With the total context cut from 8000 characters
(800×10) to 4000 (800×5):

- Answer Relevancy 0.6290 -> 0.6977 (+0.069)
- Faithfulness 0.7232 -> 0.7600 (+0.037)
- Context Precision 0.8240 -> 0.8528 (+0.029)

All three improved meaningfully. Recall moved by -0.015, within the noise
band, so there is practically no loss.

**Interpretation.** With less but denser context the model works noticeably
more focused. The drift seen with 8000 characters ("lost in the middle")
disappears at 4000. Less noise also means fewer chances to stray, which raises
Faithfulness.

**Total gain over the baseline** (500/k5 -> 800/k5):

| Metric | Difference |
|---|---|
| Context Precision | +0.048 |
| Context Recall | +0.065 |
| Faithfulness | +0.022 |
| Answer Relevancy | +0.042 |
| Retrieval F1 | +0.060 |

All four metrics went up. This is a general improvement, not an optimization
that raises one metric at the expense of the others.

## chunk_size = 1000, k = 4

The total context is again about 4000 characters, but in larger pieces. The
aim: push the "semantic completeness" hypothesis one step further.

| Metric | 800/k5 | 1000/k4 | Difference |
|---|---|---|---|
| Context Precision | 0.8528 | **0.8741** | +0.021 |
| Context Recall | 0.7073 | **0.7307** | +0.023 |
| Retrieval F1 | 0.7733 | **0.7960** | +0.023 |
| Faithfulness | **0.7600** | 0.6661 | -0.094 |
| Answer Relevancy | **0.6977** | 0.6182 | -0.080 |

**Retrieval improved, generation degraded.** Both retrieval metrics rose and F1
reached the best value in the series (0.7960). Faithfulness and Answer
Relevancy, however, fell far beyond the noise band.

**Interpretation.** 1000-character chunks help retrieval: a piece holds a
whole answer, so the judge can assess relevance easily. The same size hurts
generation: each chunk carries several ideas and irrelevant parts. The model
has to scan four long texts to extract the right information and drifts from
the context while doing so.

**Lesson:** the retrieval optimum and the generation optimum of chunk size are
not at the same point. Better retrieval metrics do not guarantee a better
system overall, which shows that optimizing on F1 alone or on retrieval scores
alone can be misleading.

## Chunk series decision: 800 / k=5

| Metric | 200/k10 | 500/k5 | 500/k10 | 800/k10 | **800/k5** | 1000/k4 |
|---|---|---|---|---|---|---|
| Ctx Precision | 0.6717 | 0.8048 | 0.7406 | 0.8240 | 0.8528 | **0.8741** |
| Ctx Recall | 0.4881 | 0.6426 | **0.7360** | 0.7227 | 0.7073 | 0.7307 |
| Faithfulness | 0.6656 | 0.7377 | 0.7245 | 0.7232 | **0.7600** | 0.6661 |
| Answer Rel. | 0.6335 | 0.6555 | 0.6812 | 0.6290 | **0.6977** | 0.6182 |
| Retrieval F1 | 0.5653 | 0.7137 | 0.7383 | 0.7701 | 0.7733 | **0.7960** |

800/k5 was chosen because it is best on three of the four metrics. 1000/k4 is
0.023 ahead on retrieval F1 but 0.08-0.09 behind on Faithfulness and Answer
Relevancy. Since the goal is to optimize all four metrics together, maximizing
retrieval F1 alone would be wrong.

---

# Experiment series 3: hybrid search (BM25 + dense)

Fixed: chunk=800, overlap=80, k=5. BM25 and vector search were run separately
and their results merged with Reciprocal Rank Fusion (RRF). k×2 candidates
were taken from each method, and the first k after fusion were kept.

| Metric | Dense (800/k5) | Hybrid | Difference |
|---|---|---|---|
| Context Precision | **0.8528** | 0.8001 | -0.053 |
| Context Recall | **0.7073** | 0.6911 | -0.016 |
| Faithfulness | **0.7600** | 0.7188 | -0.041 |
| Answer Relevancy | **0.6977** | 0.6517 | -0.046 |
| Retrieval F1 | **0.7733** | 0.7416 | -0.032 |

(The first run had 1 NaN in Answer Relevancy: the judge model's output format
could not be parsed for one question. The measurement was repeated and came
out clean.)

**Result: hybrid search lowered performance on this dataset.**

**Interpretation.** BM25 was expected to cover the areas where embeddings are
weak (code, abbreviations, proper names). That did not happen. A likely
explanation: the corpus is made of forum answers, and common financial words
("account", "bank", "money", "business") appear everywhere. BM25 promotes the
chunks where these words cluster, but because they are common they do not
discriminate. Keyword matching gives a weaker signal here than semantic
matching.

The second factor is that RRF weights both lists equally. Dense search alone
already works well, and mixing in a weaker second signal at equal weight
spoils the ranking.

**Decision:** hybrid search is not used; dense (vector) search only.
`USE_HYBRID_SEARCH = False`.

---

# Vector database comparison

RAG systems usually use a ready-made vector database such as ChromaDB, FAISS
or LanceDB. This project uses NumPy exact search instead, and the decision was
verified by measurement: ChromaDB was set up with the same 138 chunks and the
same vectors, and 30 queries were run with both methods.

| Method | Mean query time |
|---|---|
| NumPy (exact, brute force) | 0.203 ms |
| ChromaDB (HNSW) | 4.374 ms |

- NumPy is 21 times faster
- Result overlap: 100% (both methods return the same chunks)
- ChromaDB indexing time: 1.07 seconds

The measurement used `retrieve_k = 5` (after the chunk series decision, before
the reranker experiments). A second run measured 0.282 ms for NumPy and 5.167
ms for ChromaDB; the ratio held.

**Interpretation.** ANN (approximate nearest neighbour) indexes are an
optimization for searching among millions of vectors. With 138 vectors the
index itself is overhead: ChromaDB's slowness comes not from the HNSW
algorithm but from the cost of its layers (SQLite persistence, serialization,
API calls). At this scale that cost is larger than the search itself.

Exact search also returns exact results, while ANN is approximate. At this
scale there is no reason to accept approximation.

**Decision:** NumPy exact search. Besides performance, it also means fewer
dependencies and full control.

---

# Choosing the reranker model

Two rerankers are available on the server: `Qwen3-Reranker-8B` (first
choice) and `bge-reranker-v2-m3`. Both work through the `/rerank` endpoint in
a Cohere-compatible format.

A small test was run to check the format. Query: "How do I deposit a third
party cheque?" with three candidate documents.

| Document | Qwen3-Reranker-8B | bge-reranker-v2-m3 |
|---|---|---|
| "The weather in Ankara is cold in winter." | **0.903** (1st) | 0.000017 (3rd) |
| "Just have the associate sign the back..." | 0.880 (2nd) | 0.023 (2nd) |
| "A third party cheque requires endorsement..." | 0.562 (3rd) | **0.807** (1st) |

**Qwen3-Reranker-8B produced irrelevant results.** It ranked the weather
sentence, which has nothing to do with the topic, as the most relevant and the
sentence with the direct answer as the least relevant. The closeness of the
scores (0.90 / 0.88 / 0.56) also shows that it is not separating anything.

**bge-reranker-v2-m3 worked correctly.** It put the direct answer first with
0.807 and the irrelevant sentence last with 0.000017. The difference in
magnitude between the scores is a clear sign of separation.

**Possible explanation:** the Qwen3-Reranker family expects a special prompt
template (`Instruct: ... Query: ... Document: ...`). The server may be passing
the raw text straight through, in which case the model does not produce
meaningful scores.

**Decision:** `bge-reranker-v2-m3`. The model choice was changed based on
measurement.

---

# Experiment series 4: reranker

Fixed: chunk=800, overlap=80, hybrid off. Model: `bge-reranker-v2-m3`.
Approach: retrieve_k candidates are taken from vector search, reordered by the
cross-encoder, and the top_k are given to the LLM.

## retrieve_k = 20 -> top_k = 5

| Metric | No reranker (800/k5) | Reranker (20→5) | Difference |
|---|---|---|---|
| Faithfulness | 0.7600 | **0.7850** | +0.025 |
| Answer Relevancy | 0.6977 | **0.7219** | +0.024 |
| Context Precision | **0.8528** | 0.8081 | -0.045 |
| Context Recall | **0.7073** | 0.6858 | -0.022 |
| Retrieval F1 | **0.7733** | 0.7419 | -0.031 |

**The expectation was not met.** The reranker was meant to protect Precision
by filtering a wide candidate pool (high Recall). Both retrieval metrics fell.

**Generation improved clearly instead.** Faithfulness and Answer Relevancy
both rose by about +0.025, each above the noise band (±0.012). Faithfulness at
0.7850 and Answer Relevancy at 0.7219 were the highest values in the series so
far.

**Interpretation.** Ragas's retrieval metrics and generation quality split
apart here. The reranker may be promoting the chunks that actually help the
model answer, rather than those the judge marks as "relevant". The two do not
always coincide.

The same split appeared in the opposite direction in the chunk=1000/k=4
experiment: there retrieval metrics rose while generation collapsed. Together,
the two findings show that optimizing on retrieval scores alone is misleading.

## retrieve_k = 10 -> top_k = 5

| Metric | 20→5 | 10→5 |
|---|---|---|
| Faithfulness | 0.7850 | 0.7848 |
| Answer Relevancy | 0.7219 | 0.7220 |
| Context Precision | **0.8081** | 0.7924 |
| Context Recall | **0.6858** | 0.6682 |
| Retrieval F1 | **0.7419** | 0.7250 |

The generation metrics are practically identical (a difference of about
0.0002). This shows that the reranker picks largely the same chunks from both
candidate pools: the same chunks rise to the top out of 20 candidates as out
of 10.

The retrieval metrics are somewhat lower with the narrow pool, so the wide
candidate pool was preferred.

## Reranker decision

| Metric | No reranker (800/k5) | Reranker (20→5) |
|---|---|---|
| Faithfulness | 0.7600 | **0.7850** |
| Answer Relevancy | 0.6977 | **0.7219** |
| Context Precision | **0.8528** | 0.8081 |
| Context Recall | **0.7073** | 0.6858 |
| Retrieval F1 | **0.7733** | 0.7419 |
| Mean of four metrics | 0.7545 | 0.7502 |

The plain mean of the four metrics is almost the same. The decision depends
on which metrics are prioritized.

**The reranker is used.** Reasoning: Faithfulness and Answer Relevancy measure
the output the end user actually sees. Context Precision and Recall are
intermediate metrics; the user cares about whether the answer is correct and
on topic, not about how accurate the retrieved chunks are. Faithfulness works
directly as a hallucination check, and in finance a made-up answer is costly.

This decision is debatable; the scores of both configurations are shown side
by side in `benchmark_report.md`.

Chosen configuration: `retrieve_k=20`, `top_k=5`, `bge-reranker-v2-m3`.

---

# Experiment series 5: prompt engineering

Fixed: chunk=800, retrieve_k=20 → top_k=5, reranker on.

## Variant "strict": strict faithfulness rules

The aim was to raise Faithfulness. These rules were added to the baseline
prompt: "Do not infer, do not generalize, do not combine facts to reach
conclusions the context does not state." The model was also told not to add
background information that was not asked for.

| Metric | baseline | strict | Difference |
|---|---|---|---|
| Faithfulness | **0.7850** | 0.5986 | -0.186 |
| Answer Relevancy | **0.7219** | 0.4720 | -0.250 |
| Context Precision | 0.8081 | **0.8174** | +0.009 |
| Context Recall | **0.6858** | 0.6487 | -0.037 |

**The target metric moved the wrong way.** A prompt written to raise
Faithfulness lowered it by 0.186.

**Mechanism.** The number of questions answered with "not enough information"
rose from 6 to 12 (40% of the questions). For example, in q_000 the context
contains the right answer (a third-party cheque can be endorsed and
deposited), but the model replied that the context does not confirm whether
this applies to depositing a cheque made out to a partner into a business
account.

The model over-interpreted the "do not infer" rule and refused even the
smallest step needed to connect the information in the context to the
question.

**Why Faithfulness fell.** Ragas computes this metric as "how many of the
claims in the answer are supported by the context". The model produced
meta-claims such as "the context does not confirm this" and "it does not
state this explicitly". These are comments about the context, not factual
claims that can be derived from it. The judge counts them as unsupported and
the score drops.

**Lesson:** an overly restrictive prompt can damage the very metric it
targets. The balance between preventing hallucination and making the model
useless is narrow. The baseline prompt's permission to say "there is not
enough information" is enough; adding a ban on inference on top of it takes
away the system's ability to answer.

## Variant "cited": citing sources

A different mechanism was tried: instead of adding restrictions, the model was
asked to end each claim with the context block it came from, in the form
`[1]`, `[2]`.

| Metric | baseline | strict | **cited** |
|---|---|---|---|
| Faithfulness | 0.7850 | 0.5986 | **0.8244** |
| Answer Relevancy | **0.7219** | 0.4720 | 0.7025 |
| Context Precision | 0.8081 | **0.8174** | 0.8173 |
| Context Recall | 0.6858 | 0.6487 | **0.7025** |
| Retrieval F1 | 0.7419 | 0.7233 | 0.7556 |
| Mean of four metrics | 0.7502 | 0.6342 | **0.7617** |

**Faithfulness reached the highest value in the project so far** (0.8244,
+0.039). Above the noise band, a real gain.

**Mechanism.** When the model has to name the source of every claim, it
becomes harder to say something the context does not support. The need to
cite forces it back to the context throughout generation.

Answer Relevancy fell slightly (-0.019, close to the noise band), probably
because the `[1]`, `[2]` markers mix into the text.

Context Recall rose (+0.017). Retrieval did not change, so this is not a
direct retrieval effect: because the model cites more blocks, a larger share
of the ground-truth claims may find a match in the answer.

## What the prompt experiments showed

The two variants went in opposite directions, and the difference is
instructive:

- **strict** told the model *what it may not do* (do not infer, combine or
  generalize). The model became dysfunctional, refused to answer 40% of the
  time, and the target metric fell by 0.186.
- **cited** showed the model *how to do the task* (write every claim with its
  source). Faithfulness rose by 0.039.

Giving the prompt structure worked better than giving it restrictions.

---

# Experiment series 6: corpus cleaning

The dataset's context blocks are several forum answers joined together, with
broken punctuation at the joins. A scan of the corpus found:

| Defect | Count |
|---|---|
| Doubled quotes (`""`) | 104 |
| Letter directly after a quote | 87 |
| Repeated spaces | 260 |
| Sentence end without a space (`.X`) | 25 |

Example: `...Before you convert to S-Corp.You don't need to notify the IRS...`

These defects disable the `". "` separator of the recursive chunker; the
algorithm falls back a level (comma, space) and chunk boundaries land
mid-sentence.

Cleaning applied: collapse doubled quotes, add a space after sentence-ending
punctuation, add a space after a quote, collapse repeated spaces.

Result: 138 chunks -> 137 chunks. The text of 85% of the chunks changed.

| Metric | Not cleaned | **Cleaned** | Difference |
|---|---|---|---|
| Faithfulness | 0.8244 | **0.8785** | +0.054 |
| Answer Relevancy | 0.7025 | **0.7659** | +0.063 |
| Context Precision | **0.8173** | 0.7866 | -0.031 |
| Context Recall | **0.7025** | 0.6598 | -0.043 |
| Retrieval F1 | **0.7556** | 0.7176 | -0.038 |
| Mean of four metrics | 0.7617 | **0.7727** | +0.011 |

**The generation metrics improved clearly.** Faithfulness at 0.8785 and Answer
Relevancy at 0.7659 are the highest values in the project.

**Interpretation.** Chunk boundaries no longer fall mid-sentence. The model
reads whole sentences instead of dealing with fragments. Cleaning up the
broken quotes also makes the text easier to read.

The drop in the retrieval metrics comes from the shifted chunk boundaries
changing how chunks line up with the ground truth. The added spaces also make
the text longer, so slightly less content fits into the same 800 characters.

**Decision: cleaning is applied.** The reasoning is the same as for the
reranker: the generation metrics measure the output the user actually sees.
Here the difference is even clearer (generation +0.05 and +0.06).

## top_k = 3 (with the reranker)

Hypothesis: since the reranker already picks the best candidates, giving fewer
chunks could reduce noise and raise Faithfulness.

| Metric | top_k=5 | top_k=3 |
|---|---|---|
| Faithfulness | **0.8785** | 0.8492 |
| Answer Relevancy | 0.7659 | **0.7698** |
| Context Precision | 0.7866 | **0.8222** |
| Context Recall | **0.6598** | 0.6095 |
| Retrieval F1 | **0.7176** | 0.7000 |
| Mean | **0.7727** | 0.7627 |

**Not confirmed.** Faithfulness fell instead (-0.029). The -0.050 drop in
Recall explains why: three chunks are not enough, part of the answer is
missing from the context, and the model tries to fill the gap.

This repeats the pattern from the k sweep: too little context lowers
Faithfulness (the k=3 experiment also had the lowest Faithfulness). Without
enough information the model cannot stay faithful.

top_k=5 was kept.

## Reranker score threshold

The distribution of reranker scores was examined: median 0.0268, minimum
0.0003, maximum 0.9987. Half of the chunks given to the model are, according
to the reranker, almost irrelevant.

Hypothesis: set the number of contexts dynamically with a score threshold
instead of a fixed top_k. Give many chunks when many are relevant and few when
few are.

With a threshold of 0.1, the average number of contexts fell from 5 to 2.2
(min 1, max 5).

| Metric | No threshold | Threshold 0.1 |
|---|---|---|
| Faithfulness | **0.8785** | 0.8602 |
| Answer Relevancy | 0.7659 | **0.7681** |
| Context Precision | 0.7866 | **0.8046** |
| Context Recall | **0.6598** | 0.5582 |
| Retrieval F1 | **0.7176** | 0.6591 |
| Mean | **0.7727** | 0.7478 |

**Not confirmed.** Context Recall lost heavily (-0.102). An average of 2.2
chunks is not enough, and a large part of the needed information is filtered
out. Faithfulness fell too (-0.018), the same mechanism as in the top_k=3
experiment.

The +0.018 gain in Precision does not make up for these losses.

**Lesson:** the absolute values of reranker scores are not a reliable measure
of relevance. A low-scoring chunk can still carry part of the answer even
though it ranks lower. Relative ranking (top_k) works better than an absolute
threshold.

---

# Final configuration and total gain

| Parameter | Baseline | Final |
|---|---|---|
| Chunk size | 500 | 800 |
| Overlap | 50 | 80 |
| Text cleaning | No | Yes |
| retrieve_k | 5 | 20 |
| top_k | 5 | 5 |
| Reranker | No | bge-reranker-v2-m3 |
| Hybrid search | No | No (measured, rejected) |
| Prompt | baseline | cited |
| Temperature | 0.0 | 0.0 |

| Metric | Baseline | Final | Difference |
|---|---|---|---|
| Faithfulness | 0.7377 | **0.8785** | +0.141 |
| Answer Relevancy | 0.6555 | **0.7659** | +0.110 |
| Context Precision | 0.8048 | 0.7866 | -0.018 |
| Context Recall | 0.6426 | 0.6598 | +0.017 |

17 measurements (16 configurations) in total, all with zero NaN scores. The
table of all measurements is in `benchmark_report.md`.

## General conclusions

**1. Retrieval metrics and generation quality do not always move together.**
This split was seen in three separate experiments: chunk=1000/k=4 (retrieval
up, generation collapsed), the reranker (retrieval down, generation up) and
corpus cleaning (the same direction as the reranker). Optimizing on retrieval
scores alone is misleading.

**2. Total context length matters more than chunk size or k on their own.**
The best results came when their product stayed around 4000 characters. At
8000 characters the "lost in the middle" effect appeared; at 2400 characters
there was not enough information.

**3. Giving the prompt structure works better than giving it restrictions.**
A list of prohibitions (strict) made the model dysfunctional; citing sources
(cited) raised the target metric.

**4. Data quality gave a larger gain than parameter tuning.** Corpus cleaning
alone added +0.054 Faithfulness; no parameter change was as effective.

**5. Negative results are results too.** Three of the six approaches tried
were rejected (hybrid search, top_k=3, score threshold). Measuring an idea and
rejecting it with a reason, rather than rejecting it untested, makes the
architecture decisions easier to defend.

## Limitations

- The dataset has 30 examples; statistical power is low.
- A single judge model (`openai/gpt-oss-120b`); scores could change with a
  different judge.
- Measurement noise of ±0.012; differences below this band were not
  interpreted.
- Context Recall stayed around 0.66. An untried approach: query expansion.
- The corpus has 30 independent blocks and each question's answer is in its
  own block. The retrieval task is relatively easy compared with real-world
  scenarios.
