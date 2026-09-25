# Benchmark and Optimization Report

The RAG question answering system: every measurement from the baseline
configuration to the final one, the reasoning behind each decision, and the
results.

---

## 1. Summary

| Metric | Baseline | Final | Change |
|---|---|---|---|
| Faithfulness | 0.7377 | **0.8785** | **+0.141** |
| Answer Relevancy | 0.6555 | **0.7659** | **+0.110** |
| Context Precision | 0.8048 | 0.7866 | -0.018 |
| Context Recall | 0.6426 | 0.6598 | +0.017 |

**17 measurements, 16 configurations.** Every measurement finished with zero
NaN scores (no failed jobs).

| Setting | Baseline | Final |
|---|---|---|
| Chunk size / overlap | 500 / 50 | 800 / 80 |
| Text cleaning | No | **Yes** |
| Retrieval | Dense | Dense |
| retrieve_k -> top_k | 5 -> 5 | **20 -> 5** |
| Reranker | No | **bge-reranker-v2-m3** |
| Hybrid search | No | No (measured, rejected) |
| Prompt variant | baseline | **cited** |
| Temperature | 0.0 | 0.0 |

---

## 2. Measurement methodology

### 2.1 Evaluation

The Ragas library, using an LLM as the judge. Judge model:
`openai/gpt-oss-120b` (separate from the generation model).

| Metric | What it measures | Category |
|---|---|---|
| Faithfulness | Whether the answer sticks to the context (hallucination check) | Generation |
| Answer Relevancy | How well the answer addresses the question | Generation |
| Context Precision | How accurate the retrieved chunks are | Retrieval |
| Context Recall | How much of the needed information was retrieved | Retrieval |

### 2.2 Retrieval F1

Context Precision and Context Recall trade off directly. To summarize both in
one number, their harmonic mean is used:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The arithmetic mean was not used because it does not penalize lopsided
configurations. For Precision = 0.90 and Recall = 0.30, the arithmetic mean is
0.60 while the harmonic mean is 0.45.

### 2.3 Measurement noise

The same pipeline output was evaluated twice (same data, same settings,
`temperature=0`):

| Metric | Run 1 | Run 2 | Difference |
|---|---|---|---|
| Faithfulness | 0.7349 | 0.7405 | +0.0056 |
| Answer Relevancy | 0.6595 | 0.6515 | -0.0080 |
| Context Precision | 0.8056 | 0.8040 | -0.0016 |
| Context Recall | 0.6486 | 0.6365 | -0.0121 |

**Noise level: ±0.012.**

**Reading rule:** differences above 0.03 are treated as meaningful; the
0.01-0.02 range is treated as noise.

### 2.4 Validity check

Every measurement reports the number of NaN scores per metric. A NaN is a
metric that could not be computed; it is left out of the mean, which biases
the score in an unknown direction.

The first attempts produced NaNs because of rate limits and too few output
tokens; three faulty runs on the same data gave Faithfulness values as
different as 0.61 / 0.64 / 0.67. After the settings were fixed, every
measurement came out clean.

---

## 3. All experiments

In chronological order. Bold values are the best in each column.

| # | Experiment | chunk | k | Rerank | Prompt | Cleaning | Faith. | Ans.Rel. | Ctx.Prec. | Ctx.Rec. | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline | 500 | 5 | - | base | - | 0.7377 | 0.6555 | 0.8048 | 0.6426 | 0.7137 |
| 2 | k3 | 500 | 3 | - | base | - | 0.7079 | 0.6550 | 0.8444 | 0.5479 | 0.6647 |
| 3 | k10 | 500 | 10 | - | base | - | 0.7245 | 0.6812 | 0.7406 | 0.7360 | 0.7383 |
| 4 | k20 | 500 | 20 | - | base | - | 0.7475 | 0.6291 | 0.7184 | 0.7535 | 0.7353 |
| 5 | chunk200 | 200 | 10 | - | base | - | 0.6656 | 0.6335 | 0.6717 | 0.4881 | 0.5653 |
| 6 | chunk800 | 800 | 10 | - | base | - | 0.7232 | 0.6290 | 0.8240 | 0.7227 | 0.7701 |
| 7 | chunk800_k5 | 800 | 5 | - | base | - | 0.7600 | 0.6977 | 0.8528 | 0.7073 | 0.7733 |
| 8 | chunk1000_k4 | 1000 | 4 | - | base | - | 0.6661 | 0.6182 | **0.8741** | 0.7307 | **0.7960** |
| 9 | hybrid | 800 | 5 | - | base | - | 0.7188 | 0.6517 | 0.8001 | 0.6911 | 0.7416 |
| 10 | rerank20_5 | 800 | 20>5 | + | base | - | 0.7850 | 0.7219 | 0.8081 | 0.6858 | 0.7419 |
| 11 | rerank10_5 | 800 | 10>5 | + | base | - | 0.7848 | 0.7220 | 0.7924 | 0.6682 | 0.7250 |
| 12 | prompt_strict | 800 | 20>5 | + | strict | - | 0.5986 | 0.4720 | 0.8174 | 0.6487 | 0.7233 |
| 13 | prompt_cited | 800 | 20>5 | + | cited | - | 0.8244 | 0.7025 | 0.8173 | 0.7025 | 0.7556 |
| 14 | **clean_corpus** | 800 | 20>5 | + | cited | + | **0.8785** | 0.7659 | 0.7866 | 0.6598 | 0.7176 |
| 15 | clean_top3 | 800 | 20>3 | + | cited | + | 0.8492 | **0.7698** | 0.8222 | 0.6095 | 0.7000 |
| 16 | clean_thresh01 | 800 | 20>5* | + | cited | + | 0.8602 | 0.7681 | 0.8046 | 0.5582 | 0.6591 |

\* Score threshold 0.1; the average number of contexts dropped to 2.2.

The values for experiment 1 are the mean of two independent measurements.

---

## 4. Experiment series

### 4.1 Sweeping k

Fixed: chunk = 500/50, no reranker, baseline prompt.

| Metric | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| Context Precision | **0.8444** | 0.8048 | 0.7406 | 0.7184 |
| Context Recall | 0.5479 | 0.6426 | 0.7360 | **0.7535** |
| Faithfulness | 0.7079 | 0.7377 | 0.7245 | **0.7475** |
| Answer Relevancy | 0.6550 | 0.6555 | **0.6812** | 0.6291 |
| Retrieval F1 | 0.6647 | 0.7137 | **0.7383** | 0.7353 |
| Evaluation time | 39 min | 47 min | 68 min | 111 min |

**Findings**

- The precision-recall trade-off behaved like the textbook: as k grew,
  Precision fell monotonically (0.8444 -> 0.7184) and Recall rose
  monotonically (0.5479 -> 0.7535).
- At k=3 Recall collapsed (0.5479) and pulled Faithfulness down with it (the
  lowest in the series). When the context lacks information, the model tries
  to fill the gaps.
- Answer Relevancy peaked at k=10 and dropped clearly at k=20 (-0.052). The
  model losing focus in a long context ("lost in the middle") is a consistent
  explanation.

**Decision: k = 10.** Retrieval F1 peaks here. Going to k=20 gains +0.018
Recall (close to the noise band) and loses -0.022 Precision, which does not
pay for the extra cost. The k=20 evaluation also took 1.6 times as long.

**Methodological note.** Not every integer in between was tried. Differences
between neighbouring values of k were expected to stay below the noise band
(±0.012); a coarse sweep gave the same information in a third of the time.

### 4.2 Sweeping chunk size

Fixed: k=10 (k=4 for chunk1000, k=5 for chunk800_k5). Overlap is 10% of
chunk_size in every experiment.

| Metric | 200/k10 | 500/k10 | 800/k10 | 800/k5 | 1000/k4 |
|---|---|---|---|---|---|
| Context Precision | 0.6717 | 0.7406 | 0.8240 | 0.8528 | **0.8741** |
| Context Recall | 0.4881 | **0.7360** | 0.7227 | 0.7073 | 0.7307 |
| Faithfulness | 0.6656 | 0.7245 | 0.7232 | **0.7600** | 0.6661 |
| Answer Relevancy | 0.6335 | 0.6812 | 0.6290 | **0.6977** | 0.6182 |
| Retrieval F1 | 0.5653 | 0.7383 | 0.7701 | 0.7733 | **0.7960** |

**Findings**

**chunk=200 got worse on all four metrics.** The expectation was that small
chunks would at least raise Precision; they did not. 200 characters is below
the threshold of semantic completeness: a chunk cannot hold a whole idea and
starts and ends mid-sentence. The judge model finds the fragment meaningless
on its own and counts it as irrelevant.

**At chunk=800 Precision jumped by +0.083.** 800 characters is large enough to
hold a typical forum answer from the dataset in one piece.

**Total context length hypothesis.** With chunk=800/k=10, Answer Relevancy
fell (-0.052), by the same amount as in the k=20 experiment. What the two have
in common is the total context length: 800 × 10 ≈ 8000 characters. The
hypothesis was tested by halving k.

**Hypothesis confirmed (800/k5).** With the total context cut from 8000 to
4000 characters:

| Metric | 800/k10 | 800/k5 | Difference |
|---|---|---|---|
| Answer Relevancy | 0.6290 | 0.6977 | +0.069 |
| Faithfulness | 0.7232 | 0.7600 | +0.037 |
| Context Precision | 0.8240 | 0.8528 | +0.029 |

Recall moved by -0.015, within the noise band.

**chunk=1000/k=4: retrieval improved, generation degraded.** The total context
is again about 4000 characters, but in larger pieces. Retrieval F1 reached the
best value of the series (0.7960), while Faithfulness fell by 0.094 and
Answer Relevancy by 0.080. 1000-character chunks help retrieval (a piece holds
a whole answer) but hurt generation (each chunk carries several ideas and
irrelevant parts).

**Decision: chunk=800 / overlap=80 / k=5.** Best on three of the four
metrics. 1000/k4 leads by 0.023 in retrieval F1 but trails by 0.08-0.09 on the
generation metrics.

### 4.3 Hybrid search (BM25 + dense)

Fixed: chunk=800/80, k=5. Results merged with Reciprocal Rank Fusion.

| Metric | Dense | Hybrid | Difference |
|---|---|---|---|
| Context Precision | **0.8528** | 0.8001 | -0.053 |
| Context Recall | **0.7073** | 0.6911 | -0.016 |
| Faithfulness | **0.7600** | 0.7188 | -0.041 |
| Answer Relevancy | **0.6977** | 0.6517 | -0.046 |
| Retrieval F1 | **0.7733** | 0.7416 | -0.032 |

**Decision: REJECTED.** All four metrics fell.

**Explanation.** BM25 was expected to cover the areas where embeddings are
weak (code, abbreviations, proper names). The corpus is made of forum answers,
and common financial words ("account", "bank", "money", "business") appear
everywhere. BM25 promotes the chunks where these words cluster, but because
they are common they do not discriminate.

The second factor is that RRF weights both lists equally: dense search alone
already works well, and mixing in a weaker second signal at equal weight
spoils the ranking.

### 4.4 Reranker

The model choice was checked first. Query: "How do I deposit a third party
cheque?"

| Document | Qwen3-Reranker-8B | bge-reranker-v2-m3 |
|---|---|---|
| "The weather in Ankara is cold in winter." | **0.903** (1st) | 0.000017 (3rd) |
| "Just have the associate sign the back..." | 0.880 (2nd) | 0.023 (2nd) |
| "A third party cheque requires endorsement..." | 0.562 (3rd) | **0.807** (1st) |

Qwen3-Reranker ranked the off-topic sentence as the most relevant and the
direct answer as the least relevant. Its scores are so close together that it
is not separating anything. `bge-reranker-v2-m3` was chosen.

**Results** (chunk=800/80):

| Metric | No reranker | 20 -> 5 | 10 -> 5 |
|---|---|---|---|
| Faithfulness | 0.7600 | **0.7850** | 0.7848 |
| Answer Relevancy | 0.6977 | 0.7219 | **0.7220** |
| Context Precision | **0.8528** | 0.8081 | 0.7924 |
| Context Recall | **0.7073** | 0.6858 | 0.6682 |
| Retrieval F1 | **0.7733** | 0.7419 | 0.7250 |
| Mean of four metrics | 0.7545 | 0.7502 | 0.7419 |

**Findings**

- The expectation was not met: the reranker was meant to protect Precision by
  filtering a wide candidate pool, yet both retrieval metrics fell.
- Generation improved clearly instead (both about +0.025, above the noise
  band).
- The size of the candidate pool (20 vs 10) did not affect the generation
  metrics (a difference of about 0.0002); the reranker picks largely the same
  chunks from both pools. The retrieval metrics are better with the wider
  pool.

**Decision: USED, retrieve_k=20 -> top_k=5.**

Reasoning: Faithfulness and Answer Relevancy measure the output the end user
actually sees; Context Precision and Recall are intermediate metrics.
Faithfulness works directly as a hallucination check, and in finance a
made-up answer is costly.

Because the plain mean of the four metrics is almost the same, this decision
is debatable; the scores of both configurations are shown above side by side.

### 4.5 Prompt engineering

Fixed: chunk=800/80, retrieve_k=20 -> top_k=5, reranker on.

| Metric | baseline | strict | cited |
|---|---|---|---|
| Faithfulness | 0.7850 | 0.5986 | **0.8244** |
| Answer Relevancy | **0.7219** | 0.4720 | 0.7025 |
| Context Precision | 0.8081 | **0.8174** | 0.8173 |
| Context Recall | 0.6858 | 0.6487 | **0.7025** |
| Retrieval F1 | 0.7419 | 0.7233 | 0.7556 |
| Mean of four metrics | 0.7502 | 0.6342 | **0.7617** |

**The strict variant: the target metric moved the wrong way.**

The prompt gained rules like "do not infer, do not generalize, do not combine
facts". The aim was to raise Faithfulness; it fell by 0.186.

Mechanism: the number of questions answered with "not enough information"
rose from 6 to 12 (40%). Even when the context contained the right answer,
the model refused to take the smallest step needed to connect it to the
question. It also started producing meta-claims such as "the context does not
confirm this"; since these are not factual claims that can be derived from the
context, the judge counted them as unsupported.

**The cited variant: citing sources.**

Instead of adding restrictions, the model was asked to end each claim with the
block it came from (`[1]`, `[2]`). Faithfulness reached the highest value in
the project so far (0.8244, +0.039).

Mechanism: having to cite forces the model back to the context throughout
generation, which makes it harder to say something the context does not
support.

**Decision: cited.**

**General lesson.** The two variants went in opposite directions. Telling the
model *what it may not do* (strict) made it dysfunctional; showing it *how to
do the task* (cited) raised the target metric.

### 4.6 Corpus cleaning

The dataset's context blocks are several forum answers joined together. A scan
of the corpus found:

| Defect | Count |
|---|---|
| Doubled quotes (`""`) | 104 |
| Letter directly after a quote | 87 |
| Repeated spaces | 260 |
| Sentence end without a space (`.X`) | 25 |

Example: `...Before you convert to S-Corp.You don't need to notify the IRS...`

These defects disable the `". "` separator of the recursive chunker; the
algorithm falls back a level and chunk boundaries land mid-sentence.

Four rules were applied: collapse doubled quotes, add a space after
sentence-ending punctuation, add a space after a quote, collapse repeated
spaces.

Result: 138 -> 137 chunks; the text of 85% of the chunks changed.

| Metric | Not cleaned | Cleaned | Difference |
|---|---|---|---|
| Faithfulness | 0.8244 | **0.8785** | +0.054 |
| Answer Relevancy | 0.7025 | **0.7659** | +0.063 |
| Context Precision | **0.8173** | 0.7866 | -0.031 |
| Context Recall | **0.7025** | 0.6598 | -0.043 |
| Mean of four metrics | 0.7617 | **0.7727** | +0.011 |

**Decision: APPLIED.**

The generation metrics reached their highest values in the project. Chunk
boundaries no longer fall mid-sentence, so the model reads whole sentences.
The drop in the retrieval metrics comes from the shifted chunk boundaries
changing how chunks line up with the ground truth.

**This was the largest gain from any single change.** No parameter setting
added +0.054 to Faithfulness.

### 4.7 Additional attempts that were rejected

**top_k = 3.** Hypothesis: since the reranker already picks the best
candidates, fewer chunks could reduce noise and raise Faithfulness.

| Metric | top_k=5 | top_k=3 |
|---|---|---|
| Faithfulness | **0.8785** | 0.8492 |
| Context Recall | **0.6598** | 0.6095 |
| Mean of four metrics | **0.7727** | 0.7627 |

Not confirmed. Faithfulness fell instead (-0.029); the -0.050 drop in Recall
explains why. Three chunks are not enough. This repeats the pattern seen in
the k=3 experiment.

**Reranker score threshold.** Score distribution: median 0.0268, min 0.0003,
max 0.9987. Hypothesis: set the number of contexts dynamically with a score
threshold instead of a fixed top_k.

With a threshold of 0.1, the average number of contexts fell from 5 to 2.2.

| Metric | No threshold | Threshold 0.1 |
|---|---|---|
| Faithfulness | **0.8785** | 0.8602 |
| Context Recall | **0.6598** | 0.5582 |
| Mean of four metrics | **0.7727** | 0.7478 |

Not confirmed. Context Recall lost heavily (-0.102).

Lesson: the absolute values of reranker scores are not a reliable measure of
relevance. A low-scoring chunk can still carry part of the answer even though
it ranks lower. Relative ranking (top_k) works better than an absolute
threshold.

---

## 5. Vector database comparison

A performance comparison outside the Ragas measurements. ChromaDB was set up
with the same 138 chunks and the same vectors, and 30 queries were run with
both methods.

| Method | Mean query time | Second run |
|---|---|---|
| NumPy (exact, brute force) | 0.203 ms | 0.282 ms |
| ChromaDB (HNSW) | 4.374 ms | 5.167 ms |

- NumPy is 18-21 times faster
- Result overlap: **100%** (both methods return the same chunks)
- ChromaDB indexing time: 1.07 seconds

**Decision: NumPy exact search.**

ANN indexes are an optimization for searching among millions of vectors. With
138 vectors the index itself is overhead; ChromaDB's slowness comes not from
the HNSW algorithm but from the cost of its layers (SQLite persistence,
serialization, API calls).

Exact search also returns exact results, while ANN is approximate. At this
scale there is no reason to accept approximation.

---

## 6. Progression

The contribution of each step from baseline to final:

| Step | Change | Faithfulness | Ans. Relevancy |
|---|---|---|---|
| 0 | Baseline (500/50, k=5) | 0.7377 | 0.6555 |
| 1 | chunk 500 -> 800, k 5 (fixed) | 0.7600 (+0.022) | 0.6977 (+0.042) |
| 2 | Reranker added (20 -> 5) | 0.7850 (+0.025) | 0.7219 (+0.024) |
| 3 | Prompt: baseline -> cited | 0.8244 (+0.039) | 0.7025 (-0.019) |
| 4 | Corpus cleaning | **0.8785 (+0.054)** | **0.7659 (+0.063)** |

Total: Faithfulness +0.141, Answer Relevancy +0.110.

Every step contributed more than the noise band (±0.012). The largest single
contribution came from corpus cleaning.

---

## 7. Conclusions

**1. Retrieval metrics and generation quality do not always move together.**

This split was seen in three separate experiments:

| Experiment | Retrieval | Generation |
|---|---|---|
| chunk=1000/k=4 | Up (F1 0.7960) | Collapsed (-0.09) |
| Reranker | Down (F1 0.7419) | Up (+0.025) |
| Corpus cleaning | Down (F1 0.7176) | Up (+0.06) |

Optimizing on retrieval scores alone, or on F1 alone, would have been
misleading.

**2. Total context length matters more than chunk size or k on their own.**

The best results came when chunk size × k stayed around 4000 characters. At
8000 characters the "lost in the middle" effect appeared; at 2400 characters
there was not enough information.

**3. Giving the prompt structure works better than giving it restrictions.**

A list of prohibitions (strict) made the model dysfunctional and cut the
target metric by 0.186. Citing sources (cited) raised the same metric by
0.039.

**4. Data quality gave a larger gain than parameter tuning.**

Corpus cleaning alone added +0.054 Faithfulness. No hyperparameter change was
as effective.

**5. Negative results are results too.**

Three of the six approaches tried were rejected: hybrid search, top_k=3 and
the reranker score threshold. These attempts give a measured reason for why
the unused approaches are not used.

---

## 8. Limitations

- **The dataset has 30 examples.** Statistical power is low; the score of a
  single question can shift the overall mean by 3%.
- **A single judge model.** Nothing was measured with a judge other than
  `openai/gpt-oss-120b`; absolute scores could change with a different judge.
  Relative comparisons are not expected to be affected.
- **Measurement noise of ±0.012.** Differences below this band were not
  interpreted.
- **Context Recall stayed around 0.66.** An untried approach is query
  expansion.
- **The retrieval task is relatively easy.** The corpus has 30 independent
  blocks and each question's answer is in its own block. Real-world document
  collections are much larger and messier.

---

## 9. Final configuration

```
chunk_size        = 800
chunk_overlap     = 80
text cleaning     = on
retrieval         = dense (NumPy exact search)
retrieve_k        = 20
top_k             = 5
reranker          = bge-reranker-v2-m3
hybrid search     = off
prompt variant    = cited
temperature       = 0.0
max_tokens        = 512
```

| Metric | Value |
|---|---|
| Faithfulness | 0.8785 |
| Answer Relevancy | 0.7659 |
| Context Precision | 0.7866 |
| Context Recall | 0.6598 |
| Retrieval F1 | 0.7176 |
