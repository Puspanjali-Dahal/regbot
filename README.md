# RegBot , Retrieval-Augmented Generation over UK Financial Services Regulation

A retrieval system that answers compliance questions about UK payments regulation and cites the
specific provision the answer came from.

[Notebook: full code, evaluation and demo outputs](./regbot.ipynb) · [Technical report (PDF)](https://puspanjali-dahal.github.io/docs/regbot-report.pdf)

---

## The problem

A FinTech product owner asking "what are the exemptions from strong customer authentication?"
needs three things a general-purpose language model cannot reliably give:

1. **Precision.** Regulation numbers, monetary thresholds, and the exact conditions attached to
   exemptions are easily approximated or lost during model training.
2. **Currency.** Regulation is amended. A model with fixed weights silently goes out of date.
3. **Traceability.** For professional use, an answer has to be checkable against a named provision,
   not simply trusted.

RAG addresses all three by construction. A standalone LLM addresses none of them.

## The corpus

Three pieces of UK legislation, sourced directly from legislation.gov.uk:

| Document | Reference |
|---|---|
| Payment Services Regulations 2017 | SI 2017/752 |
| Electronic Money Regulations 2011 | SI 2011/99 |
| Financial Services and Markets Act 2000 | c.8 |

**1.3M characters → 527 chunks.** Paragraph-aware chunking, accumulating to ~1500 characters with
200-character overlap so a provision is not split mid-sentence. PDFs fetched with an MD5 manifest
to skip unnecessary re-downloads, text extracted with PyMuPDF and a block-level fallback for pages
where standard extraction returns too little.

## Architecture

Two pipelines share the same generation model and the same prompt. **Only retrieval differs**, so
any measured difference is attributable to retrieval alone.

**Baseline** , embed the query, return the 3 nearest chunks by cosine similarity.

**Enhanced** , three additions, each targeting a specific failure mode:

| Stage | What it does | Why |
|---|---|---|
| **HyDE** | LLM drafts a hypothetical regulatory passage; that gets embedded instead of the raw question | Formal regulatory language sits closer in embedding space to actual legislation than a short user question does |
| **Hybrid retrieval** | Blends HyDE vector search with BM25 keyword search on the original query, weighted 60/40 | Surfaces chunks sharing exact terminology even when embeddings are not the closest match |
| **Cross-encoder reranking** | 10 hybrid candidates scored jointly by `ms-marco-MiniLM-L-6-v2`, top 3 kept | Joint query-chunk scoring beats independent embeddings for final selection |

## Results

Ten queries across five categories: simple factual, deep context, ambiguous, product-owner
specific, and one edge case.

| Metric | Baseline | Enhanced | Change |
|---|---|---|---|
| Retrieval precision | 0.900 | **0.967** | +7.4% |
| Answer completeness | 0.642 | **0.700** | +9.0% |
| Answer correctness | 0.582 | **0.710** | **+22.0%** |

The gain is not uniform, and where it concentrates is the interesting part:

| Question type | Baseline | Enhanced | Change |
|---|---|---|---|
| Ambiguous | 0.498 | 0.758 | **+0.260** |
| Deep context | 0.596 | 0.741 | +0.145 |
| Edge case | 0.610 | 0.720 | +0.110 |
| Simple factual | 0.643 | 0.699 | +0.056 |
| Product owner | 0.565 | 0.620 | +0.055 |

**The clearest single case.** On an ambiguous query about consent requirements, the baseline
retrieved nothing relevant , precision 0.00 , and produced a generic answer from the wrong statute.
The enhanced pipeline reached precision 1.00 and cited section 67(2)(b) on the required form of
consent. That is precisely the gap HyDE and hybrid retrieval were chosen to close.

Diminishing returns on simple factual queries are expected: once direct embedding similarity
already retrieves the right chunk, there is nothing left for query expansion to fix.

## What didn't work

Four findings I would not have got from a system that only worked.

**HyDE can steer retrieval away from the answer.** On the insolvency edge case, the hypothetical
passage referenced the Insolvency Act 1986 , not in the corpus. Hybrid search followed it away
from the relevant safeguarding provisions and the enhanced pipeline returned "not in the context"
while the baseline answered. A hypothetical document that is slightly off-target is worse than no
hypothetical document at all.

**The recall figure of 0.016 is an evaluation artefact, not a result.** Retrieving 3 chunks from a
corpus of 527 makes corpus-wide recall structurally tiny regardless of retrieval quality. A
meaningful measure needs a hand-labelled relevance set per query. Reporting 0.016 as a weakness
would misrepresent the system.

**Grounding fell slightly, 0.375 → 0.359.** The enhanced pipeline synthesised across chunks rather
than reproducing one passage verbatim. For a compliance reader, an answer combining two provisions
is often more useful than an excerpt , which makes grounding the less informative metric here, not
a regression to fix.

**Keyword-based precision is a proxy.** Replacing it with human-annotated relevance would make both
precision and recall trustworthy. That is the first thing I would change.

## Limitations

- Output capped at 600 tokens to keep response times workable. Some answers end before a full
  explanation.
- Evaluation is 10 queries. Enough to show direction, not enough for confident effect sizes.
- Corpus is three statutes. Real compliance work spans FCA handbook material and guidance too.

## Stack

`Python` · `ChromaDB` · `sentence-transformers (all-MiniLM-L6-v2)` · `rank_bm25` ·
`cross-encoder (ms-marco-MiniLM-L-6-v2)` · `LLaMA 3.1 8B via Groq` · `PyMuPDF` · `Gradio`

## Running it

Open `regbot.ipynb` and run the cells in order. You will need a free
[Groq](https://console.groq.com) API key set in the config cell.

The first run downloads the three statutes from legislation.gov.uk, extracts and chunks the text,
and builds the ChromaDB collection , roughly 5 minutes. An MD5 manifest means subsequent runs skip
documents that have not changed. The final cell launches the Gradio interface with a toggle
between the baseline and enhanced pipelines.

## References

Gao, L., Ma, X., Lin, J. and Callan, J. (2022) *Precise Zero-Shot Dense Retrieval without
Relevance Labels*. [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)

Robertson, S. and Zaragoza, H. (2009) *The Probabilistic Relevance Framework: BM25 and Beyond*.
Foundations and Trends in Information Retrieval, 3(4), pp. 333,389.

Reimers, N. and Gurevych, I. (2019) *Sentence-BERT: Sentence Embeddings using Siamese
BERT-Networks*. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)

---

*Individual project, MSc Business Analytics, Warwick Business School. The accompanying written
report is not published here.*
