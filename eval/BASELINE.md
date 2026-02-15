# Evaluation Baseline

Baseline scores established on 2026-02-15 against the 27-chunk corpus (6 documents).

## Retrieval Evaluation (k=5)

| Metric | Score |
|--------|-------|
| Precision@5 | 0.76 |
| Recall@5 | 1.00 |
| MRR | 1.00 |

All 25 queries retrieved their expected source document in the top-5 results, with
the correct document always appearing at rank 1 (MRR = 1.0). Precision is 0.76
because some top-5 results include chunks from non-target documents, which is
expected behavior for a small corpus.

### Retrieval by Topic Area

| Topic | Precision@5 | Recall@5 | MRR |
|-------|-------------|----------|-----|
| client_relations | 1.00 | 1.00 | 1.00 |
| meeting_procedures | 0.64 | 1.00 | 1.00 |
| operations | 0.60 | 1.00 | 1.00 |
| pricing | 0.96 | 1.00 | 1.00 |
| trade_standards | 0.60 | 1.00 | 1.00 |

## Answer Quality Evaluation (LLM-as-Judge, 1-5 scale)

| Metric | Score |
|--------|-------|
| Accuracy | 4.72 |
| Groundedness | 4.64 |
| Faithfulness | 4.68 |
| Source Match Rate | 100% |
| Total Tokens | 25,808 |

### Answer Quality by Topic Area

| Topic | Accuracy | Groundedness | Faithfulness |
|-------|----------|--------------|--------------|
| client_relations | 4.60 | 4.60 | 4.60 |
| meeting_procedures | 5.00 | 4.40 | 5.00 |
| operations | 4.60 | 4.80 | 4.60 |
| pricing | 4.40 | 4.40 | 4.40 |
| trade_standards | 5.00 | 5.00 | 4.80 |

### Weak Queries

One query scored below 4 on accuracy:

- **pricing-5** (accuracy=3): "How do pricing benchmarks differ between a major metro
  area and a rural area for a designer with 5 years of experience?" — This is a
  cross-comparison question requiring the model to pull and compare figures from
  multiple case studies. Some figures were incorrect in the generated answer.

## Configuration

- Embedding model: `text-embedding-3-small`
- LLM model: `gpt-4o-mini`
- Judge model: `gpt-4o-mini`
- Reranking: disabled
- Context token budget: 4000
- Chunk size: 1000 / overlap: 200
