# DesignRAG — Demo Talking Points

Reference notes for presenting DesignRAG. Not a script — just key points to hit.

## What Is This?

DesignRAG is the **knowledge layer** for an AI-native interior design studio. It makes institutional design expertise — pricing formulas, trade standards, meeting procedures, product recommendations — searchable and citable via natural language questions.

**The core idea:** A solo designer's expertise lives in scattered documents (PDFs, notes, templates). RAG makes that knowledge instantly accessible with cited answers, so nothing gets lost or forgotten.

## Architecture: Why RAG-Only?

The broader system separates two types of data:

| Type | Examples | Storage | Why |
|---|---|---|---|
| **Institutional knowledge** | Pricing formulas, trade standards, maintenance tips | ChromaDB (vector search) | Unstructured, benefits from semantic search |
| **Operational data** | Client records, project timelines, invoices | Postgres (relational) | Structured, needs exact queries and joins |

**DesignRAG handles only the first type.** If you ask "What is the Smith project timeline?" it correctly says "I don't have that information" — because client-specific data belongs in Postgres, not the knowledge base.

## The RAG Pipeline

```
Ingestion:  Document → Load → Classify (LLM) → Chunk → Embed → Store
Retrieval:  Question → Embed → Vector Search → [Rerank] → Trim → LLM → Cited Answer
```

### Key decisions worth discussing:

1. **LLM classifier at ingestion** — Every document is auto-tagged with a topic area and document type by gpt-4o-mini. This enables metadata filtering at query time ("only search pricing documents"). One LLM call per document, not per chunk.

2. **Chunking strategy** — 1,000 characters with 200-character overlap. Recursive splitting tries paragraph boundaries first, then sentences, then words. The overlap ensures concepts spanning a split point appear in both chunks.

3. **No hybrid search** — With a 27-chunk corpus, pure vector similarity already achieves 100% recall. Adding keyword search would add complexity for marginal benefit at this scale. The evaluation harness would catch it if this becomes a problem.

4. **Context window management** — tiktoken counts exact tokens locally before sending to the LLM. Greedy selection: add chunks by relevance until the budget (4,000 tokens) is hit. If a single chunk exceeds the budget, it gets truncated at a token boundary rather than skipped.

5. **LLM-as-judge evaluation** — 25 hand-labeled Q&A pairs scored by gpt-4o-mini on accuracy, groundedness, and faithfulness. Baseline: 4.72/5 accuracy, 100% source match rate.

## Numbers to Reference

- **Corpus:** 6 documents, 27 chunks, 5 topic areas
- **Retrieval:** Recall@5 = 1.00, MRR = 1.00 (correct doc always at rank #1)
- **Answer quality:** 4.72/5 accuracy, 4.68/5 faithfulness, 100% source match
- **Test suite:** 50 tests, no API key required to run
- **Latency:** ~1-2 seconds per query (embedding + vector search + LLM generation)

## Demo Flow

1. **Show the Streamlit UI** — ask "How do I calculate the FIE per square foot?" to demonstrate a clean, cited answer
2. **Show metadata filtering** — toggle the topic area filter to "pricing" and ask the same question — same answer, but retrieval is narrowed
3. **Show the scoping boundary** — ask "What is the Smith project timeline?" to demonstrate the system correctly saying it doesn't know
4. **Show the evaluation harness** — run `uv run python -m eval.evaluate_retrieval` to show automated quality measurement
5. **Show the Swagger UI** — open `/docs` to show the auto-generated API documentation
