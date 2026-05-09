# Technical Note — PDF-Constrained Conversational Agent

## Overview

This system is a multi-turn RAG (Retrieval-Augmented Generation) chatbot that answers questions exclusively from an uploaded PDF. The user uploads a document, which is chunked and embedded into a per-session Qdrant vector collection. On every question, the most semantically relevant chunks are retrieved and injected into the prompt context — the LLM never sees the full document and has no access to its own training knowledge for answering. Every factual claim is required to cite a page number inline, and the system refuses to answer anything not found in the document.

The core design principle is strict grounding: the model is constrained to the retrieved context only, with citation requirements, explicit refusal instructions, and score-based relevance filtering working together to prevent hallucination. Conversational context is handled by rewriting follow-up queries into standalone retrieval queries before they reach the vector store, so multi-turn conversations maintain retrieval quality even when the user references prior answers.

## Architecture

```
User
 │
 ▼
React + Vite Frontend (Vercel)
 │   POST /upload  — PDF file
 │   POST /ask     — { query, session_id, history }
 │   DELETE /session — cleanup
 ▼
FastAPI Backend (Render)
 ├── main.py           — HTTP routes, session state, history accumulation
 ├── pdf_processor.py  — PDF text extraction + RecursiveCharacterTextSplitter
 ├── embeddings.py     — sentence-transformers + Qdrant vector store
 └── agent.py          — query rewriting, retrieval, Mistral API, citations
         │
         ├──► Qdrant Cloud (vector store, one collection per session UUID)
         └──► Mistral API  (mistral-small-latest, chat completions)
```

**Data flow (upload):**
PDF → PyMuPDF (per-page text) → RecursiveCharacterTextSplitter (per-page) → embed (all-MiniLM-L6-v2) → upsert to Qdrant collection

**Data flow (query):**
Query + history → contextual rewrite (if needed) → keyword enrichment → synonym expansion → Qdrant cosine search → score filter (≥ 0.20) → inject chunks into user message → Mistral API → citation extraction → structured JSON response

## Component Breakdown

### PDF Processing (`pdf_processor.py`)

Text is extracted page-by-page with PyMuPDF (`fitz`). Each page's text is split independently using LangChain's `RecursiveCharacterTextSplitter`:

- **chunk_size = 900 characters** (~130 words): sized to hold a full paragraph while leaving ample token budget for the system prompt, history, and multiple chunks in the context window.
- **chunk_overlap = 200 characters** (~22%): preserves cross-sentence continuity at chunk boundaries so a sentence starting in one chunk and completing in the next is not split without context.
- **Separator hierarchy**: `["\n\n", "\n", ". ", "! ", "? ", " ", ""]` — the splitter tries each separator in order, preferring paragraph breaks, then sentence ends, then word breaks, only splitting mid-word as a last resort.
- **Per-page splitting**: each page is split independently before chunks are combined. This guarantees chunks never cross page boundaries, so `[Page N]` citations remain accurate.

Each chunk is stored as `{"text": str, "page": int, "chunk_index": int}`.

### Embedding & Retrieval (`embeddings.py`)

**Model**: `sentence-transformers/all-MiniLM-L6-v2` — produces 384-dimensional float32 embeddings. Chosen for its small footprint (~90 MB), fast CPU inference, and strong semantic performance on English text. No GPU required on Render free tier.

**Vector store**: Qdrant with `Distance.COSINE`. One collection is created per `session_id` (UUID). This gives natural session isolation — no metadata filter is needed on search, and cleanup is a single `delete_collection` call.

**Connection**: reads `QDRANT_URL` and `QDRANT_API_KEY` at startup. If set, connects to Qdrant Cloud (persistent, remote). If not set, falls back to `QdrantClient(":memory:")` — same Qdrant semantics locally but ephemeral.

**Score filtering**: after Qdrant returns the top-10 candidates, any chunk with cosine similarity below `MIN_RETRIEVAL_SCORE` (default 0.20, env var configurable) is discarded before the LLM sees it. Thresholds:
- > 0.50 — clearly on-topic
- 0.20–0.50 — related / weakly associated (included)
- < 0.20 — vocabulary overlap only (filtered out)

If all candidates are filtered, an empty list is returned and the caller's fallback chain kicks in.

### Agent & Prompt Design (`agent.py`)

**Model**: `mistral-small-latest` via direct REST call to the Mistral chat completions endpoint. No SDK dependency.

**System prompt**: static 21-rule instruction set, sent as the first message on every turn. Key rules:
- **Rule 1** — answer only from `[DOCUMENT EXCERPTS]` block; never use training knowledge.
- **Rule 2** — every factual claim must carry an inline `[Page N]` citation.
- **Rule 3** — if the excerpts don't contain the answer, refuse with the exact sentence: `"I'm sorry, but the uploaded document does not contain information about [topic]."`
- **Rule 5** — if the document lists N items, the answer must list all N items (no selective summarization).
- **Rules 13, 15, 16** — extract and reason about code conditions, examples, and execution flow explicitly.

**Context injection**: retrieved chunks are embedded into the user message (not the system prompt), so each turn can retrieve different chunks and the model sees only the most relevant content for that specific question.

**Multi-turn history**: `main.py` maintains a `histories` dict keyed by `session_id`. Each completed `(user, assistant)` pair is appended and passed to `get_answer()` on the next call. The full history is forwarded to the Mistral API, giving the model dialogue continuity.

### Conversational Query Rewriting

Follow-up queries like "Explain the second point more simply" or "What does that mean?" cannot be sent directly to the vector store — they lack standalone semantic content. The pipeline resolves them in three steps before retrieval:

1. **`_is_contextual_query()`** — detects pronoun words (`it`, `this`, `that`), ordinal words (`second`, `last`, `previous`), follow-up phrases (`"tell me more"`, `"what about"`), or queries with fewer than 4 words.

2. **`_rewrite_for_retrieval()`** — if the query is contextual, calls Mistral (`max_tokens=60`, `temperature=0`) with the last 4 history turns and asks it to produce a standalone search query. For example: `"Explain the second point more simply"` → `"Open/Closed Principle Adapter Pattern"`. Falls back to the original query if the API key is missing or the call fails.

3. **`_extract_keywords()`** + enrichment — regardless of rewriting, the last assistant response is mined for distinctive content words (≥4 chars, non-stopword), and the top 10 are appended to the retrieval query. This boosts recall for follow-up questions that reuse the same topic area without explicit keywords.

**Retrieval drift prevention**: the rewritten/enriched query is used *only* for Qdrant search. The original user query always goes to `_build_user_message()` and the LLM prompt.

**Synonym expansion** (`_expand_query()`): adds domain synonyms for matched keywords — e.g. `"pitfalls"` expands to include `"problem issue drawback limitation disadvantage risk"`. Capped at 2 distinct expansion sets to avoid diluting the embedding.

**Fallback chain** (4 levels):
1. Expanded retrieval query (rewritten + enriched + synonyms)
2. Retrieval query without synonym expansion
3. Original query with synonyms (guards against enrichment drift)
4. Raw original query — if still empty, issue an immediate refusal without calling the LLM

### Anti-Hallucination Measures

Hallucination is blocked at multiple independent layers:

| Layer | Mechanism |
|-------|-----------|
| **Retrieval constraint** | LLM only sees the top-10 retrieved chunks, not the full document or its own knowledge |
| **Score filtering** | Chunks with cosine similarity < 0.20 are discarded before they reach the LLM |
| **System prompt (Rule 1)** | Explicit instruction to never use training knowledge; every claim must be traceable to an excerpt |
| **Citation requirement (Rule 2)** | Inline `[Page N]` after every factual claim anchors the answer to specific document locations |
| **Refusal instruction (Rule 3)** | Exact refusal wording enforced; no hedging or partial confabulation |
| **Full extraction rule (Rule 5)** | Prevents selective summarization that omits inconvenient or redundant-seeming items |
| **Reasoning rules (13, 15–18)** | Force extraction from code, examples, and nested conditions — common sources of missed information |

## Retrieval Quality

The score filter threshold (default 0.20) can be tuned via the `MIN_RETRIEVAL_SCORE` environment variable without restarting the server — the value is read on every retrieval call. Lower values increase recall (more chunks pass) at the cost of weaker relevance; higher values tighten precision but risk filtering out genuinely useful partial matches.

The 4-level fallback chain ensures that retrieval never silently fails: if the richest query (rewritten + enriched + expanded) finds nothing, each subsequent fallback simplifies the query until the raw original is tried. Only if all four levels return empty is a refusal issued.

Logging at each stage (`[agent]`, `[embeddings]`) records the final retrieval query, score filter counts, top chunk score, and Mistral token usage, enabling post-hoc debugging of any missed retrieval.

## Limitations & Future Work

- **No cross-session persistence**: session state (`sessions`, `histories` dicts in `main.py`) is in-memory. A Render dyno restart loses all active sessions. Future work: persist sessions to Redis or a lightweight DB.
- **In-memory Qdrant fallback**: without `QDRANT_URL`, vectors are held in the process — evicted on restart and capped by dyno RAM. Suitable for development only.
- **Image-only PDFs**: PyMuPDF extracts text layer only. Scanned PDFs without OCR return empty text and the system refuses all queries. Future work: integrate an OCR step (e.g. pytesseract).
- **Single PDF per session**: uploading a second PDF replaces the first (the existing Qdrant collection is dropped). Future work: multi-document sessions with per-document metadata filtering.
- **Context window cap**: very long PDFs produce many chunks, but only the top-10 are sent to the LLM. Deep content buried after page 50 may be consistently outscored by earlier pages. Future work: maximal marginal relevance (MMR) re-ranking.
- **Language**: `all-MiniLM-L6-v2` is English-optimised. Non-English PDFs will have degraded retrieval quality.
