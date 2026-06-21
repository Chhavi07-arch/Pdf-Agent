# PDF Chat Agent

A document-grounded conversational AI that lets you upload a PDF and ask questions about it. Every answer is cited with page numbers and strictly limited to what the document actually says — no hallucination, no guessing.

**Live demo:** [pdf-agent-frontend.vercel.app](https://pdf-agent-frontend.vercel.app)

---

## What it does

Upload a PDF → ask questions in natural language → get answers with inline `[Page N]` citations pulled directly from the document. If the answer isn't in the document, the agent refuses clearly rather than making something up.

- Multiple PDFs can be uploaded in the same session, each with isolated context
- Full multi-turn conversation — follow-up questions work correctly
- Structured refusals for out-of-scope questions
- Works on text-layer PDFs up to 20 MB

---

## Architecture

```
User
 │
 ▼
React + Vite (Vercel)
 │  VITE_API_BASE_URL
 ▼
FastAPI (Render)
 ├── POST /upload  →  pdf_processor.py  →  embeddings.py  →  Qdrant
 │                    └── [background task] summarizer.py  →  Mistral API (doc summary)
 ├── POST /chat    →  summarizer.py (doc-level fast path, if summary ready)
 │                    └── embeddings.py (retrieve)  →  agent.py  →  Mistral API
 ├── GET  /health        →  liveness probe (wakes Render on frontend load)
 ├── GET  /health/debug  →  model/Qdrant status + live connectivity probe
 └── DELETE /session/:id →  drops Qdrant collection for that session
```

Session metadata and conversation history live in Python dicts in RAM, behind repository interfaces (intentional for assessment scope; Redis + a persistent session store would be the production choice). Vector embeddings are stored in Qdrant. **A running Qdrant server is required** — set `QDRANT_URL` (and `QDRANT_API_KEY`) to a reachable cluster. There is no in-memory fallback: if Qdrant is unconfigured or unreachable, `/upload` and `/chat` return HTTP 503 `"Qdrant server not working"` rather than silently degrading to ephemeral storage.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS |
| Backend | Python 3.11, FastAPI, Uvicorn |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector database | Qdrant (Cloud or in-memory fallback) |
| LLM | Mistral `mistral-small-latest` via Mistral API |
| PDF parsing | PyMuPDF (fitz) |
| Frontend hosting | Vercel |
| Backend hosting | Render (free tier) |

---

## How the pipeline works

### 1. Upload

```
PDF bytes
  → PyMuPDF extracts text per page
  → RecursiveCharacterTextSplitter splits each page independently:
      separators: \n\n → \n → ". " → " " → char
      chunk_size=900 chars, chunk_overlap=200 chars
  → sentence-transformers encodes chunks in batches of 8
  → embeddings + metadata upserted into a per-session Qdrant collection
  → 201 returned to frontend  ← upload is complete from the client's view
  → [BackgroundTask] summarizer.py calls Mistral to generate a structured
      doc summary (title, topics, document type, entities) from ~6 representative
      chunks. Stored in session metadata for the document-level fast path.
```

Chunks end at natural linguistic boundaries (paragraph → sentence → word) rather than mid-character. Each page is split independently so a chunk never spans two pages, guaranteeing accurate `[Page N]` citations.

The summary generation runs **after** the 201 is sent (FastAPI `BackgroundTasks`) so it never adds latency to the upload response. On startup, the embedding model and Qdrant client are preloaded via a warmup call so the first upload does not bear the cold-start penalty.

### 2. Chat

```
User question + conversation history
  → Document-level fast path (if query matches global patterns AND doc summary is ready):
      is_document_level_query(): detect "summarize this", "what is this about?", etc.
      format_summary_answer(): compose response from stored summary — no RAG call
  ↓ (all other queries go through RAG below)
  → _normalize_retrieval_query(): clean up retrieval query before vector search
      strip meta-prefixes ("what is meant by", "define")
      replace "/" with space ("shape/semantics" → "shape semantics")
      strip quotation marks
  → _is_contextual_query(): detect pronouns / ordinals / follow-up phrases
      if contextual → _rewrite_for_retrieval(): lightweight Mistral call
          resolves "the second point" → "Open/Closed Principle Adapter Pattern"
  → _extract_keywords(): mine last assistant response for document vocabulary
      enrich retrieval query with topic keywords from prior answer
  → _expand_query(): add domain synonyms (e.g. "cache" → "caching batch buffering")
  → Qdrant cosine similarity search → top-10 candidate chunks
  → score filter: drop chunks below MIN_RETRIEVAL_SCORE (default 0.20)
  → 4-level fallback if empty:
      1. expanded retrieval query (rewritten + enriched + synonyms)
      2. retrieval query without synonym expansion
      3. original query + synonyms  (guards against enrichment drift)
      4. raw original query → if still empty: immediate refusal
  → original (non-rewritten) query injected with filtered chunks into user message
  → 21-rule system prompt + conversation history sent to Mistral
  → response parsed for [Page N] citations; hallucinated page numbers stripped
  → answer + cited_pages + is_refusal returned to frontend
```

**Retrieval drift prevention:** the rewritten/enriched query is used *only* for Qdrant vector search. The original user question always goes to the LLM prompt — the model answers what the user actually asked.

### 3. Vector search

Uses Qdrant cosine similarity search. Each upload creates a dedicated Qdrant collection named after the session UUID. `QDRANT_URL` must point at a reachable Qdrant Cloud cluster (or any Qdrant server); vectors are stored remotely and persist across Render restarts with zero local RAM overhead. There is **no in-memory fallback** — if `QDRANT_URL` is unset the app raises a configuration error, and if the server is unreachable it returns HTTP 503 `"Qdrant server not working"`.

---

## Anti-hallucination design

The system has multiple independent layers that each enforce grounding:

**1. Retrieval-only context**
The LLM never sees the full document — only the top-10 most relevant chunks. It physically cannot cite something it was not given.

**2. 21-rule system prompt**
Key rules enforced on every response:
- Answer only from the `[DOCUMENT EXCERPTS]` block — never from training knowledge
- Cite `[Page N]` inline after every factual claim
- If excerpts lack the answer, respond with the exact refusal sentence: *"I'm sorry, but the uploaded document does not contain information about [topic]."*
- Full extraction: if a section has 7 bullet points, the answer must have 7 bullet points, starting from item 1
- Code condition extraction: every `if/else` condition must be stated explicitly (e.g. "only applies to GET requests")
- No selective similarity skipping: similar-sounding items (e.g. SRP violation and OCP violation) are always listed separately
- Code flow completeness: condition → action → continuation, never just the action
- Outer condition priority: in nested logic, the outermost `if` is always stated first
- Section awareness: clearly labeled document sections (e.g. "Pitfalls", "Forces") must be searched before refusing

**3. Citation validation**
After the LLM responds, every `[Page N]` citation is cross-checked against the set of pages actually present in the retrieved chunks. Hallucinated page numbers (pages the model was never shown) are stripped from the returned `cited_pages` list before the response reaches the frontend.

**4. Retrieval query normalization**
Before any vector search, the query is normalized: meta-prefixes ("what is meant by", "define") are stripped, slashes are replaced with spaces (`"shape/semantics"` → `"shape semantics"`), and quotation marks are removed. This ensures unusual punctuation or phrasing doesn't degrade embedding similarity compared to how the PDF author wrote the same concept.

**5. Conversational query rewriting**
Follow-up queries ("Explain the second point more simply", "Why does that happen?") cannot be sent directly to the vector store — they lack standalone semantic content. The pipeline resolves them first:
- `_is_contextual_query()` detects ordinal words (`second`, `last`), pronouns (`it`, `this`, `that`), and unambiguous follow-up phrases (`"tell me more"`, `"why is that"`).
- `_rewrite_for_retrieval()` makes a lightweight Mistral call (60 tokens, temperature=0) to produce a standalone search query by resolving references using the last 4 conversation turns.
- `_extract_keywords()` mines the last assistant response for distinctive content words and appends them to the retrieval query, boosting recall on topic-continuation follow-ups.
- The rewritten query is **only** used for Qdrant search; the original question always reaches the LLM.

**6. Query expansion**
After conversational rewriting, the query is augmented with domain synonyms to reduce vocabulary mismatch (e.g. "pitfalls" also retrieves "drawback limitation disadvantage risk"). Capped at 2 expansion sets to avoid diluting the embedding.

**7. 4-level fallback retrieval**
If the richest retrieval query returns nothing, the pipeline retries: expanded → unexpanded → original + synonyms → raw original.

**8. Immediate refusal on zero retrieval**
If all four fallback levels return no chunks, the agent refuses without making an API call. The `is_refusal()` function also catches LLM-paraphrased refusals via a regex pattern, so even non-canonical refusal wording is correctly flagged in the API response.

---

## Technical note

### Why RAG instead of full-document stuffing

Sending the entire PDF to the LLM on every message would exceed context limits for any non-trivial document, cost significantly more per token, and give the model too much irrelevant content to sift through. RAG (Retrieve-Augment-Generate) retrieves only the chunks most relevant to the specific question — the model sees less noise, responses are faster, and grounding is tighter because the cited content is right there in the context window.

### Chunking strategy

Chunks use `RecursiveCharacterTextSplitter` with **900-character target size and 200-character overlap**, split per page. Key decisions:

- **`RecursiveCharacterTextSplitter` over manual sliding window** — walks a separator hierarchy (`\n\n` → `\n` → `. ` → ` ` → char) before falling back to raw character splits. In practice, nearly all chunks end at paragraph or sentence boundaries. The old 400-char window frequently cut mid-sentence, producing fragment embeddings that matched queries poorly.
- **900-char chunk size (~130 words)** — captures a full paragraph as one semantic unit. At 400 chars (old), the embedding model had to represent an incomplete thought; at 900 chars it can represent a complete argument. 10 chunks × 900 chars ≈ 1,400 LLM tokens — well within Mistral's context window even with system prompt and history.
- **Per-page splitting** — a chunk never crosses a page boundary. This guarantees that the `page` metadata attached to every chunk is always accurate, which directly enables trustworthy `[Page N]` citations.
- **200-character overlap** — ensures sentences near chunk boundaries appear in both adjacent chunks. At the paragraph-split level the splitter naturally groups complete paragraphs, so overlap mainly benefits dense prose that splits at sentence boundaries.
- **Character-based length function** — deterministic, no tokeniser dependency, consistent behaviour across different PDF text densities.

**Old vs new comparison:**

| | Old (400-char window) | New (RCS 900-char) |
|---|---|---|
| Split point | Mid-character (arbitrary) | Paragraph → sentence → word |
| Typical chunk | Fragment of a paragraph | Complete paragraph |
| Overlap | 80 chars (may cut mid-word) | 200 chars at sentence boundaries |
| Embedding quality | Incomplete semantic unit | Complete semantic unit |
| Citations | Accurate (per-page) | Accurate (per-page, unchanged) |

### Embedding model

`all-MiniLM-L6-v2` was chosen because:
- 90 MB on disk — fits comfortably within Render's free-tier memory budget alongside FastAPI
- 384-dimensional output — a good balance between embedding quality and storage/compute cost
- Strong semantic quality for English text retrieval tasks despite its size

### Retrieval score filtering

After Qdrant returns the top-k candidates, chunks below the minimum cosine similarity threshold are dropped before they reach the LLM. This prevents low-quality, weakly-related context from entering the prompt and confusing the model.

**Qdrant scoring:** `Distance.COSINE` returns cosine similarity (dot product of L2-normalised vectors), range −1 to +1. For sentence-transformer embeddings on real text, practical scores fall in [0, 0.9].

| Score range | Meaning | Action |
|---|---|---|
| > 0.50 | Directly relevant | Included |
| 0.20–0.50 | Related / weak association | Included |
| < 0.20 | Vocabulary overlap only | **Filtered out** |

**Default threshold: 0.20** — conservative enough that genuinely relevant queries always pass, but removes the truly unrelated candidates that would only add noise. Configurable via `MIN_RETRIEVAL_SCORE` env var.

If filtering removes all candidates, the fallback path in `agent.py` retries with the raw (unexpanded) query. If that also returns nothing, the agent issues the standard refusal sentence rather than making an API call.

### Vector database: Qdrant

The system uses Qdrant as the vector database. A running Qdrant server is **required** — the connection is configured from environment variables:

- **`QDRANT_URL` + `QDRANT_API_KEY` set (Qdrant Cloud)** — vectors are stored remotely and persist across Render restarts. The Render process holds only the embedding model (~90 MB); per-session RAM overhead drops to zero.
- **`QDRANT_URL` unset** — there is no in-memory fallback. The vector store raises a configuration error and requests fail fast with HTTP 503 `"Qdrant server not working"`. This is deliberate: silently degrading to ephemeral, process-local memory hides a broken deployment and loses data on every restart.

Each PDF upload creates a dedicated Qdrant collection named after the session UUID. This gives natural isolation, makes deletions atomic (one `delete_collection` call), and avoids the need for per-query metadata filters. The collection is dropped automatically when the user removes the document.

**Why Qdrant over the previous NumPy store:** the NumPy approach required manual L2 normalisation + matrix dot-product on every query and held all vectors in process RAM. Qdrant moves storage and search to a purpose-built engine with proper HNSW indexing, typed payload storage, and a well-defined API. With Qdrant Cloud the RAM savings on Render are material: a session with 100 chunks × 384 dims × 4 bytes = 154 KB is now stored remotely rather than in the Python process.

**Why Qdrant over the previous ChromaDB attempt:** ChromaDB's Python package pulls in ONNX runtime, SQLite, and a C++ HNSW extension — enough overhead to push the Render free tier process over 512 MB with two concurrent sessions. `qdrant-client` is a lightweight REST/gRPC client with no heavy native extensions; the actual vector work happens in the Qdrant engine (remote or in-process via the lightweight embedded mode).

### System prompt design

The prompt has grown to 21 explicit rules through iterative testing against real PDFs. The progression addressed specific failure modes:

| Failure observed | Rule added |
|---|---|
| Missing bullet points | Rule 5 — pre-write checklist, count verification |
| Skipping first list item | Rule 5 — explicit "locate item 1 first" |
| Ignoring `if` conditions in code | Rule 13 — code condition extraction |
| Similar items collapsed into one | Rule 14 — no selective similarity skipping |
| Refusing on example-based answers | Rule 15 — example & demonstration reasoning |
| Partial code flow ("fetches response") | Rule 20 — full execution flow extraction |
| Missing outer `if` in nested logic | Rule 21 — outer condition priority |

The rules are injected as a static system prompt, which means the same guardrails apply to every turn of every conversation, regardless of what the user asks.

### Conversational query rewriting

Sending a follow-up like "Explain the second point more simply" directly to the vector store returns zero relevant chunks — the embedding has no semantic content without the prior context. The pipeline resolves this before retrieval:

1. `_is_contextual_query()` checks for ordinal words (`second`, `last`, `previous`), pronoun words (`it`, `this`, `that`, `they`), unambiguous follow-up phrases (`"tell me more"`, `"why is that"`, `"elaborate"`), or queries with 2 or fewer words. Broad phrases like `"what about"` are intentionally excluded — they are equally likely to introduce a new standalone topic.
2. If contextual, `_rewrite_for_retrieval()` calls Mistral with `max_tokens=60`, `temperature=0` and the last 4 history turns, producing a standalone search query. Example: `"Explain the second point more simply"` → `"Open/Closed Principle Adapter Pattern"`.
3. `_extract_keywords()` mines the last assistant response for content words (≥4 chars, non-stopword) and appends the top 10 to the retrieval query. This boosts recall even for non-contextual follow-ups that stay on the same topic.

The rewritten query is only used for Qdrant search. The original question always goes to the LLM prompt.

### Query expansion

After conversational rewriting, the query is augmented with domain synonyms if a known keyword is detected (e.g. `"pitfalls"` → adds `"problem issue drawback limitation disadvantage risk"`). This compensates for vocabulary mismatch between how the user phrases a question and how the PDF author wrote the answer. At most 2 expansion sets fire per query, and the synonym sets are deduplicated to avoid repetition when multiple related keywords appear in the same query.

### Multi-turn conversation

Conversation history is maintained per session in a Python list. On each `/chat` call, the full history is prepended before the current user turn in the API request. Context chunks are re-retrieved fresh on every turn — follow-up questions retrieve different chunks if they ask about a different part of the document, rather than being locked to the chunks from turn 1. The conversational rewriting layer ensures that follow-up queries which reference prior answers still retrieve the right chunks from the vector store.

### Document-level summary memory

At upload time, `summarizer.py` generates a lightweight structured summary of the document from ≤6 representative chunks (first 2 chunks for title/intro, heading-bearing chunks for structure, evenly-spaced fallback). The result — title, 2–4 sentence description, topics, document type, entities — is stored in the session metadata dict.

On every `/chat` call, `is_document_level_query()` checks whether the question targets the document as a whole (`"What is this document about?"`, `"Summarize the document"`, `"What topics are covered?"`). If yes and the summary is available, `format_summary_answer()` composes a direct response from the stored metadata — bypassing vector retrieval entirely.

**Key properties:**
- Summary generation runs as a FastAPI `BackgroundTask` — the 201 upload response is sent first, so upload latency is unaffected even if the Mistral summary call takes several seconds.
- If the background task has not completed when the first document-level query arrives, `is_document_level_query()` falls through to the normal RAG path — degraded but correct.
- Any failure in `generate_doc_summary()` returns `None` silently — the upload still succeeds.

### Trade-offs summary

| Decision | Chosen | Alternative | Reason for choice |
|---|---|---|---|
| Vector database | Qdrant (server required) | ChromaDB, Pinecone | Lightweight client, persistent cloud option, free tier |
| Collection strategy | One collection per session | Shared + filter | Simpler isolation, atomic deletion, no query filter overhead |
| Chunking | `RecursiveCharacterTextSplitter` 900/200 | Manual char window 400/80 | Paragraph/sentence boundaries → better embeddings |
| Retrieval k | 10 candidates → score filter | Fixed 5 or 10 | Oversample then discard noise rather than hard-cap |
| Score threshold | 0.20 (configurable) | No threshold | Drops vocabulary-overlap-only chunks; configurable per env |
| Conversational rewrite | Lightweight Mistral call (60 tokens) | Off / HyDE / local model | Reuses existing API key; no new dependency; 4-turn window bounded |
| Retrieval drift guard | Original query always goes to LLM | Rewritten query to LLM | Ensures the model answers what the user asked, not the search query |
| Query normalization | Pre-search text cleanup (no API call) | None | Free fix for slash/quote/prefix issues that degrade embedding quality |
| Doc-level fast path | Pre-generated summary (BackgroundTask) | Full RAG on every query | Instant answers for overview queries; no extra latency on upload |
| Fallback chain | 4 levels (expanded → unexpanded → original+syn → raw) | Single attempt | Never silently fails; each level reduces specificity gracefully |
| Session state | In-memory dict | Redis | Assessment scope — Redis adds ops overhead |
| LLM | Mistral small | GPT-4o, Claude | Cost-effective, strong instruction following |
| Prompt style | Static system prompt | Dynamic per-query | Simpler, consistent guardrails across all turns |

---

## Project structure

The backend follows SOLID layering: small interface contracts in `app/interfaces/`,
concrete adapters in `app/infrastructure/`, orchestration in `app/services/`, and a
single composition root (`app/container.py`) that wires the object graph. The
top-level modules (`main.py`, `agent.py`, `embeddings.py`, `pdf_processor.py`,
`summarizer.py`) are thin facades that delegate to the container.

```
pdf-agent/
├── backend/
│   ├── main.py                 # FastAPI routes + validation (delegates to repositories/services)
│   ├── agent.py                # Facade → ChatService
│   ├── embeddings.py           # Facade → HybridRetriever + vector store
│   ├── pdf_processor.py        # Facade → parser + chunker
│   ├── summarizer.py           # Facade → SummaryService
│   ├── app/
│   │   ├── config.py           # Settings (single env source) + constants
│   │   ├── errors.py           # ConfigError, QdrantUnavailableError
│   │   ├── container.py        # Composition root — wires the whole graph
│   │   ├── domain/models.py    # Chunk, RetrievedChunk, DocSummary, Session, ChatResult
│   │   ├── interfaces/         # ABCs: Embedder, VectorStore, KeywordIndex, Reranker,
│   │   │                       #       Retriever, LLMClient, DocumentParser, Chunker, repos
│   │   ├── infrastructure/     # QdrantVectorStore (no fallback), SentenceTransformerEmbedder,
│   │   │                       #   BM25KeywordIndex, CrossEncoderReranker, MistralClient,
│   │   │                       #   PyMuPDFParser, RecursiveChunker, in-memory repositories
│   │   └── services/           # HybridRetriever, ChatService, QueryBuilder, PromptBuilder,
│   │                           #   CitationService, SummaryService
│   ├── requirements.txt
│   ├── .python-version         # 3.11
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx                      # Session state, send/upload handlers
    │   └── components/
    │       ├── PDFUploader.jsx          # Drop zone, upload list, session switching, cycling stage text
    │       ├── ChatWindow.jsx           # Message scroll, input bar, typing indicator
    │       └── MessageBubble.jsx        # User/assistant bubbles, citation chips, refusal card
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## Local development

### Prerequisites

- Python 3.11
- Node.js 18+
- A Mistral API key from [console.mistral.ai](https://console.mistral.ai/api-keys)

### Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set MISTRAL_API_KEY

# Start the server
uvicorn main:app --reload --port 8000
```

Backend runs at `http://localhost:8000`.
Interactive API docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend

npm install

# Point the frontend at the local backend
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local

npm run dev
```

Frontend runs at `http://localhost:5173`.

---

## Environment variables

### Backend (`.env`)

| Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | — | **Required.** Mistral API key |
| `QDRANT_URL` | *(blank)* | **Required.** Qdrant cluster URL. If unset, `/upload` and `/chat` return HTTP 503 — there is no in-memory fallback. |
| `QDRANT_API_KEY` | *(blank)* | Qdrant Cloud API key. Required when `QDRANT_URL` points at a cloud cluster. |
| `MIN_RETRIEVAL_SCORE` | `0.20` | Minimum cosine similarity score for a chunk to reach the LLM. |
| `MAX_PDF_SIZE_MB` | `20` | Reject uploads larger than this |
| `ALLOWED_ORIGINS` | `http://localhost:5173,...` | CORS allowed origins |

### Frontend (`.env.local`)

| Variable | Description |
|---|---|
| `VITE_API_BASE_URL` | Backend base URL, e.g. `https://your-backend.onrender.com` |

---

## Deployment

### Backend → Render

1. Push the repo to GitHub.
2. Create a new **Web Service** on Render pointed at the repo.
3. Set **Root Directory** to `backend`.
4. Set **Build Command:** `pip install -r requirements.txt`
5. Set **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables:
   - `MISTRAL_API_KEY=<your Mistral key>`
   - `QDRANT_URL=<your Qdrant Cloud cluster URL>`
   - `QDRANT_API_KEY=<your Qdrant Cloud API key>`
7. Deploy.

The frontend pings `/health` on load to wake the free-tier instance before the user's first upload.

### Frontend → Vercel

1. Import the repo on Vercel.
2. Set **Root Directory** to `frontend`.
3. Add environment variable: `VITE_API_BASE_URL=https://<your-render-service>.onrender.com`
4. Deploy. Vercel auto-detects Vite — no build config needed.

---

## API reference

### `POST /upload`
Upload and index a PDF.

**Request:** `multipart/form-data` with field `file` (`.pdf`, max 20 MB)

**Response `201`:**
```json
{
  "session_id": "uuid",
  "filename": "document.pdf",
  "chunk_count": 46,
  "message": "PDF uploaded and indexed successfully. Ready to chat."
}
```

**Error codes:** `413` file too large · `415` not a PDF · `422` image-only / no extractable text · `503` Qdrant server not working

### `POST /chat`
Ask a question about an uploaded PDF.

**Request:**
```json
{ "session_id": "uuid", "message": "What is the adapter pattern?" }
```

**Response `200`:**
```json
{
  "answer": "The adapter pattern translates... [Page 3].",
  "cited_pages": [3, 8],
  "chunks_used": 10,
  "is_refusal": false,
  "session_id": "uuid"
}
```

**Error codes:** `404` session not found · `400` empty message · `502` Mistral API error · `503` Qdrant server not working

### `GET /health`
Liveness probe. Returns `{"status": "ok", "active_sessions": N}`.

### `GET /health/debug`
Detailed system status — returns initialization state of the embedding model and Qdrant client, plus a live Qdrant connectivity probe. Useful for diagnosing production issues without tailing Render logs.

```json
{
  "status": "ok",
  "embedding_model": "loaded",
  "qdrant_client": "initialized",
  "qdrant_ping": true,
  "mistral_key": "set",
  "active_sessions": 2
}
```

### `DELETE /session/:id`
Remove a session, drop its Qdrant collection, and free its in-memory state.

---

## Test instructions for evaluators

The deployed interface is live at **[pdf-agent-frontend.vercel.app](https://pdf-agent-frontend.vercel.app)** — no setup required. Use any text-layer PDF (design patterns, CS papers, technical specs work well).

> The Render free-tier backend sleeps after inactivity. The first upload after a cold start may take **30–60 seconds**. Subsequent requests are fast.

---

### Test 1 — Basic factual retrieval with citation

**Steps:**
1. Upload any technical PDF.
2. Ask a direct factual question about something on a specific page, e.g. *"What is the adapter pattern?"*

**Expected:**
- Answer contains `[Page N]` inline citations.
- The cited page numbers match what is actually on those pages in your PDF.
- No information is added that isn't in the document.

---

### Test 2 — Out-of-scope refusal

**Steps:**
1. Ask something that has no answer in your PDF, e.g. *"What is the capital of France?"* or *"Explain quantum computing."*

**Expected:**
- Response begins with exactly: *"I'm sorry, but the uploaded document does not contain information about…"*
- No hallucinated answer. No partial guess.
- `is_refusal: true` in the API response.

---

### Test 3 — Completeness (multi-point extraction)

**Steps:**
1. Use a PDF that contains a numbered or bulleted list (pitfalls, violations, steps, rules).
2. Ask: *"List all the pitfalls"* or *"What are all the [X] mentioned?"*

**Expected:**
- Every item from the list is present in the answer — including the first one.
- Nothing is summarised or merged.
- Item count in the answer matches item count in the document.

---

### Test 4 — Code condition extraction

**Steps:**
1. Use a PDF with a code example that has an `if` condition.
2. Ask: *"How does the caching logic work?"* or *"Explain the code."*

**Expected:**
- The answer explicitly states the condition (e.g. *"this only applies to GET requests"*).
- The full flow is described: condition → action → what happens next.
- Outer conditions in nested `if` blocks are stated first.

---

### Test 5 — Multi-turn follow-up

**Steps:**
1. Ask a question and receive an answer.
2. Ask a follow-up that refers to the previous answer, e.g. *"Can you expand on the second point?"* or *"Why does that happen?"*

**Expected:**
- The agent correctly understands what "that" or "the second point" refers to.
- The follow-up answer is still grounded in the document with citations.

---

### Test 6 — Multiple PDFs, session isolation

**Steps:**
1. Upload PDF A. Ask a question specific to PDF A.
2. Upload PDF B using **Add another PDF**. Switch to PDF B's session.
3. Ask the same question.

**Expected:**
- Answers are different and each only references content from its own PDF.
- Switching back to PDF A's session restores its conversation history.

---

### Test 7 — API directly (no UI)

The FastAPI interactive docs are available at:

```
https://pdf-agent-backend.onrender.com/docs
```

You can test all endpoints — `/upload`, `/chat`, `/health`, `/session/{id}` — directly from the browser without any client setup.

**Useful curl snippets:**

```bash
# Health check
curl https://pdf-agent-backend.onrender.com/health

# Upload a PDF
curl -X POST https://pdf-agent-backend.onrender.com/upload \
  -F "file=@your-document.pdf"

# Chat (use session_id from upload response)
curl -X POST https://pdf-agent-backend.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<uuid>", "message": "What is the main topic?"}'
```

---

### What to observe in Render logs

The backend emits structured logs on every request. Under **Logs** in the Render dashboard you can observe the full retrieval pipeline in real time:

```
[main] Parsing 'document.pdf' (0.24 MB)…
[pdf_processor] Extracted text from 9 page(s).
[pdf_processor] Created 46 chunk(s) from 9 page(s).
[embeddings] Embedding 46 chunk(s) for session 'abc-123'…
[embeddings] Stored 46 chunk(s) for session 'abc-123'.

[agent] Query: 'what is the adapter pattern?'
[agent] Expanded query: 'what is the adapter pattern? convert transform translate map'
[embeddings] Retrieving top-10 chunk(s) for session 'abc-123'…
[embeddings] Retrieved 10 chunk(s); top score=0.6241.
[agent] Calling mistral-small-latest | chunks=10 | history_turns=0 | top_chunk_score=0.6241
[agent] Response done | is_refusal=False | cited_pages=[3, 8] | tokens=2245in/139out
```

These logs confirm: which chunks were retrieved, what score they had, whether the answer was a refusal, and which pages were cited.

---

- **Text-layer PDFs only.** Scanned or image-only PDFs have no extractable text and are rejected with a `422`.
- **In-memory sessions.** Sessions are lost if the Render instance restarts (free tier spins down after inactivity). Re-upload the PDF to continue.
- **Single process.** No shared state between instances. Use Redis + a persistent vector store for multi-instance deployments.
- **Context window.** Only the top-10 most relevant chunks are sent to the LLM. Answers about topics spread thinly across many pages may miss detail from non-retrieved chunks.
- **Mistral rate limits.** Heavy usage on a hobby key will produce `429` errors, surfaced in the UI as "Something went wrong".
