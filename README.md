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
 ├── POST /upload  →  pdf_processor.py  →  embeddings.py  →  in-memory numpy store
 ├── POST /chat    →  embeddings.py (retrieve)  →  agent.py  →  Mistral API
 ├── GET  /health  →  liveness probe (wakes Render on frontend load)
 └── DELETE /session/:id
```

The backend is fully stateless per-process: sessions, conversation history, and embeddings all live in Python dicts in RAM. This is intentional for the assessment scope — a production system would use Redis + a persistent vector database.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS |
| Backend | Python 3.11, FastAPI, Uvicorn |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector search | Exact cosine similarity via NumPy (no external vector DB) |
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
  → chunk_pages() splits into 400-char overlapping windows (80-char overlap)
  → sentence-transformers encodes chunks in batches of 8
  → L2-normalised embeddings stored in memory as numpy array
  → session_id returned to frontend
```

Chunking is character-based with overlap to preserve context across chunk boundaries. Each chunk never spans two pages so page citations are always accurate.

### 2. Chat

```
User question
  → _expand_query() adds synonyms (e.g. "cache" → "caching batch buffering")
  → cosine similarity search across session's embeddings → top-10 chunks
  → chunks injected into user message as [DOCUMENT EXCERPTS] block
  → 21-rule system prompt + conversation history sent to Mistral
  → response parsed for [Page N] citations
  → answer + cited_pages + is_refusal returned to frontend
```

### 3. Vector search

Uses exact numpy dot-product on L2-normalised vectors, which equals cosine similarity. For the collection sizes involved (tens to low hundreds of chunks), this is both faster and more accurate than approximate HNSW search, and saves ~150 MB of RAM compared to ChromaDB.

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

**3. Query expansion**
Before retrieval, the query is expanded with domain synonyms to reduce missed chunks from vocabulary mismatch (e.g. "convert" also retrieves "transform", "translate", "map").

**4. Fallback retrieval**
If the expanded query retrieves nothing, the raw query is tried as a fallback before refusing.

**5. Immediate refusal on zero retrieval**
If no chunks are retrieved at all, the agent refuses without making an API call.

---

## Technical note

### Why RAG instead of full-document stuffing

Sending the entire PDF to the LLM on every message would exceed context limits for any non-trivial document, cost significantly more per token, and give the model too much irrelevant content to sift through. RAG (Retrieve-Augment-Generate) retrieves only the chunks most relevant to the specific question — the model sees less noise, responses are faster, and grounding is tighter because the cited content is right there in the context window.

### Chunking strategy

Chunks are **400 characters with 80-character overlap**, split per page. Key decisions:

- **Character-based over token-based** — deterministic, no tokeniser dependency, easier to reason about sizes across different PDF text densities.
- **Per-page splitting** — a chunk never crosses a page boundary. This guarantees that the `page` metadata attached to every chunk is always accurate, which directly enables trustworthy `[Page N]` citations.
- **80-character overlap** — sentences at chunk edges are included in both adjacent chunks, so the retriever does not miss a relevant sentence just because it fell at a boundary.

### Embedding model

`all-MiniLM-L6-v2` was chosen because:
- 90 MB on disk — fits comfortably within Render's free-tier memory budget alongside FastAPI and the numpy store
- 384-dimensional output — small enough that exact similarity search over hundreds of chunks is faster than HNSW approximation
- Strong semantic quality for English text retrieval tasks despite its size

### NumPy vector store vs ChromaDB

The original design used ChromaDB. It was replaced with a plain Python dict + NumPy for two concrete reasons:

1. **Memory.** ChromaDB pulls in ONNX runtime, SQLite, and a C++ HNSW extension. Combined with PyTorch, this pushed the process over Render's 512 MB free-tier limit when a second PDF was uploaded. The NumPy store uses ~70 KB per session (46 chunks × 384 dims × float32) — effectively zero overhead.
2. **Accuracy.** HNSW is an approximate algorithm designed for millions of vectors. For collections of 50–500 chunks, exact cosine similarity is both more accurate (no approximation error) and faster.

Trade-off accepted: embeddings are in-memory only and lost on process restart. For the assessment scope this is acceptable — a production deployment would persist to a vector database (Pinecone, pgvector, Weaviate).

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

### Query expansion

Before retrieval, the query is augmented with a synonym set if a known keyword is detected (e.g. "cache" → adds "caching batch buffering"). This compensates for vocabulary mismatch between how the user phrases a question and how the PDF author wrote the answer. Expansion is additive — the original query is always preserved — and only one expansion fires per query (first match wins) to avoid diluting the embedding.

### Multi-turn conversation

Conversation history is maintained per session in a Python list. On each `/chat` call, the full history is prepended before the current user turn in the API request. Importantly, context chunks are re-retrieved fresh on every turn rather than cached — this means follow-up questions retrieve different chunks if they ask about a different part of the document, rather than being locked to the chunks from turn 1.

### Trade-offs summary

| Decision | Chosen | Alternative | Reason for choice |
|---|---|---|---|
| Vector store | NumPy in-memory | ChromaDB, Pinecone | Memory budget, exact accuracy for small N |
| Chunking unit | Character | Token | Deterministic, no tokeniser dependency |
| Retrieval k | 10 | 5, 20 | Balance between context coverage and prompt size |
| Session state | In-memory dict | Redis | Assessment scope — Redis adds ops overhead |
| LLM | Mistral small | GPT-4o, Claude | Cost-effective, strong instruction following |
| Prompt style | Static system prompt | Dynamic per-query | Simpler, consistent guardrails across all turns |

---

## Project structure

```
pdf-agent/
├── backend/
│   ├── main.py            # FastAPI app — routes, validation, session state
│   ├── agent.py           # Prompt construction, query expansion, Mistral call
│   ├── embeddings.py      # Encoding, numpy vector store, cosine retrieval
│   ├── pdf_processor.py   # PyMuPDF parsing, character-based chunking
│   ├── requirements.txt
│   ├── .python-version    # 3.11.9
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx                      # Session state, send/upload handlers
    │   └── components/
    │       ├── PDFUploader.jsx          # Drop zone, upload list, session switching
    │       ├── ChatWindow.jsx           # Message scroll, input bar, typing indicator
    │       └── MessageBubble.jsx        # User/assistant bubbles, citation chips
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
6. Add environment variable: `MISTRAL_API_KEY=<your key>`
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

**Error codes:** `413` file too large · `415` not a PDF · `422` image-only / no extractable text

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

**Error codes:** `404` session not found · `400` empty message · `502` Mistral API error

### `GET /health`
Liveness probe. Returns `{"status": "ok", "active_sessions": N}`.

### `DELETE /session/:id`
Remove a session and free its in-memory embeddings.

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
