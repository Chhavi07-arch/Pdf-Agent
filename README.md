# PDF-Constrained Conversational Agent

Chat with any PDF. Every answer is grounded strictly in the document — no hallucination, citations included.

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + Tailwind CSS (Vite) |
| Backend | Python 3.11 + FastAPI |
| LLM | Claude claude-sonnet-4-20250514 (Anthropic SDK) |
| PDF parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| Vector store | ChromaDB (local persistence) |
| Deploy: backend | Render |
| Deploy: frontend | Vercel |

## Local Development

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in ANTHROPIC_API_KEY
uvicorn main:app --reload
```

API runs at `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
# create frontend/.env.local with: VITE_API_BASE_URL=http://localhost:8000
npm run dev
```

UI runs at `http://localhost:5173`.

## API Reference

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| POST | `/upload` | `multipart/form-data` — `file` field | `{ session_id: string }` |
| POST | `/chat` | `{ session_id, message }` | `{ answer: string, citations: number[] }` |
| DELETE | `/session/{session_id}` | — | `{ status: "deleted" }` |

## Deployment

<!-- TODO: Add Render + Vercel deployment steps once the app is built -->
