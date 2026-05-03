"""
main.py — FastAPI application: routes, validation, and async orchestration.

All CPU-bound work (embedding, retrieval, Claude call) is offloaded to a thread
pool via asyncio.to_thread() so the event loop stays unblocked between requests.

In-memory dicts hold session metadata and conversation history. This is
intentional for the assessment scope; a production system would use Redis.
"""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import get_answer, is_refusal
from embeddings import delete_session as _chroma_delete
from embeddings import embed_and_store
from pdf_processor import parse_and_chunk

# Load .env before any os.getenv calls below
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", "20"))
MAX_PDF_SIZE_BYTES: int = MAX_PDF_SIZE_MB * 1024 * 1024

ALLOWED_ORIGINS: list[str] = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

# ---------------------------------------------------------------------------
# In-memory session state
# (module-level so all route handlers share the same dicts within one process)
# ---------------------------------------------------------------------------

# session_id -> {"filename": str, "chunk_count": int, "created_at": str}
sessions: dict[str, dict] = {}

# session_id -> list[{"role": "user"|"assistant", "content": str}]
histories: dict[str, list[dict]] = {}

# ---------------------------------------------------------------------------
# Lifespan — startup banner only; singletons in embeddings/agent are lazy
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    api_key_status = "SET" if os.getenv("MISTRAL_API_KEY") else "*** NOT SET — /chat will fail ***"
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    print("=" * 62)
    print("  PDF-Constrained Conversational Agent")
    print("  ─────────────────────────────────────────────────────────")
    print(f"  MISTRAL_API_KEY : {api_key_status}")
    print(f"  MAX_PDF_SIZE_MB    : {MAX_PDF_SIZE_MB}")
    print(f"  CHROMA_PERSIST_DIR : {chroma_dir}")
    print(f"  ALLOWED_ORIGINS    : {ALLOWED_ORIGINS}")
    print("=" * 62)
    yield
    print("[main] Shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PDF-Constrained Conversational Agent",
    version="0.1.0",
    description="Upload a PDF, chat with it. Every answer is grounded and cited.",
    lifespan=lifespan,
)

# Wildcard CORS for assessment; tighten to ALLOWED_ORIGINS before production.
# Note: allow_credentials must be False when allow_origins=["*"] per the CORS spec.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: str
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {"message": "PDF Chat Agent API", "status": "running"}


@app.get("/health", summary="Health check")
async def health():
    """
    Lightweight liveness probe used by Render and the frontend.

    Returns the number of active in-memory sessions so it doubles as a
    quick diagnostic during development.
    """
    return {"status": "ok", "active_sessions": len(sessions)}


@app.post("/upload", status_code=status.HTTP_201_CREATED, summary="Upload a PDF")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Parse a PDF into chunks, embed them, and create a chat session.

    Validates file type (extension + magic bytes) and size before processing.
    Returns a session_id that must be included in all subsequent /chat calls.

    Raises:
        415: File is not a PDF.
        413: File exceeds MAX_PDF_SIZE_MB.
        400: File is empty.
        422: PDF is password-protected, image-only, or contains no text.
    """
    filename = file.filename or ""

    # --- Type validation: extension ---
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .pdf files are accepted.",
        )

    # --- Read bytes (needed for size check and magic-byte validation) ---
    pdf_bytes = await file.read()

    # --- Empty file guard ---
    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # --- Type validation: PDF magic bytes (%PDF-) ---
    # Guards against a renamed non-PDF bypassing the extension check.
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File does not appear to be a valid PDF (missing PDF header).",
        )

    # --- Size validation ---
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size ({len(pdf_bytes) / 1_048_576:.1f} MB) exceeds "
                f"the {MAX_PDF_SIZE_MB} MB limit."
            ),
        )

    # --- Parse and chunk — CPU-bound, run in thread pool ---
    print(f"[main] Parsing '{filename}' ({len(pdf_bytes) / 1_048_576:.2f} MB)…")
    try:
        chunks = await asyncio.to_thread(parse_and_chunk, pdf_bytes)
    except ValueError as exc:
        # Covers: password-protected, image-only, no extractable text
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # --- Assign session ID ---
    session_id = str(uuid.uuid4())

    # --- Embed and store — CPU-bound (model inference), run in thread pool ---
    await asyncio.to_thread(embed_and_store, chunks, session_id)

    # --- Persist session metadata ---
    sessions[session_id] = {
        "filename": filename,
        "chunk_count": len(chunks),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    histories[session_id] = []

    print(
        f"[main] Session created | id={session_id} "
        f"| file={filename} | chunks={len(chunks)}"
    )

    return {
        "session_id": session_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "message": "PDF uploaded and indexed successfully. Ready to chat.",
    }


@app.post("/chat", summary="Chat with the uploaded PDF")
async def chat(request: ChatRequest):
    """
    Answer a question grounded strictly in the uploaded PDF.

    Retrieves relevant chunks from ChromaDB, passes them to Claude with the
    anti-hallucination system prompt, and returns a cited answer.

    Conversation history for the session is maintained in-memory so the
    agent supports follow-up questions within the same session.

    Raises:
        404: session_id not found (PDF not uploaded or session expired).
        400: message is empty.
        502: Anthropic API call failed (surfaced as-is for debuggability).
    """
    # --- Session validation ---
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Session '{request.session_id}' not found. "
                "Please upload a PDF first."
            ),
        )

    # --- Message validation ---
    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )

    history = histories[request.session_id]

    print(
        f"[main] Chat request | session={request.session_id} "
        f"| history_turns={len(history)} | query={message!r:.80}"
    )

    # --- Get grounded answer — I/O + CPU-bound, run in thread pool ---
    # Pass a shallow copy of history so get_answer cannot mutate our list.
    try:
        result = await asyncio.to_thread(
            get_answer,
            message,
            request.session_id,
            list(history),
        )
    except Exception as exc:
        # Surface API errors clearly rather than returning a 500 with no context
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {exc}",
        ) from exc

    # --- Update conversation history ---
    # Store the raw user query (not the context-injected version) so history
    # stays concise; each new turn always retrieves fresh context anyway.
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result["answer"]})

    refusal = is_refusal(result["answer"])
    print(
        f"[main] Chat answered | session={request.session_id} "
        f"| is_refusal={refusal} | cited_pages={result['cited_pages']}"
    )

    return {
        "answer": result["answer"],
        "cited_pages": result["cited_pages"],
        "chunks_used": result["chunks_used"],
        "is_refusal": refusal,
        "session_id": request.session_id,
    }


@app.delete("/session/{session_id}", summary="Delete a session")
async def delete_session(session_id: str):
    """
    Remove a session's vector data from ChromaDB and clear its history.

    Safe to call after the user is done with a document. Frees ChromaDB
    memory and removes the conversation log.

    Raises:
        404: session_id not found.
    """
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    # Delete ChromaDB collection — I/O, run in thread pool
    await asyncio.to_thread(_chroma_delete, session_id)

    sessions.pop(session_id, None)
    histories.pop(session_id, None)

    print(f"[main] Session deleted | id={session_id}")

    return {"message": "Session deleted successfully.", "session_id": session_id}
