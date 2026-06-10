"""
main.py — FastAPI application: routes, validation, and async orchestration.

All CPU-bound work (embedding, retrieval, Mistral call) is offloaded to a
thread pool via asyncio.to_thread() so the event loop stays unblocked.

Upload pipeline (sequential, all timed):
  1. Validation  — fast, synchronous
  2. Parse + chunk — CPU-bound, thread pool
  3. Embed + Qdrant — CPU-bound + network, thread pool  ← heaviest stage
  4. Return 201   — response sent here; upload is complete from client's view
  5. Summary gen  — Mistral API call, runs as a BackgroundTask AFTER response

Moving summary generation to a background task is critical for Render free-tier
stability: the embed+Qdrant stage can take 15-30 s on a shared CPU, and adding a
synchronous Mistral call on top risks exceeding Render's 60 s proxy timeout.

In-memory dicts hold session metadata and conversation history. This is
intentional for assessment scope; a production system would use Redis.
"""

import asyncio
import json
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import get_answer, is_refusal, stream_answer
from embeddings import check_qdrant_connectivity, delete_session as _delete_session
from embeddings import embed_and_store, get_last_retrieval_debug, get_status, warmup
from pdf_processor import parse_and_chunk
from summarizer import format_summary_answer, generate_doc_summary, is_document_level_query

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
# ---------------------------------------------------------------------------

# session_id -> {"filename", "chunk_count", "created_at", "doc_summary"}
sessions: dict[str, dict] = {}

# session_id -> list[{"role": "user"|"assistant", "content": str}]
histories: dict[str, list[dict]] = {}

# ---------------------------------------------------------------------------
# Background task: summary generation after upload response is sent
# ---------------------------------------------------------------------------


async def _generate_summary_background(
    session_id: str,
    chunks: list,
    filename: str,
    api_key: str,
) -> None:
    """
    Generate the document summary and store it in the session metadata.

    Runs as a FastAPI BackgroundTask so it executes AFTER the /upload response
    has been sent to the client.  This keeps the upload response time short
    (no Mistral API call on the critical path) while still providing summary
    data for document-level queries that arrive shortly after.

    The first chat request that needs the summary may arrive before this task
    completes.  In that case, sessions[session_id]["doc_summary"] is still None
    and is_document_level_query() falls through to vector retrieval — degraded
    but correct.  The summary is available for all subsequent turns.
    """
    try:
        print(f"[main] BG summary: starting for session={session_id}")
        t0 = time.monotonic()
        doc_summary = await asyncio.to_thread(generate_doc_summary, chunks, filename, api_key)
        elapsed = time.monotonic() - t0

        if session_id not in sessions:
            print(f"[main] BG summary: session {session_id} was deleted before task completed — discarding.")
            return

        sessions[session_id]["doc_summary"] = doc_summary
        if doc_summary:
            print(
                f"[main] BG summary: stored in {elapsed:.1f}s | "
                f"type={doc_summary.get('document_type')!r} | session={session_id}"
            )
        else:
            print(f"[main] BG summary: generation returned None in {elapsed:.1f}s | session={session_id}")

    except Exception as exc:
        print(f"[main] BG summary: unhandled error for session={session_id} | {exc!r}")
        print(traceback.format_exc())


# ---------------------------------------------------------------------------
# Lifespan — warmup + startup banner
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """
    Application lifespan: warm up heavy singletons at startup so the first
    upload request does not bear the cold-start penalty.

    Without warmup:
      • First upload triggers model load (~30-60 s on Render free tier)
      • Combined with embedding + Qdrant this can exceed Render's 60 s timeout

    With warmup:
      • Model and Qdrant client are ready before any request arrives
      • First upload only pays for embed + Qdrant (~10-20 s typical)
    """
    api_key_status = "SET" if os.getenv("MISTRAL_API_KEY") else "*** NOT SET — /chat will fail ***"
    qdrant_url = os.getenv("QDRANT_URL", "").strip()

    print("=" * 62)
    print("  PDF-Constrained Conversational Agent")
    print("  ─────────────────────────────────────────────────────────")
    print(f"  MISTRAL_API_KEY : {api_key_status}")
    print(f"  QDRANT_URL      : {'SET → Cloud mode' if qdrant_url else 'not set → :memory: mode'}")
    print(f"  MAX_PDF_SIZE_MB : {MAX_PDF_SIZE_MB}")
    print(f"  ALLOWED_ORIGINS : {ALLOWED_ORIGINS}")
    print("=" * 62)

    # Pre-initialize the sentence-transformer model and Qdrant client.
    # This moves the cold-start cost from the first /upload request to startup,
    # where there is no user-facing timeout to worry about.
    t_warmup = time.monotonic()
    try:
        await asyncio.to_thread(warmup)
        print(f"[main] Warmup complete in {time.monotonic() - t_warmup:.1f}s — ready for requests.")
    except Exception as exc:
        print(f"[main] WARNING: Warmup failed in {time.monotonic() - t_warmup:.1f}s: {exc!r}")
        print("[main] Model/client will lazy-initialize on first request (higher first-upload latency).")

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


@app.get("/health", summary="Liveness probe")
async def health():
    """Lightweight liveness probe used by Render and the frontend."""
    return {"status": "ok", "active_sessions": len(sessions)}


@app.get("/health/debug", summary="Detailed system status")
async def health_debug():
    """
    Returns initialization state of all heavy singletons plus a live
    Qdrant connectivity probe.  Useful for diagnosing production issues
    without tailing Render logs.

    Expected healthy response:
      embedding_model: "loaded"
      qdrant_client:   "initialized"
      qdrant_ping:     true
      mistral_key:     "set"
    """
    emb_status = get_status()
    qdrant_ping: bool | str
    if emb_status["client_initialized"]:
        try:
            qdrant_ping = await asyncio.to_thread(check_qdrant_connectivity)
        except Exception as exc:
            qdrant_ping = f"error: {exc!r}"
    else:
        qdrant_ping = "client_not_initialized"

    return {
        "status":           "ok",
        "embedding_model":  "loaded" if emb_status["model_loaded"] else "not_loaded",
        "qdrant_client":    "initialized" if emb_status["client_initialized"] else "not_initialized",
        "qdrant_ping":      qdrant_ping,
        "mistral_key":      "set" if os.getenv("MISTRAL_API_KEY") else "MISSING",
        "active_sessions":  len(sessions),
    }


@app.post("/upload", status_code=status.HTTP_201_CREATED, summary="Upload a PDF")
async def upload_pdf(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Parse a PDF into chunks, embed them, store in Qdrant, and open a chat session.

    Validation → parse+chunk → embed+store → send 201 → [background: summary]

    Summary generation is queued as a BackgroundTask so the client receives
    the 201 response as soon as Qdrant indexing completes, without waiting
    for the Mistral summary API call.

    Raises:
        415: Not a PDF (extension or magic bytes).
        413: File exceeds MAX_PDF_SIZE_MB.
        400: File is empty.
        422: PDF is password-protected, image-only, or has no extractable text.
        500: Unexpected failure in parse or embed stage (full traceback logged).
    """
    t_upload_start = time.monotonic()
    filename = file.filename or "unknown.pdf"

    print(f"[main] Upload received: '{filename}'")

    # ── Validation ────────────────────────────────────────────────────────────

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .pdf files are accepted.",
        )

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File does not appear to be a valid PDF (missing %PDF- header).",
        )

    size_mb = len(pdf_bytes) / 1_048_576
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({size_mb:.1f} MB) exceeds the {MAX_PDF_SIZE_MB} MB limit.",
        )

    print(f"[main] Validation passed | size={size_mb:.2f} MB")

    # ── Stage 1: Parse + chunk ────────────────────────────────────────────────

    t_stage = time.monotonic()
    print(f"[main] Stage 1/3: Parsing and chunking '{filename}'…")
    try:
        chunks = await asyncio.to_thread(parse_and_chunk, pdf_bytes)
    except ValueError as exc:
        # Known bad-PDF conditions (password-protected, image-only, empty)
        print(f"[main] Stage 1 FAILED (bad PDF): {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        print(f"[main] Stage 1 FAILED (unexpected):\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF parsing failed: {exc}",
        ) from exc

    print(
        f"[main] Stage 1 complete: {len(chunks)} chunks | "
        f"{time.monotonic() - t_stage:.1f}s"
    )

    # ── Stage 2: Embed + Qdrant upsert ───────────────────────────────────────

    session_id = str(uuid.uuid4())
    t_stage = time.monotonic()
    print(f"[main] Stage 2/3: Embedding {len(chunks)} chunks → Qdrant | session={session_id}")
    try:
        await asyncio.to_thread(embed_and_store, chunks, session_id)
    except Exception as exc:
        print(f"[main] Stage 2 FAILED:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding or Qdrant storage failed: {exc}",
        ) from exc

    print(
        f"[main] Stage 2 complete: Qdrant indexed | "
        f"{time.monotonic() - t_stage:.1f}s"
    )

    # ── Persist session (summary populated by background task) ────────────────

    sessions[session_id] = {
        "filename":    filename,
        "chunk_count": len(chunks),
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "status":      "ready",  # embed is synchronous — session only exists once indexed
        "doc_summary": None,     # filled in by _generate_summary_background
    }
    histories[session_id] = []

    # ── Stage 3: Queue summary generation (runs AFTER response is sent) ───────

    api_key = os.getenv("MISTRAL_API_KEY", "")
    background_tasks.add_task(
        _generate_summary_background, session_id, chunks, filename, api_key
    )
    print(f"[main] Stage 3/3: Summary generation queued as background task.")

    total = time.monotonic() - t_upload_start
    print(
        f"[main] Upload complete | session={session_id} | file='{filename}' | "
        f"chunks={len(chunks)} | total_time={total:.1f}s"
    )

    return {
        "session_id":  session_id,
        "filename":    filename,
        "chunk_count": len(chunks),
        "message":     "PDF uploaded and indexed successfully. Ready to chat.",
    }


@app.post("/chat", summary="Chat with the uploaded PDF")
async def chat(request: ChatRequest):
    """
    Answer a question grounded strictly in the uploaded PDF.

    Document-level queries ("What is this about?", "Summarize this document")
    are answered from the pre-generated document summary if available, bypassing
    vector retrieval entirely.  All other queries go through the full RAG pipeline.

    Raises:
        404: session_id not found.
        400: empty message.
        502: Mistral API call failed.
    """
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{request.session_id}' not found. Please upload a PDF first.",
        )

    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )

    history = histories[request.session_id]
    session_meta = sessions[request.session_id]

    print(
        f"[main] Chat | session={request.session_id} "
        f"| turns={len(history)} | query={message!r:.80}"
    )

    # ── Document-level fast path ──────────────────────────────────────────────

    doc_summary = session_meta.get("doc_summary")

    if is_document_level_query(message) and doc_summary:
        print(f"[main] ROUTE: doc_summary (skipping vector retrieval)")
        answer = format_summary_answer(message, doc_summary, session_meta["filename"])
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant",  "content": answer})
        print(f"[main] Chat answered | source=doc_summary | is_refusal=False")
        return {
            "answer":      answer,
            "cited_pages": [],
            "chunks_used": 0,
            "is_refusal":  False,
            "session_id":  request.session_id,
        }

    if is_document_level_query(message) and not doc_summary:
        # Summary not yet ready (background task still running) or generation failed.
        print(f"[main] ROUTE: doc-level query but summary unavailable — vector retrieval fallback")

    # ── Normal vector retrieval path ──────────────────────────────────────────

    try:
        result = await asyncio.to_thread(
            get_answer,
            message,
            request.session_id,
            list(history),
        )
    except Exception as exc:
        print(f"[main] Chat pipeline error:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {exc}",
        ) from exc

    history.append({"role": "user",     "content": message})
    history.append({"role": "assistant", "content": result["answer"]})

    refusal = is_refusal(result["answer"])
    print(
        f"[main] Chat answered | source=vector_retrieval | "
        f"is_refusal={refusal} | cited_pages={result['cited_pages']}"
    )

    return {
        "answer":      result["answer"],
        "cited_pages": result["cited_pages"],
        "chunks_used": result["chunks_used"],
        "is_refusal":  refusal,
        "session_id":  request.session_id,
    }


@app.post("/chat/stream", summary="Chat with the uploaded PDF (streaming)")
async def chat_stream(request: ChatRequest):
    """
    Streaming variant of /chat — emits the answer token-by-token as NDJSON.

    Wire protocol (newline-delimited JSON, media type application/x-ndjson):
        {"type": "token", "text": "..."}                       ← zero or more
        {"type": "done",  "answer", "cited_pages",
                          "is_refusal", "chunks_used"}          ← exactly one
        {"type": "error", "text": "..."}                       ← on failure

    Document-level queries are answered from the pre-generated summary and
    streamed as a single token event (no Mistral call), mirroring /chat.

    Conversation history is persisted only once the stream finishes, using the
    full accumulated answer captured from the terminal "done" event.

    Raises (before streaming begins):
        404: session_id not found.
        400: empty message.
    """
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{request.session_id}' not found. Please upload a PDF first.",
        )

    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )

    history = histories[request.session_id]
    session_meta = sessions[request.session_id]
    doc_summary = session_meta.get("doc_summary")

    print(
        f"[main] Chat/stream | session={request.session_id} "
        f"| turns={len(history)} | query={message!r:.80}"
    )

    # ── Document-level fast path — stream the summary as a single token ───────
    if is_document_level_query(message) and doc_summary:
        print(f"[main] ROUTE: doc_summary (streaming, skipping vector retrieval)")
        answer = format_summary_answer(message, doc_summary, session_meta["filename"])
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": answer})

        def _summary_stream():
            yield json.dumps({"type": "token", "text": answer}, ensure_ascii=False) + "\n"
            yield json.dumps(
                {"type": "done", "answer": answer, "cited_pages": [],
                 "is_refusal": False, "chunks_used": 0},
                ensure_ascii=False,
            ) + "\n"

        return StreamingResponse(_summary_stream(), media_type="application/x-ndjson")

    if is_document_level_query(message) and not doc_summary:
        print(f"[main] ROUTE: doc-level query but summary unavailable — vector retrieval fallback")

    # ── Normal streaming retrieval path ──────────────────────────────────────
    # Snapshot history BEFORE the new turn so the agent sees prior context only.
    history_snapshot = list(history)

    def _event_stream():
        final_answer = None
        try:
            for line in stream_answer(message, request.session_id, history_snapshot):
                # Capture the full answer from the terminal "done" event so we
                # can persist it to conversation history once the stream ends.
                stripped = line.strip()
                if stripped:
                    try:
                        evt = json.loads(stripped)
                        if evt.get("type") == "done":
                            final_answer = evt.get("answer", "")
                    except json.JSONDecodeError:
                        pass
                yield line
        except Exception as exc:
            print(f"[main] Chat/stream pipeline error:\n{traceback.format_exc()}")
            yield json.dumps(
                {"type": "error", "text": f"LLM call failed: {exc}"},
                ensure_ascii=False,
            ) + "\n"
        finally:
            # Persist the turn only if we got a complete answer.
            if final_answer is not None:
                history.append({"role": "user",      "content": message})
                history.append({"role": "assistant", "content": final_answer})
                print(
                    f"[main] Chat/stream persisted | source=vector_retrieval | "
                    f"is_refusal={is_refusal(final_answer)}"
                )

    return StreamingResponse(_event_stream(), media_type="application/x-ndjson")


@app.get("/debug/retrieval/{session_id}", summary="Latest retrieval diagnostics")
async def debug_retrieval(session_id: str):
    """
    Return the most recent retrieval diagnostics captured for a session
    (query, rewritten query, candidate counts, top scores, and per-chunk
    semantic/bm25/fused/rerank scores). Observability only.

    Raises:
        404: session_id not found.
    """
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    diagnostics = get_last_retrieval_debug(session_id)
    return {
        "session_id":  session_id,
        "diagnostics": diagnostics,
        "message":     None if diagnostics else "No retrieval has run for this session yet.",
    }


@app.get("/session/{session_id}", summary="Session metadata + document summary")
async def get_session(session_id: str):
    """
    Return session metadata including the pre-generated document summary.

    The summary is produced by a background task shortly after upload (see
    summarizer.generate_doc_summary), so immediately after /upload the "summary"
    field may still be null — the frontend polls until it is populated.

    This endpoint only EXPOSES the existing summary; it does not generate it.

    Raises:
        404: session_id not found.
    """
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    meta = sessions[session_id]
    doc = meta.get("doc_summary")

    return {
        "session_id":  session_id,
        "filename":    meta["filename"],
        "chunk_count": meta["chunk_count"],
        "status":      meta.get("status", "ready"),
        "summary":     doc.get("summary") if doc else None,
        "summary_meta": {
            "title":         doc.get("title"),
            "topics":        doc.get("topics"),
            "document_type": doc.get("document_type"),
        } if doc else None,
    }


@app.get("/session/{session_id}/status", summary="Session indexing status")
async def session_status(session_id: str):
    """
    Report whether a session's content is indexed and ready for chat.

    Embedding runs synchronously on the /upload critical path (kept that way
    for Render free-tier 60 s-timeout safety), so a session only appears in the
    sessions dict once indexing has completed — meaning this endpoint returns
    "ready" for any session it can find. The frontend polls it as a safety net
    before unlocking the chat input; it is the integration point that would
    report "indexing" if embedding were ever moved to a background task.

    Raises:
        404: session_id not found.
    """
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    meta = sessions[session_id]
    return {
        "status":      meta.get("status", "ready"),
        "chunk_count": meta.get("chunk_count", 0),
    }


@app.delete("/session/{session_id}", summary="Delete a session")
async def delete_session(session_id: str):
    """
    Remove a session's Qdrant collection and in-memory state.

    Raises:
        404: session_id not found.
    """
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    await asyncio.to_thread(_delete_session, session_id)
    sessions.pop(session_id, None)
    histories.pop(session_id, None)

    print(f"[main] Session deleted | id={session_id}")
    return {"message": "Session deleted successfully.", "session_id": session_id}
