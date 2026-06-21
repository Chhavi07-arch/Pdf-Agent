"""
evaluation.py — Standalone retrieval evaluation harness (Phase 2.5).

Embeds a PDF into an isolated session, runs a fixed set of test questions
through the retrieval pipeline, and reports per-question diagnostics plus a
summary (queries tested, average retrieval score, refusal rate, average latency).

Usage:
    python evaluation.py [PATH_TO_PDF] [--full]

    PATH_TO_PDF   PDF to evaluate. If omitted, falls back to the EVAL_PDF env
                  var, then the first *.pdf found in ./sample_pdfs or
                  ../sample_pdfs.
    --full        Also call the full agent pipeline (get_answer → Mistral API)
                  so the recorded refusal status reflects the LLM's actual
                  decision and latency includes generation. Requires
                  MISTRAL_API_KEY. Without it, evaluation is retrieval-only
                  (free, fast): "refused" means zero chunks retrieved.

Isolation:
    The in-memory Qdrant fallback has been removed, so evaluation runs against the
    configured Qdrant server (QDRANT_URL must be set). A unique eval-<uuid> session
    is created and deleted afterwards, so the script never touches your application
    collections even on a shared cluster.

This is an observability tool — it does not modify any retrieval or agent logic.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

from dotenv import load_dotenv

# Load .env (MISTRAL_API_KEY, QDRANT_*) before importing embeddings/agent.
load_dotenv()

from agent import _HIGH_CONFIDENCE_SCORE, get_answer, is_refusal  # noqa: E402
from embeddings import delete_session, embed_and_store, retrieve_relevant_chunks  # noqa: E402
from pdf_processor import parse_and_chunk  # noqa: E402

# Medium-confidence floor (mirrors agent.py's confidence bands; HIGH comes from
# the imported _HIGH_CONFIDENCE_SCORE so the two never drift).
_MEDIUM_CONFIDENCE_SCORE = 0.30

TEST_QUESTIONS = [
    "What is adapter pattern?",
    "What are drawbacks?",
    "What are disadvantages?",
    "What happens if provider returns null values?",
    "How are ratings normalized?",
]


def _confidence_band(score: float) -> str:
    if score >= _HIGH_CONFIDENCE_SCORE:
        return "HIGH"
    if score >= _MEDIUM_CONFIDENCE_SCORE:
        return "MEDIUM"
    return "LOW"


def _find_pdf(cli_path: str | None) -> str | None:
    """Resolve the PDF to evaluate from CLI arg → EVAL_PDF → sample_pdfs."""
    if cli_path:
        return cli_path
    env_path = os.getenv("EVAL_PDF")
    if env_path:
        return env_path
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.path.join(here, "sample_pdfs"), os.path.join(here, "..", "sample_pdfs")):
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                if name.lower().endswith(".pdf"):
                    return os.path.join(base, name)
    return None


def run_evaluation(pdf_path: str, full: bool) -> int:
    print("=" * 70)
    print("  RETRIEVAL EVALUATION")
    print(f"  PDF   : {pdf_path}")
    print(f"  Mode  : {'FULL (retrieval + LLM answer)' if full else 'RETRIEVAL-ONLY'}")
    print(f"  Qdrant: {os.getenv('QDRANT_URL', '').strip() or '(unset)'}")
    print("=" * 70)

    if not os.getenv("QDRANT_URL", "").strip():
        print("ERROR: QDRANT_URL must be set — the in-memory fallback has been removed.")
        return 1

    if full and not os.getenv("MISTRAL_API_KEY"):
        print("ERROR: --full requires MISTRAL_API_KEY. Aborting.")
        return 1

    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    session_id = f"eval-{uuid.uuid4()}"
    try:
        chunks = parse_and_chunk(pdf_bytes)
        embed_and_store(chunks, session_id)
        print(f"\nIndexed {len(chunks)} chunk(s).\n")

        results = []
        for q in TEST_QUESTIONS:
            t0 = time.perf_counter()
            retrieved = retrieve_relevant_chunks(q, session_id, top_k=7, original_query=q)
            latency = time.perf_counter() - t0

            top_score = retrieved[0]["score"] if retrieved else 0.0
            pages = sorted({c["page"] for c in retrieved})
            sections = []
            for c in retrieved:
                if c.get("section", "Unknown") not in sections:
                    sections.append(c.get("section", "Unknown"))
            refused = len(retrieved) == 0
            confidence = _confidence_band(top_score) if retrieved else "NONE"

            if full and retrieved:
                t1 = time.perf_counter()
                ans = get_answer(q, session_id, [])
                latency = time.perf_counter() - t1
                refused = is_refusal(ans["answer"])
                if ans["cited_pages"]:
                    pages = ans["cited_pages"]

            results.append({
                "query": q, "top_score": top_score, "pages": pages,
                "sections": sections, "confidence": confidence,
                "refused": refused, "latency": latency,
            })

            print(f"Q: {q}")
            print(f"   pages={pages}  sections={sections[:3]}")
            print(f"   top_score={top_score:.4f}  confidence={confidence}  "
                  f"refused={refused}  latency={latency:.2f}s")
            print()

        # ── Summary ──────────────────────────────────────────────────────────
        n = len(results)
        avg_score = sum(r["top_score"] for r in results) / n if n else 0.0
        refusal_rate = sum(1 for r in results if r["refused"]) / n if n else 0.0
        avg_latency = sum(r["latency"] for r in results) / n if n else 0.0

        print("=" * 70)
        print("  SUMMARY")
        print("-" * 70)
        print(f"  Queries tested        : {n}")
        print(f"  Average retrieval score: {avg_score:.4f}")
        print(f"  Refusal rate          : {refusal_rate:.0%}  "
              f"({sum(1 for r in results if r['refused'])}/{n})")
        print(f"  Average latency       : {avg_latency:.2f}s")
        print("=" * 70)
        return 0

    finally:
        delete_session(session_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval evaluation harness.")
    parser.add_argument("pdf", nargs="?", help="Path to the PDF to evaluate.")
    parser.add_argument("--full", action="store_true",
                        help="Also run the full agent pipeline (Mistral API) for true refusal status.")
    args = parser.parse_args()

    pdf_path = _find_pdf(args.pdf)
    if not pdf_path:
        print("ERROR: No PDF found. Pass a path, set EVAL_PDF, or add a PDF to sample_pdfs/.")
        print("Usage: python evaluation.py [PATH_TO_PDF] [--full]")
        return 1
    if not os.path.isfile(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path!r}.")
        return 1

    return run_evaluation(pdf_path, full=args.full)


if __name__ == "__main__":
    sys.exit(main())
