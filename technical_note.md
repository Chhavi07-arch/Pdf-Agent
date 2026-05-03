# Technical Note — PDF-Constrained Conversational Agent

## Overview

<!-- TODO: 1-2 paragraph summary of what the system does and why each design decision was made -->

## Architecture

<!-- TODO: Insert architecture diagram (Mermaid or image) showing:
     User → Frontend (React/Vercel) → FastAPI (Render) → Claude API
                                                       → ChromaDB (local)
                                    ← structured JSON response
-->

## Component Breakdown

### PDF Processing (`pdf_processor.py`)
<!-- TODO: Explain chunking strategy — why 500 tokens, why 50-token overlap -->

### Embedding & Retrieval (`embeddings.py`)
<!-- TODO: Explain choice of all-MiniLM-L6-v2, cosine similarity, top-k=5, ChromaDB isolation per session -->

### Agent & Prompt Design (`agent.py`)
<!-- TODO: Explain the system prompt strategy:
     - Strict grounding instruction
     - Inline citation format [Page N]
     - Refusal wording
     - Prompt caching on static system prompt
     - Multi-turn history management
-->

### Anti-Hallucination Measures
<!-- TODO: Enumerate every layer of defense:
     1. System prompt constraint
     2. Only retrieved chunks passed as context (no full doc)
     3. Refusal instruction with exact wording
     4. Temperature=0 (deterministic outputs)
     5. Citations force the model to stay grounded
-->

## Retrieval Quality

<!-- TODO: Describe how retrieval quality is measured / validated during testing -->

## Limitations & Future Work

<!-- TODO: e.g. image-only PDFs, very long documents exceeding context, session persistence across restarts -->
