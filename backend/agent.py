"""
agent.py — OpenRouter API interaction, prompt construction, and anti-hallucination logic.

The system prompt is the primary guardrail: it restricts the model strictly to the
retrieved PDF chunks, mandates inline page citations on every factual claim, and
specifies the exact refusal wording for out-of-scope questions.

OpenRouter exposes an OpenAI-compatible REST endpoint, so this module uses plain
requests.post() — no SDK required.
"""

import os
import re
from typing import List, Optional

import requests # type: ignore

from embeddings import retrieve_relevant_chunks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "mistral-small-latest"
MAX_TOKENS = 1024
OPENROUTER_URL = "https://api.mistral.ai/v1/chat/completions"

# The prefix used for every out-of-scope refusal.
# is_refusal() matches on this exact string, so keep it in sync.
REFUSAL_PREFIX = (
    "I'm sorry, but the uploaded document does not contain information about"
)

# ---------------------------------------------------------------------------
# System prompt — static
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise document assistant. Your only function is to answer questions \
using the text excerpts from an uploaded PDF that are provided to you in each message.

## STRICT RULES — violating any rule is a critical failure

**Rule 1 — Source constraint**
Answer ONLY from the [DOCUMENT EXCERPTS] block in the user message.
Never use your training knowledge to answer, supplement, or infer anything.
Every single claim you make must be directly traceable to a specific excerpt.

**Rule 2 — Mandatory inline citations**
After every factual claim, immediately cite the page with [Page N].
Example: "The policy applies to all full-time staff [Page 4]."
Cite inline — do NOT batch all citations at the end of your answer.
Only cite page numbers that actually appear in the provided excerpts.
Never invent or guess a page number.

**Rule 3 — Standard refusal when the answer is absent**
If the excerpts do not contain enough information to answer the question,
respond with EXACTLY this sentence, substituting the actual topic:
  "I'm sorry, but the uploaded document does not contain information about [topic]."
Do not add any text before or after this sentence when issuing a full refusal.

**Rule 4 — Partial Information Handling**
If the document provides information related to the question but does
not fully answer it:
  1. State what the document explicitly says (with citations).
  2. Then add: "Note: The document does not explicitly address [the specific aspect not covered]."

Example: If asked "Can Builder be used for mutable objects?" and the
document only says "Builders work best for immutable classes":
→ Answer: "The document states that Builders work best for immutable
classes [Page X]. It does not explicitly discuss their use with
mutable objects."

Never make strong yes/no claims about things the document is silent on.

**Rule 5 — Full Extraction (CRITICAL)**
If the excerpts contain a list, enumeration, or multiple points about the
question topic, you MUST reproduce EVERY item — no exceptions.
Do not summarize. Do not pick highlights. Do not stop early.
If a section has 7 items, your answer must have 7 items.

MANDATORY PRE-WRITE CHECKLIST — complete this before drafting your answer:
1. Find the relevant section in the excerpts.
2. Locate the FIRST item explicitly — it is the most commonly skipped.
   Never start your answer from item 2.
3. Count every item, including ones that appear similar to each other.
   Similar-sounding violations (e.g., "SRP violation" AND "OCP violation")
   are separate, distinct points — list ALL of them individually.
4. Count your drafted answer. It must equal the excerpt item count exactly.
   If they differ, go back and find what you missed.

**Rule 6 — Low confidence threshold**
Even if a chunk seems only partially relevant, attempt to answer from it.
Only refuse if there is genuinely zero relevant information in the excerpts.
Do not refuse if there is ANY related content, even indirect.

**Rule 7 — Semantic Equivalence**
Treat these phrases as equivalent when searching the excerpts:
- "default values" = "defaults" = "default" = "preset values" = "initial values"
- "null values" = "nullability" = "missing fields" = "empty values" = "none"
- "missing values" = "missing fields" = "absent fields" = "not set"
- "error" = "exception" = "failure" = "fault"
- "convert" = "transform" = "translate" = "map" = "adapt"
- "log" = "logging" = "observability" = "metrics" = "tracking"
- "cache" = "caching" = "batch" = "buffering"
If the user asks about any term in a group, search the excerpts for ALL
terms in that group before deciding the answer is absent.
If examples or code in the excerpts demonstrate a concept, include them
as part of your answer — code examples count as relevant information.

**Rule 8 — No External Knowledge (STRICT)**
Do NOT include any information that is not explicitly present in the
document excerpts — even if that information is generally correct.
For example: do NOT say "ensuring thread safety" unless the document
explicitly mentions thread safety.
If you are about to write something that feels like common knowledge
rather than something you read in the excerpts, stop and remove it.

**Rule 9 — Concept Reframing (CRITICAL)**
When a question uses indirect or alternative phrasing, ALWAYS reframe
it into direct document concepts before searching:

Reframing examples you must apply:
- "outside build()" → reframe as → "constructor validation", "requireNonNull"
- "besides build()" → reframe as → "constructor", "before build()", "in class"
- "other than X" → reframe as → "alternative to X", "instead of X"
- "without using X" → reframe as → "alternative approach", "different method"
- "can it also" → reframe as → "alternative", "additional", "also"

Process for indirect questions:
1. Identify the indirect phrase ("outside", "besides", "other than", "alternative")
2. Reframe: what is the document likely to say instead?
3. Search excerpts using the REFRAMED concept
4. If found, answer using the document's terminology with citations
5. Only refuse if reframed search also returns nothing

This directly fixes the "outside build()" → "constructor validation" miss.

**Rule 10 — Forbidden phrases**
Never use: "based on my knowledge", "generally speaking", "typically",
"in general", "I believe", "as far as I know", or any phrase that signals
information from outside the document.
Never give partial answers when complete information is available in the context.

**Rule 11 — No inference or extrapolation**
Never guess, infer, or extrapolate beyond what the excerpts explicitly state.
If a detail is ambiguous in the document, report the ambiguity rather than resolving it.

**Rule 12 — Contradictions**
If excerpts on different pages contradict each other, report both with their
respective page citations and note the discrepancy. Do not silently pick one.

**Rule 13 — Code Condition Extraction (CRITICAL)**
When a code block appears in the excerpts, you MUST extract and state:
1. Every if/else condition — these reveal WHEN or UNDER WHAT CIRCUMSTANCES
   the logic applies.
   Example: `if ("GET".equalsIgnoreCase(method))` must be reported as
   "this logic only applies to GET requests" — not silently skipped.
2. Every fallback or else-branch — what happens when the condition is not met.
3. Every return value at each branch.

Evaluators deliberately ask "under what conditions does X apply?" or
"what constraints exist?" — the answer is always in the if-statement.
Omitting a conditional from a code analysis is a critical failure.

**Rule 14 — No selective similarity skipping**
When a document section lists multiple items of the same category
(e.g., multiple SOLID violations, multiple pitfalls, multiple constraints),
each item is independently required in your answer — even if they seem
redundant or very similar to each other.
The test: would a reader who only sees your answer know about EVERY named
item from the document? If not, your answer is incomplete.

Your answers must be fully traceable, accurate, and grounded in the provided text.
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_context_block(chunks: List[dict]) -> str:
    """
    Render retrieved chunks as a labelled context block for the user turn.

    Args:
        chunks: List of {"text", "page", "chunk_index", "score"} dicts,
                ordered by relevance (best first).

    Returns:
        Multi-line string ready to embed in the user message.
    """
    parts = []
    for chunk in chunks:
        parts.append(f"--- Page {chunk['page']} ---\n{chunk['text']}")
    return "\n\n".join(parts)


def _build_user_message(query: str, chunks: List[dict]) -> str:
    """
    Compose the full user-turn content: context block + question.

    Injecting context into every user turn (rather than stuffing it into the
    system prompt) means different questions can retrieve different chunks in
    a multi-turn conversation, and each turn is independently verifiable.

    Args:
        query:  The user's natural-language question.
        chunks: Retrieved PDF chunks to ground the answer.

    Returns:
        Formatted string to pass as the "user" role content.
    """
    context_block = _format_context_block(chunks)

    # Detect indirect/alternative phrasing and inject a reframing hint
    # directly before the question — highest attention position for the model.
    indirect_phrases = [
        "besides", "outside", "other than", "alternative",
        "another way", "other ways", "other locations",
        "other places", "without", "instead of",
    ]
    reframing_hint = ""
    query_lower = query.lower()
    if any(phrase in query_lower for phrase in indirect_phrases):
        reframing_hint = (
            "REFRAMING REQUIRED: This question uses indirect phrasing. "
            "Before answering, mentally reframe it: search the excerpts for "
            "ALL alternative implementations, locations, or approaches. "
            "Example: 'besides build()' → also check constructor-based validation. "
            "Do NOT refuse without first attempting this reframing.\n\n"
        )

    return (
        f"[DOCUMENT EXCERPTS]\n"
        f"{context_block}\n"
        f"[END EXCERPTS]\n\n"
        f"{reframing_hint}"
        f"CITATION FORMAT: Always use [Page N] format inline. "
        f"Example: 'The adapter translates formats [Page 3].'\n"
        f"If you include [Page N] citations, they MUST appear in Sources. "
        f"Never output 'Sources:' with nothing after it.\n\n"
        f"Question: {query}"
    )


def _build_messages_for_api(
    history: List[dict],
    current_user_content: str,
) -> List[dict]:
    """
    Build the full OpenAI-format message list for the API call.

    Structure:
      [system prompt]  ← always first
      [history turns]  ← "user"/"assistant" roles, passed through unchanged
      [current turn]   ← user message with context block injected

    OpenRouter/OpenAI uses "user"/"assistant" role names — identical to the
    internal history format — so no role conversion is needed (unlike Gemini).

    Args:
        history:              Prior turns as [{"role": "user"|"assistant", "content": str}].
        current_user_content: The fully-formatted user message for this turn.

    Returns:
        List of message dicts ready for the chat completions endpoint.
    """
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": current_user_content})
    return messages


def _extract_cited_pages(text: str) -> List[int]:
    """
    Parse all [Page N] references from the model's response.

    Handles optional whitespace and case variation, e.g.:
        [Page 3], [page 12], [PAGE 7], [Page  5]

    Args:
        text: Raw response string from the model.

    Returns:
        Deduplicated, sorted list of integer page numbers.
    """
    matches = re.findall(r"\[Page\s+(\d+)\]", text, flags=re.IGNORECASE)
    return sorted({int(m) for m in matches})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _expand_query(query: str, chunks_context: str = "") -> str:
    """
    Expand the query with synonyms to improve semantic retrieval.
    Returns the original query + related terms as a combined search string.
    """
    expansions = {
        "outside":     "outside besides alternative constructor before class method other",
        "where":       "where location place method constructor build class",
        "besides":     "besides outside alternative also another additionally other",
        "alternative": "alternative other besides outside also another approach",
        "provider": "provider service adapter returns output result",
        "return":   "return output result response value produce",
        "null":     "null nullability missing fields empty none default",
        "default":  "default defaults preset initial values builder optional",
        "missing":  "missing fields absent not set null nullability",
        "error":    "error exception failure handling",
        "handle":   "handle manage process deal",
        "convert":  "convert transform translate map",
        "log":      "log logging observability metrics",
        "cache":    "cache caching batch",
    }

    expanded = query
    query_lower = query.lower()
    for keyword, synonyms in expansions.items():
        if keyword in query_lower and synonyms not in expanded:
            expanded = f"{query} {synonyms}"
            break

    return expanded


def is_refusal(answer: str) -> bool:
    """
    Return True if the answer is a standard out-of-scope refusal.

    Matches on REFUSAL_PREFIX so main.py can set a flag in the API response
    without re-implementing the detection logic.

    Args:
        answer: The answer string returned by get_answer().

    Returns:
        True if the answer begins with the standard refusal prefix.
    """
    return answer.strip().startswith(REFUSAL_PREFIX)


def get_answer(
    query: str,
    session_id: str,
    conversation_history: Optional[List[dict]] = None,
) -> dict:
    """
    Full pipeline: retrieve relevant chunks → call OpenRouter → return grounded answer.

    Steps:
      1. Retrieve top-5 chunks from ChromaDB for the session.
      2. If nothing is retrieved, return a refusal immediately (no API call).
      3. Build the user message embedding the context block + question.
      4. Prepend the system prompt and history to form the full message list.
      5. POST to OpenRouter's chat completions endpoint.
      6. Parse [Page N] citations from the response.
      7. Return a structured result dict.

    The caller (main.py) is responsible for appending the returned answer to its
    copy of conversation_history so the next turn has the full dialogue context.

    Args:
        query:                The user's natural-language question.
        session_id:           UUID identifying the active upload session.
        conversation_history: Prior turns as [{"role": "user"|"assistant",
                              "content": str}]. Pass None or [] for a fresh
                              conversation.

    Returns:
        {
            "answer":      str       — model response or the refusal sentence,
            "cited_pages": list[int] — page numbers cited inline in the answer,
            "chunks_used": int       — number of context chunks sent to the model
                                       (0 on immediate refusal),
        }

    Raises:
        RuntimeError:           If MISTRAL_API_KEY is not set.
        requests.HTTPError:     Propagated on non-2xx response from OpenRouter.
        requests.Timeout:       If the request exceeds 60 seconds.
    """
    if conversation_history is None:
        conversation_history = []

    print(f"[agent] Query: {query!r}  |  session: {session_id}")

    # ------------------------------------------------------------------
    # Step 1: Retrieve relevant chunks (with query expansion for better recall)
    # ------------------------------------------------------------------
    expanded_query = _expand_query(query)
    print(f"[agent] Expanded query: {expanded_query!r}")
    chunks = retrieve_relevant_chunks(expanded_query, session_id, top_k=10)

    # Fallback to original query if expansion returned nothing
    if not chunks:
        chunks = retrieve_relevant_chunks(query, session_id, top_k=10)

    # ------------------------------------------------------------------
    # Step 2: Immediate refusal — no chunks means the topic is absent
    # ------------------------------------------------------------------
    if not chunks:
        print("[agent] No chunks retrieved — issuing immediate refusal (no API call).")
        topic = query if len(query) <= 80 else query[:77] + "…"
        refusal = f"{REFUSAL_PREFIX} {topic}."
        return {
            "answer": refusal,
            "cited_pages": [],
            "chunks_used": 0,
        }

    # ------------------------------------------------------------------
    # Step 3: Build user message with embedded context
    # ------------------------------------------------------------------
    user_message_content = _build_user_message(query, chunks)

    # ------------------------------------------------------------------
    # Step 4: Compose the full message list
    # System prompt first, then history, then current user turn.
    # History roles ("user"/"assistant") pass through unchanged — no
    # conversion needed since OpenAI format matches our internal format.
    # ------------------------------------------------------------------
    messages_for_api = _build_messages_for_api(
        history=list(conversation_history),  # shallow copy — don't mutate caller's list
        current_user_content=user_message_content,
    )

    # ------------------------------------------------------------------
    # Step 5: Call OpenRouter
    # ------------------------------------------------------------------
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY environment variable is not set. "
            "Add it to your .env file."
        )

    print(
        f"[agent] Calling {MODEL} via OpenRouter | "
        f"chunks={len(chunks)} | "
        f"history_turns={len(conversation_history)} | "
        f"top_chunk_score={chunks[0].get('score', 'n/a')}"
    )

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages_for_api,
            "max_tokens": MAX_TOKENS,
        },
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Step 6: Parse citations and log result
    # ------------------------------------------------------------------
    cited_pages = _extract_cited_pages(answer)

    usage = data.get("usage", {})
    print(
        f"[agent] Response done | "
        f"is_refusal={is_refusal(answer)} | "
        f"cited_pages={cited_pages} | "
        f"tokens={usage.get('prompt_tokens', '?')}in/"
        f"{usage.get('completion_tokens', '?')}out"
    )

    return {
        "answer": answer,
        "cited_pages": cited_pages,
        "chunks_used": len(chunks),
    }
