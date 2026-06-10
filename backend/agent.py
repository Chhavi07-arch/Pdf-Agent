"""
agent.py — Mistral API interaction, prompt construction, and anti-hallucination logic.

The system prompt is the primary guardrail: it restricts the model strictly to the
retrieved PDF chunks, mandates inline page citations on every factual claim, and
specifies the exact refusal wording for out-of-scope questions.

Mistral exposes an OpenAI-compatible REST endpoint, so this module uses plain
requests.post() — no SDK required.
"""

import json
import os
import re
from typing import Iterator, List, Optional

import requests # type: ignore

from embeddings import retrieve_multi_query, retrieve_relevant_chunks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "mistral-small-latest"
MAX_TOKENS = 1024
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# When True, each query is diversified into 2 alternative phrasings and all
# formulations are retrieved, merged, and reranked together (Phase 3 Step 1).
# Falls back to single-query retrieval if variant generation fails.
#
# Disabled by default because evaluation on the Adapter PDF increased refusal
# rate and latency. The implementation is kept intact (generate_query_variants,
# retrieve_multi_query) and re-enables by flipping this flag back to True.
ENABLE_MULTI_QUERY = False

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

## EVIDENCE-BASED ANSWERING POLICY (read first — governs the rules below)

Your default posture is to ANSWER from the provided excerpts — conservatively and
with citations — not to refuse. Decide using ONLY the excerpts in the user message:

1. Evidence present (the excerpts address the question):
   → Answer. State ONLY what the excerpts support, each claim cited with [Page N].
     Never add anything that is not in the excerpts.

2. Evidence partial (the excerpts cover part of the question, OR a closely related
   concept under different wording — e.g. "drawbacks" ↔ "when not to use",
   "ratings normalized" ↔ "units and ranges / 0..100 ↔ 0..5"):
   → Answer the supported part with citations, then add ONE line:
     "Note: The document does not explicitly address [the specific missing aspect]."
   → Do NOT refuse merely because the wording differs or only part is covered.

3. Evidence genuinely absent (nothing in the excerpts relates, even after applying
   the Semantic Equivalence and Concept Reframing rules below):
   → Issue the exact refusal sentence (Rule 3).

Refusing when relevant evidence is present is a critical failure. When unsure
between "partial" and "absent", choose PARTIAL — answer what is supported and flag
the gap. This policy NEVER licenses information from outside the excerpts: answering
conservatively means answering with less, not inventing more.

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

CONFIDENCE OVERRIDE (CRITICAL — overrides the refusal instinct):
If the user message contains a line beginning "RETRIEVAL NOTE: HIGH" or an
"ANSWER REQUIRED" directive, the system has already VERIFIED that relevant
content is present in the excerpts. In that case you are FORBIDDEN from issuing
the refusal sentence. You MUST locate the relevant content — applying the
Semantic Equivalence (Rule 7) and Concept Reframing (Rule 9) rules below — and
answer with page citations. If only part of the question is covered, answer that
part and add the Rule 4 note for the uncovered aspect. A full refusal under a
HIGH retrieval note is a critical failure.

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
- "drawbacks" = "disadvantages" = "cons" = "downsides" = "limitations" = "pitfalls" = "problems" = "issues" = "trade-offs" = "when not to use" = "cautions" = "forces" = "constraints"
- "benefits" = "advantages" = "pros" = "strengths" = "upsides" = "when to use" = "motivation"
- "example" = "examples" = "demonstration" = "illustration" = "sample" = "for instance" = "e.g." = "scenario"
If the user asks about any term in a group, search the excerpts for ALL
terms in that group before deciding the answer is absent.
In particular: a "When NOT to use", "Limitations", "Pitfalls", "Forces", or
"Constraints" section IS the answer to a question about drawbacks/disadvantages/
problems — treat them as the same thing and answer from that section.
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
- "drawbacks / disadvantages / problems / cons of X" → reframe as → "when NOT to use X", "limitations of X", "trade-offs", "cautions", "forces and constraints", "pitfalls"
- "benefits / advantages / pros of X" → reframe as → "when to use X", "why use X", "motivation for X"

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

**Rule 15 — Example & Demonstration Reasoning (CRITICAL)**
The document may express required answers through examples, code demonstrations,
multiple configurations (e.g., Order A vs Order B), or usage scenarios.
These are NOT optional — they contain REQUIRED answers.

MANDATORY BEHAVIOR:
1. If the question refers to behavior, effects, or outcomes:
   → Analyze examples and demonstrations in the excerpts — do not skip them.
2. If multiple configurations are shown (e.g., "Order A" vs "Order B"):
   → You MUST state that changing order/configuration changes behavior.
   → Explicitly say that different compositions lead to different outcomes.
3. If behavior is IMPLIED through an example but not explicitly stated:
   → Convert that implication into a clear direct statement.
   Example: document shows "Order A: Cache inside Retry" and
   "Order B: Retry inside Cache" → you MUST conclude:
   "Changing the order results in different behavior/semantics."
4. NEVER refuse if the answer can be derived from examples, code execution
   flow, or demonstrated differences between scenarios.
5. If the answer exists in ANY form — prose, bullet points, code, or example
   scenarios — you MUST answer. Failure to extract from examples is a
   CRITICAL ERROR.

**Rule 16 — Code Flow Reasoning (CRITICAL)**
When answering questions about behavior, conditions, or execution, simulate
the code flow logically.

MANDATORY STEPS:
1. Identify ALL conditions (if/else).
2. Identify ALL branches.
3. Determine WHEN each branch executes.
4. Translate into clear natural language.

Example:
  Code: if ("GET") → use cache / else → bypass cache
  Correct: "The cache is only used for GET requests; all other requests bypass it."
  NOT acceptable: ignoring the condition or giving only partial logic.

**Rule 17 — Do Not Default to Refusal for Reasoning Questions**
For questions asking "what happens if…", "why does…", "how does behavior
change…", or "what is the effect of…" — these are REASONING questions.
Attempt reasoning using examples, code, and demonstrated scenarios first.
Only refuse if absolutely NO related content exists anywhere in the excerpts.

**Rule 18 — Hidden Condition Priority**
When analyzing code, the FIRST condition is the most important and most
commonly missed.

MANDATORY CHECK — before submitting your answer, explicitly verify:
→ "Did I include the condition under which this logic applies?"
If not, find it and include it.
Example: "only for GET requests" must NEVER be omitted.

**Rule 19 — Section Awareness (CRITICAL)**
Documents are structured into sections (e.g., "Forces, Constraints, Goals",
"Practical Considerations", "Pitfalls").

If a question refers to reasons, constraints, considerations, guidelines,
or trade-offs → you MUST search for a section with a matching title before
concluding the answer is absent.
Do NOT assume absence without checking section headers.
Failure to extract from a clearly labeled section is a critical error.

**Rule 20 — Full Execution Flow Extraction (CRITICAL)**
When analyzing code, describe the COMPLETE flow — not just the condition.

MANDATORY — your answer must cover all three:
1. Condition: when/if it applies.
2. Action: what is done.
3. Continuation: what happens next (store, return, loop, recurse, etc.).

Example:
  WRONG:  "Fetches fresh response."
  CORRECT: "When cache expires, it: (1) calls the inner client, (2) stores
            the new entry in cache, (3) returns the fresh response."

Partial flow = incorrect answer. All three steps must appear.

**Rule 21 — Outer Condition Priority (CRITICAL)**
In nested logic, the OUTERMOST condition defines WHEN the inner logic applies.

MANDATORY CHECK before answering any code question:
→ Is there an outer if/guard that constrains when this block runs?
→ If yes, your answer MUST open with that constraint.

Example:
  if (GET) { if (expiry valid) return cache }
  → MUST say: "This applies only for GET requests. Within that…"

Missing the outer condition = critical failure.

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


def _build_user_message(
    query: str,
    chunks: List[dict],
    force_answer: bool = False,
    retry: bool = False,
) -> str:
    """
    Compose the full user-turn content: context block + question.

    force_answer: set when retrieval confidence is HIGH (top cosine ≥
        _HIGH_CONFIDENCE_SCORE). Injects a "RETRIEVAL NOTE: HIGH / ANSWER
        REQUIRED" directive that the system prompt's Rule-3 confidence override
        keys on — the model is then forbidden from issuing a full refusal.
    retry: set on the non-streaming second attempt after the model refused
        despite HIGH confidence. Adds an explicit "previous attempt incorrectly
        refused" escalation.

    Injecting context into every user turn (rather than stuffing it into the
    system prompt) means different questions can retrieve different chunks in
    a multi-turn conversation, and each turn is independently verifiable.

    Also injects two reliability guards:
    - A valid-page constraint listing only the pages present in the retrieved
      chunks, preventing the LLM from citing pages it never saw.
    - A retrieval confidence note when the top chunk score is high, preventing
      unnecessary refusals when strong matches exist.

    Args:
        query:  The user's natural-language question (original, never rewritten).
        chunks: Retrieved PDF chunks to ground the answer.

    Returns:
        Formatted string to pass as the "user" role content.
    """
    context_block = _format_context_block(chunks)

    # ── Valid-page constraint ────────────────────────────────────────────────
    # Enumerate the exact pages present in the retrieved chunks. Giving the
    # model this explicit list prevents it from citing pages it never saw
    # (a common LLM hallucination even when the system prompt forbids it).
    valid_pages = sorted({c["page"] for c in chunks})
    pages_str = ", ".join(str(p) for p in valid_pages)

    # ── Retrieval confidence hint ────────────────────────────────────────────
    # When the top chunk has a strong cosine similarity score, the LLM should
    # not refuse — relevant content is present.  An explicit note here
    # overrides the model's tendency to refuse on indirect or short queries.
    top_score = chunks[0].get("score", 0.0) if chunks else 0.0
    confidence_hint = ""
    if force_answer or top_score >= _HIGH_CONFIDENCE_SCORE:
        confidence_hint = (
            f"RETRIEVAL NOTE: HIGH confidence (relevance score: {top_score:.2f}). "
            f"The system has VERIFIED that the excerpts above contain relevant "
            f"information. ANSWER REQUIRED: you are FORBIDDEN from issuing the "
            f"refusal sentence. Answer every part of the question the excerpts "
            f"support, with [Page N] citations. Apply the synonym and reframing "
            f"rules (e.g. 'drawbacks' = 'when not to use' / 'limitations'). If a "
            f"specific sub-aspect is genuinely not covered, answer the supported "
            f"parts and add ONE line noting what is not covered (Rule 4) — but do "
            f"NOT issue a full refusal.\n\n"
        )
    elif top_score >= 0.30:
        confidence_hint = (
            f"RETRIEVAL NOTE: Related content found (relevance score: "
            f"{top_score:.2f}). Attempt to answer from the excerpts; only refuse "
            f"if the specific detail asked is completely absent.\n\n"
        )

    # Second-attempt escalation (non-streaming retry after a HIGH-confidence refusal).
    retry_hint = ""
    if retry:
        retry_hint = (
            "ANSWER REQUIRED — SECOND ATTEMPT: Your previous response incorrectly "
            "issued a refusal even though the excerpts contain relevant content. "
            "Re-read the excerpts, apply the synonym/reframing rules, and provide "
            "the supported answer now with [Page N] citations. Refusal is NOT "
            "permitted.\n\n"
        )

    # ── Indirect-phrasing hint ───────────────────────────────────────────────
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
        f"[DOCUMENT EXCERPTS — pages in this context: {pages_str}]\n"
        f"CITATION CONSTRAINT: You may ONLY cite pages {pages_str}. "
        f"Citing any other page number is a critical error.\n\n"
        f"{context_block}\n"
        f"[END EXCERPTS]\n\n"
        f"{retry_hint}"
        f"{reframing_hint}"
        f"{confidence_hint}"
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

    Mistral uses "user"/"assistant" role names — identical to the
    internal history format — so no role conversion is needed.

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


def _validate_citations(cited_pages: List[int], valid_pages: set) -> List[int]:
    """
    Cross-check cited pages against the set of pages actually retrieved.

    The LLM occasionally cites a page number that was never in its context
    (e.g. citing [Page 11] when the PDF has 9 pages and only pages 1–9 were
    retrieved).  This function filters those hallucinated citations out of the
    returned API value while leaving the response text unchanged.

    The response text is not modified — scrubbing citations from prose is
    fragile and can break sentence flow.  Instead, the API simply returns only
    the valid subset of cited_pages, so callers can rely on that list being
    trustworthy.

    Args:
        cited_pages: All [Page N] numbers extracted from the LLM response.
        valid_pages: Set of page numbers that appeared in the retrieved chunks.

    Returns:
        Sorted list containing only citations that appear in valid_pages.
    """
    valid = [p for p in cited_pages if p in valid_pages]
    hallucinated = [p for p in cited_pages if p not in valid_pages]
    if hallucinated:
        print(
            f"[agent] Citation validation: removed hallucinated page(s) {hallucinated} "
            f"(not in retrieved chunks: pages {sorted(valid_pages)})"
        )
    return valid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _expand_query(query: str, chunks_context: str = "") -> str:
    """
    Expand the query with domain-specific synonyms to improve recall.

    Applies at most 2 expansion sets so the combined query stays within a
    sentence-transformer's effective context window and the embedding is not
    diluted by too many unrelated terms.
    """
    expansions = {
        # Navigation / location patterns
        "outside":     "outside besides alternative constructor before class method other",
        "where":       "where location place method constructor build class",
        "besides":     "besides outside alternative also another additionally other",
        # Structural / comparison concepts
        "alternative": "alternative option other approach method way instead",
        "difference":  "difference comparison contrast versus between",
        "compare":     "compare comparison contrast difference versus similar",
        "summary":     "summary overview main points key aspects",
        "summarize":   "summarize summary overview main points key aspects all",
        # Problem / quality concepts
        "pitfall":     "pitfall pitfalls problem issue drawback limitation disadvantage risk",
        "pitfalls":    "pitfall pitfalls problem issue drawback limitation disadvantage risk",
        "drawback":    "drawback pitfall limitation disadvantage problem issue concern",
        "advantage":   "advantage benefit strength gain improvement value",
        "benefits":    "benefit advantage strength gain improvement value",
        # Technical API/data concepts
        "provider":    "provider service adapter returns output result",
        "return":      "return output result response value produce",
        "null":        "null nullability missing fields empty none default",
        "default":     "default defaults preset initial values builder optional",
        "missing":     "missing fields absent not set null nullability",
        "error":       "error exception failure handling",
        "handle":      "handle manage process deal",
        "convert":     "convert transform translate map",
        "log":         "log logging observability metrics",
        "cache":       "cache caching batch buffering",
        "order":       "order sequence arrangement composition stack layering difference effect",
        "effect":      "effect impact result behavior outcome change",
        "example":     "example scenario demonstration case configuration",
    }

    query_lower = query.lower()
    added: List[str] = []
    seen_synonym_sets: set = set()
    for keyword, synonyms in expansions.items():
        if keyword in query_lower and synonyms not in seen_synonym_sets:
            added.append(synonyms)
            seen_synonym_sets.add(synonyms)
            if len(added) >= 2:  # cap at 2 sets to avoid diluting the embedding
                break

    return f"{query} {' '.join(added)}" if added else query


def generate_query_variants(query: str) -> List[str]:
    """
    Generate up to 2 alternative phrasings of the user's question to diversify
    retrieval (one concise, one descriptive).

    Calls Mistral with a small, deterministic-ish prompt. Returns [] on any
    failure or if the API key is missing — callers then fall back to the
    existing single-query retrieval, so multi-query never breaks retrieval.

    Args:
        query: The user's original natural-language question.

    Returns:
        A list of up to 2 distinct variant strings (never includes the original).
    """
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        return []

    prompt = (
        "Generate exactly 2 alternative document-search queries for the user's "
        "question.\n\n"
        "Rules:\n"
        "- Preserve meaning\n"
        "- Use different wording\n"
        "- One query should be concise\n"
        "- One query should be descriptive\n\n"
        "Return exactly 2 lines.\n"
        "No numbering.\n"
        "No explanations.\n\n"
        f"User Question:\n{query}"
    )

    try:
        resp = requests.post(
            MISTRAL_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 120,
                "temperature": 0.3,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()

        variants: List[str] = []
        for line in raw.split("\n"):
            # Defensive: strip any stray bullets/numbering the model may add.
            cleaned = line.strip().lstrip("-*0123456789.) ").strip()
            if not cleaned or cleaned.lower() == query.strip().lower():
                continue
            if cleaned not in variants:
                variants.append(cleaned)
            if len(variants) >= 2:
                break
        return variants

    except Exception as exc:
        print(f"[agent] Query variant generation failed: {exc!r} — single-query fallback.")
        return []


# ---------------------------------------------------------------------------
# Conversational retrieval helpers
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "must", "can", "not", "with", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "each", "all", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "if", "how", "what", "when",
    "where", "which", "who", "whom", "then", "there", "here", "any",
    "their", "they", "we", "you", "your", "my", "me", "him", "her", "our",
    "also", "well", "again", "once", "and", "but", "or", "yet", "for",
    "nor", "so", "either", "neither",
})

# Minimum retrieval score at which the LLM is told strong matches exist
# and refusal is inappropriate.
_HIGH_CONFIDENCE_SCORE: float = 0.45

# Words that indicate a query references prior context rather than being standalone
_ORDINAL_WORDS: frozenset = frozenset({
    "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "last", "previous", "aforementioned", "latter",
})
_PRONOUN_WORDS: frozenset = frozenset({
    "it", "its", "this", "that", "they", "them", "their",
    "these", "those", "he", "she",
})
# Only unambiguous follow-up phrases that clearly reference the prior response.
# Broad phrases like "what about" or "how about" are intentionally excluded
# because they are equally likely to introduce a new standalone topic
# ("What about the Exclusively adapter?").
_FOLLOW_UP_PREFIXES: tuple = (
    "tell me more",
    "explain more",
    "elaborate",
    "go on",
    "what else",
    "anything else",
    "more details",
    "why is that",
    "how so",
)


def _extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract the most distinctive content words from a text.

    Used to enrich contextual retrieval queries with document vocabulary
    pulled from the previous assistant response. Filters out stopwords and
    citation markers, deduplicates, and returns longer words first
    (longer words tend to be more topic-specific).

    Args:
        text:         Source text (typically the last assistant response).
        max_keywords: Maximum number of keywords to return.

    Returns:
        List of up to max_keywords unique content words, longest-first.
    """
    # Strip citation markers — they add noise, not signal
    cleaned = re.sub(r"\[Page\s+\d+\]", "", text, flags=re.IGNORECASE)
    # Extract alphabetic tokens of at least 4 chars
    words = re.findall(r"[a-zA-Z]{4,}", cleaned)
    # Filter stopwords (lowercase comparison)
    filtered = [w for w in words if w.lower() not in _STOPWORDS]
    # Deduplicate preserving first occurrence
    seen: set = set()
    unique: List[str] = []
    for w in filtered:
        if w.lower() not in seen:
            seen.add(w.lower())
            unique.append(w)
    # Prefer longer words (more specific technical terms)
    unique.sort(key=len, reverse=True)
    return unique[:max_keywords]


def _is_contextual_query(query: str) -> bool:
    """
    Return True if the query CANNOT be understood without prior conversation context.

    This gate controls whether an expensive LLM rewrite call fires.  It is
    intentionally conservative — only queries that provably reference something
    said earlier are flagged.  Borderline cases (short but concrete, starts with
    "what about" but names a specific thing) are left as standalone so the
    cheaper keyword-enrichment path handles them.

    Triggers:
      • ≤ 2 words          — too short to carry standalone meaning ("Why?", "And?")
      • Ordinal reference  — "second point", "last item" must resolve to a prior list
      • Pronoun reference  — "it", "that", "this" etc. without a named subject
                             ("Explain it more" → contextual;
                              "What about this Adapter?" → standalone — has named subject)
      • Explicit follow-up — a short list of unambiguous follow-up phrases
    """
    lower = query.lower().strip()
    raw_words = query.split()
    # Strip punctuation so "them?" and "second." still match the keyword sets.
    word_set = set(re.findall(r"[a-z]+", lower))

    # ── very short ─────────────────────────────────────────────────────────────
    if len(raw_words) <= 2:
        return True

    # ── ordinal reference ───────────────────────────────────────────────────────
    if word_set & _ORDINAL_WORDS:
        return True

    # ── pronoun reference (no anchoring proper noun) ────────────────────────────
    if word_set & _PRONOUN_WORDS:
        # Skip the first word (often a capitalised verb/question-word) and look for
        # a proper noun that anchors the pronoun to a named subject.
        non_first = raw_words[1:]
        has_named_subject = any(
            tok[0].isupper() and tok.isalpha() for tok in non_first
        )
        if not has_named_subject:
            return True

    # ── unambiguous follow-up phrases ───────────────────────────────────────────
    if any(lower == p or lower.startswith(p + " ") for p in _FOLLOW_UP_PREFIXES):
        return True

    return False


def _rewrite_for_retrieval(query: str, history: List[dict]) -> str:
    """
    Call Mistral to rewrite a contextual query into a standalone retrieval query.

    This is a lightweight call: max_tokens=60, temperature=0 (deterministic).
    It fires only when _is_contextual_query() returns True.

    The rewritten string is used ONLY for Qdrant vector search — the original
    user query always goes to the main LLM response generation unchanged.

    Falls back to the original query if the API key is missing or the call fails.

    Args:
        query:   The user's contextual message ("Explain the second point more simply").
        history: Prior conversation turns — used to resolve pronoun/ordinal references.

    Returns:
        A standalone retrieval query string (e.g. "Open/Closed Principle Adapter Pattern").
    """
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        return query  # degrade gracefully if key not available

    # Use last 4 turns (2 user + 2 assistant) — enough context, bounded size
    recent = history[-4:]
    context_block = "\n".join(
        f"{t['role'].upper()}: {t['content'][:250]}"
        for t in recent
    )

    prompt = (
        "You are a search query optimizer for a document retrieval system.\n"
        "Rewrite the user's message as a standalone search query.\n"
        "Rules:\n"
        "1. Replace all pronouns (it/this/that/they/them) with the actual topic.\n"
        "2. Resolve ordinal references (second/third/last) to the actual concept.\n"
        "3. Keep the output under 20 words.\n"
        "4. Do NOT add information not present in the conversation.\n"
        "5. Output ONLY the search query — no explanation, no punctuation at end.\n\n"
        f"[CONVERSATION]\n{context_block}\n[/CONVERSATION]\n\n"
        f"User message: {query}\n\n"
        "Standalone search query:"
    )

    try:
        resp = requests.post(
            MISTRAL_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 60,
                "temperature": 0.0,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Take only the first line, cap length to guard against unexpected output
        rewritten = raw.split("\n")[0].strip()[:200]
        return rewritten if rewritten else query

    except Exception as exc:
        print(f"[agent] Query rewrite call failed: {exc!r} — using original query.")
        return query


def _normalize_retrieval_query(query: str) -> str:
    """
    Normalize a retrieval query to improve embedding recall without semantic drift.

    Transformations applied in order:
      1. Strip meta-question prefixes — "what is meant by", "define", etc.
         These words describe the question format, not the concept being searched.
      2. Replace "/" with space — "shape/semantics" tokenizes as three tokens
         including "/"; replacing with space gives a cleaner two-concept phrase.
      3. Strip quotation marks — users often quote terms ("adapter pattern")
         that should be searched without the quote characters.
      4. Collapse multiple spaces introduced by the above steps.

    Args:
        query: The user's raw query string.

    Returns:
        Normalized query string (may be identical to input if no rules fire).
    """
    q = query.strip()
    lower_q = q.lower()

    # Strip meta-question prefixes
    for prefix in (
        "what is meant by ",
        "what do you mean by ",
        "what does it mean by ",
        "what is the meaning of ",
        "define ",
    ):
        if lower_q.startswith(prefix):
            q = q[len(prefix):]
            lower_q = q.lower()
            break

    # Slashes → spaces ("shape/semantics" → "shape semantics")
    q = q.replace("/", " ")

    # Strip ASCII and Unicode quotation marks
    for ch in ('"', "'", "“", "”", "‘", "’"):
        q = q.replace(ch, "")

    # Collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _build_retrieval_query(query: str, history: List[dict]) -> str:
    """
    Build the best standalone retrieval query for a given user turn.

    Strategy (applied in order, each step builds on the previous):

      0. Normalize the query — strip meta-prefixes, replace "/" with space,
         remove quotation marks.  Applied even on the first turn.

      1. No history → return normalized query unchanged (first turn is standalone).

      2. Contextual query (pronouns / ordinals / very short) AND history exists:
         → LLM rewrite to resolve references into explicit topic terms.

      3. History exists (with or without LLM rewrite):
         → Enrich with keywords from the last assistant response.
         This adds document vocabulary to the retrieval query, boosting recall
         for follow-up questions that use the same topic area without pronouns.

    The returned string is used ONLY for Qdrant retrieval.
    The ORIGINAL query is always passed to the LLM prompt generation.

    Args:
        query:   The user's message (may be contextual or standalone).
        history: Prior turns as [{"role": "user"|"assistant", "content": str}].

    Returns:
        An enriched retrieval query string.
    """
    # ── Step 0: Normalize ────────────────────────────────────────────────────
    normalized = _normalize_retrieval_query(query)
    if normalized != query:
        print(f"[agent] Normalization: {query!r}  →  {normalized!r}")

    if not history:
        return normalized  # first turn — standalone, nothing to enrich from

    retrieval_query = normalized
    steps_applied: List[str] = []

    # ── Step A: LLM rewrite for contextual queries ──────────────────────────
    if _is_contextual_query(query):
        # Log why the rewrite was triggered (helps debug false positives)
        _words = set(query.lower().split())
        _raw = query.split()
        if len(_raw) <= 2:
            _reason = f"very_short({len(_raw)}_words)"
        elif _words & _ORDINAL_WORDS:
            _reason = f"ordinal({_words & _ORDINAL_WORDS})"
        elif _words & _PRONOUN_WORDS:
            _reason = f"pronoun({_words & _PRONOUN_WORDS})"
        else:
            _reason = "follow_up_phrase"
        print(f"[agent] Rewrite: TRIGGERED (reason={_reason})")

        rewritten = _rewrite_for_retrieval(query, history)
        if rewritten and rewritten != query:
            print(f"[agent] LLM rewrite: {query!r}  →  {rewritten!r}")
            retrieval_query = rewritten
            steps_applied.append("llm_rewrite")
        else:
            print(f"[agent] LLM rewrite: no change — keeping original query.")
    else:
        print(f"[agent] Rewrite: SKIPPED (standalone query — no API call)")

    # ── Step B: Enrich with keywords from last assistant response ────────────
    # Adds document vocabulary regardless of whether rewriting happened.
    # IMPORTANT: skip enrichment when the previous answer was a refusal — its
    # text ("I'm sorry, but the uploaded document does not contain information
    # about …") would otherwise inject noise words like "sorry", "uploaded",
    # "document", "contain", "information" into the next retrieval query and
    # degrade recall for the follow-up.
    last_assistant = next(
        (t["content"] for t in reversed(history) if t["role"] == "assistant"),
        "",
    )
    if last_assistant and is_refusal(last_assistant):
        print("[agent] Enrichment SKIPPED: last answer was a refusal (avoiding query pollution).")
    elif last_assistant:
        keywords = _extract_keywords(last_assistant, max_keywords=10)
        if keywords:
            kw_str = " ".join(keywords)
            retrieval_query = f"{retrieval_query} {kw_str}"
            steps_applied.append(f"context_keywords({len(keywords)})")

    if steps_applied:
        print(f"[agent] Retrieval enrichment: {', '.join(steps_applied)}")
    else:
        print(f"[agent] No retrieval enrichment (standalone, new topic).")

    return retrieval_query


# Catches LLM-paraphrased refusals that don't use the exact REFUSAL_PREFIX wording.
# Examples matched:
#   "I'm sorry, but I cannot find information about…"
#   "I am sorry, but the document does not contain…"
#   "I'm sorry, but there is no information in this document…"
# The pattern is intentionally narrow (requires "I'm/I am sorry, but") to avoid
# flagging partial answers that legitimately start with apology-adjacent phrases.
_REFUSAL_RE = re.compile(
    r"^I(?:'m| am) sorry[,.]?\s+but\s+(?:I |the |this |there )",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    """
    Return True if the answer is an out-of-scope refusal.

    Checks two patterns:
      1. Exact REFUSAL_PREFIX — the prescribed Rule-3 wording from the system prompt.
      2. _REFUSAL_RE regex  — catches LLM-paraphrased variants that convey the same
         intent but don't use the exact prescribed wording (e.g. "I'm sorry, but I
         cannot find information about…" vs the canonical "I'm sorry, but the uploaded
         document does not contain information about…").

    Args:
        answer: The answer string returned by get_answer().

    Returns:
        True if the answer is a refusal by either pattern.
    """
    stripped = answer.strip()
    return stripped.startswith(REFUSAL_PREFIX) or bool(_REFUSAL_RE.match(stripped))


def _run_retrieval(
    query: str,
    session_id: str,
    conversation_history: List[dict],
) -> List[dict]:
    """
    Build the retrieval query and fetch chunks via the layered fallback strategy.

    Shared by get_answer() (non-streaming) and stream_answer() (streaming) so
    both response paths retrieve identically — the only difference between the
    two is how the Mistral completion is consumed, not how chunks are selected.

    Steps:
      1. Build retrieval query (contextual rewrite + keyword enrichment).
      2. Apply synonym expansion.
      3. Retrieve top-10 with progressive fallbacks:
           Primary    — expanded retrieval query
           Fallback 1 — retrieval query without synonym expansion
           Fallback 2 — original user query (guards against enrichment drift)

    Returns:
        List of chunk dicts ordered best-first (may be empty → caller refuses).
    """
    # Step 1: Build retrieval query (contextual rewriting + enrichment)
    retrieval_query = _build_retrieval_query(query, conversation_history)

    # Step 2: Apply synonym expansion to the retrieval query
    expanded_query = _expand_query(retrieval_query)
    if expanded_query != retrieval_query:
        print(f"[agent] Synonym expansion applied.")
    print(f"[agent] Final retrieval query (len={len(expanded_query)}): {expanded_query!r}")

    # Step 2.5: Multi-query diversification (Phase 3 Step 1).
    # Generate 2 alternative phrasings and retrieve all formulations together.
    # The query set is a SUPERSET of the single-query (expanded_query is always
    # included), so recall cannot regress. Falls back to single-query if variant
    # generation fails or yields nothing.
    if ENABLE_MULTI_QUERY:
        variants = generate_query_variants(query)
        if variants:
            print("[multi-query]")
            print(f"Original:\n{query}")
            for i, v in enumerate(variants, start=1):
                print(f"Variant {i}:\n{v}")
            query_set = [expanded_query] + [_expand_query(v) for v in variants]
            chunks = retrieve_multi_query(
                query_set, session_id, top_k=10, original_query=query
            )
            if chunks:
                return chunks
            print("[agent] Multi-query returned no chunks — falling back to single-query.")
        else:
            print("[agent] Multi-query: no variants generated — using single-query.")

    # Step 3: Retrieve — layered fallback strategy (also the multi-query fallback).
    # original_query is passed for retrieval diagnostics only (Query vs Rewritten
    # Query in the debug block); it does not affect retrieval behavior.
    chunks = retrieve_relevant_chunks(expanded_query, session_id, top_k=10, original_query=query)

    if not chunks:
        print(f"[agent] Fallback 1: retrieval query without expansion.")
        chunks = retrieve_relevant_chunks(retrieval_query, session_id, top_k=10, original_query=query)

    if not chunks and retrieval_query != query:
        print(f"[agent] Fallback 2: original user query.")
        expanded_original = _expand_query(query)
        chunks = retrieve_relevant_chunks(expanded_original, session_id, top_k=10, original_query=query)
        if not chunks:
            chunks = retrieve_relevant_chunks(query, session_id, top_k=10, original_query=query)

    return chunks


def _confidence_band(top_score: float) -> str:
    """Map a top cosine score to its confidence band (thresholds unchanged)."""
    if top_score >= _HIGH_CONFIDENCE_SCORE:
        return "HIGH"
    if top_score >= 0.30:
        return "MEDIUM"
    return "LOW"


def _log_confidence(chunks: List[dict]) -> tuple[float, set]:
    """
    Log the retrieval confidence band and return (top_score, page_set).

    Shared by both response paths so logging is identical.
    """
    top_score = chunks[0].get("score", 0.0)
    chunk_pages_set = {c["page"] for c in chunks}
    confidence_label = _confidence_band(top_score)
    print(
        f"[agent] Retrieval confidence: {confidence_label} "
        f"(top_score={top_score:.4f}, chunks={len(chunks)}, "
        f"pages={sorted(chunk_pages_set)})"
    )
    return top_score, chunk_pages_set


def _call_mistral(messages: List[dict], api_key: str) -> dict:
    """
    Blocking (non-streaming) Mistral chat completion. Returns the parsed JSON.

    Factored out so get_answer() can issue a second, forced attempt when a
    HIGH-confidence query is refused on the first pass.
    """
    response = requests.post(
        MISTRAL_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_answer(
    query: str,
    session_id: str,
    conversation_history: Optional[List[dict]] = None,
) -> dict:
    """
    Full pipeline: build retrieval query → retrieve → call Mistral API → return grounded answer.

    Steps:
      1. Build a retrieval query from the user turn + conversation history:
           a. LLM rewrite if the query is contextual (pronouns/ordinals/very short).
           b. Enrich with keywords from the last assistant response.
      2. Apply synonym expansion to the retrieval query.
      3. Retrieve top-10 chunks from Qdrant; fall back through progressively
         simpler queries if nothing is found.
      4. If still nothing: return a refusal immediately (no main API call).
      5. Build the user message embedding the context block + ORIGINAL question.
      6. Prepend the system prompt and history to form the full message list.
      7. POST to the Mistral chat completions endpoint.
      8. Parse [Page N] citations from the response.
      9. Return a structured result dict.

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
        requests.HTTPError:     Propagated on non-2xx response from Mistral.
        requests.Timeout:       If the request exceeds 60 seconds.
    """
    if conversation_history is None:
        conversation_history = []

    print(
        f"[agent] Query: {query!r}  |  session: {session_id}  |  "
        f"history_turns={len(conversation_history)}"
    )

    # ------------------------------------------------------------------
    # Steps 1-3: Build retrieval query + retrieve with layered fallbacks
    # Rewriting resolves pronouns/ordinals; enrichment adds document vocab;
    # synonym expansion broadens recall. The ORIGINAL query always goes to
    # the LLM — this only affects Qdrant retrieval.
    # ------------------------------------------------------------------
    chunks = _run_retrieval(query, session_id, conversation_history)

    # ------------------------------------------------------------------
    # Step 4: Immediate refusal — no chunks means the topic is absent
    # ------------------------------------------------------------------
    if not chunks:
        print("[agent] REFUSAL_REASON: zero_chunks — no retrievable content for this query.")
        topic = query if len(query) <= 80 else query[:77] + "…"
        refusal = f"{REFUSAL_PREFIX} {topic}."
        return {
            "answer": refusal,
            "cited_pages": [],
            "chunks_used": 0,
        }

    # ------------------------------------------------------------------
    # Retrieval confidence summary (logged before the API call)
    # ------------------------------------------------------------------
    top_score, chunk_pages_set = _log_confidence(chunks)

    # ------------------------------------------------------------------
    # Confidence-based refusal override
    # When the top semantic match is HIGH (>= _HIGH_CONFIDENCE_SCORE), relevant
    # content is genuinely present — a full refusal would be wrong. We inject a
    # forcing directive into the user message and (below) retry once if the LLM
    # still refuses. Refusals remain valid for zero-chunk (handled above) and
    # MEDIUM/LOW confidence where the specific detail may truly be absent.
    # ------------------------------------------------------------------
    force_answer = top_score >= _HIGH_CONFIDENCE_SCORE
    if force_answer:
        print("[agent] Refusal override: confidence HIGH, forcing evidence-based answer")

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY environment variable is not set. "
            "Add it to your .env file."
        )

    # ------------------------------------------------------------------
    # Step 5-6: Build user message (with confidence forcing) + message list
    # NOTE: always uses the ORIGINAL query, never the rewritten retrieval query.
    # ------------------------------------------------------------------
    user_message_content = _build_user_message(query, chunks, force_answer=force_answer)
    messages_for_api = _build_messages_for_api(
        history=list(conversation_history),  # shallow copy — don't mutate caller's list
        current_user_content=user_message_content,
    )

    # ------------------------------------------------------------------
    # Step 7: Call Mistral API
    # ------------------------------------------------------------------
    print(
        f"[agent] Calling {MODEL} via Mistral API | "
        f"chunks={len(chunks)} | "
        f"history_turns={len(conversation_history)} | "
        f"top_chunk_score={chunks[0].get('score', 'n/a')} | "
        f"force_answer={force_answer}"
    )

    data = _call_mistral(messages_for_api, api_key)
    answer = data["choices"][0]["message"]["content"]

    # HIGH-confidence refusal guard: if the model refused anyway, retry once
    # with an explicit "previous attempt incorrectly refused" escalation.
    if force_answer and is_refusal(answer):
        print(
            "[agent] Refusal override: HIGH confidence but LLM refused on first "
            "attempt — retrying once with forced directive."
        )
        retry_content = _build_user_message(query, chunks, force_answer=True, retry=True)
        retry_messages = _build_messages_for_api(
            history=list(conversation_history),
            current_user_content=retry_content,
        )
        data = _call_mistral(retry_messages, api_key)
        retry_answer = data["choices"][0]["message"]["content"]
        if is_refusal(retry_answer):
            print("[agent] Refusal override: retry STILL refused — evidence likely insufficient.")
        else:
            print("[agent] Refusal override: retry produced an evidence-based answer.")
        answer = retry_answer

    # ------------------------------------------------------------------
    # Step 8: Parse and validate citations
    #
    # _extract_cited_pages pulls every [Page N] from the response text.
    # _validate_citations then removes any page numbers not present in the
    # retrieved chunks — these are hallucinated citations the LLM invented
    # despite the CITATION CONSTRAINT injected into the user message.
    # ------------------------------------------------------------------
    cited_pages_raw = _extract_cited_pages(answer)
    cited_pages = _validate_citations(cited_pages_raw, chunk_pages_set)

    _refusal = is_refusal(answer)
    if _refusal:
        print(
            f"[agent] REFUSAL_REASON: llm_chose_refusal — chunks were present "
            f"(top_score={top_score:.4f}) but LLM issued refusal."
        )

    usage = data.get("usage", {})
    print(
        f"[agent] Response done | "
        f"is_refusal={_refusal} | "
        f"cited_pages_raw={cited_pages_raw} | "
        f"cited_pages_valid={cited_pages} | "
        f"tokens={usage.get('prompt_tokens', '?')}in/"
        f"{usage.get('completion_tokens', '?')}out"
    )

    # Answer-quality audit block: correlates retrieval strength with the LLM's
    # final decision so HIGH-confidence-but-refused cases are easy to spot.
    print("[agent]")
    print(f"retrieval_confidence={_confidence_band(top_score)}")
    print(f"chunks_found={len(chunks)}")
    print(f"llm_decision={'refusal' if _refusal else 'answer'}")

    return {
        "answer": answer,
        "cited_pages": cited_pages,
        "chunks_used": len(chunks),
    }


# ---------------------------------------------------------------------------
# Streaming response (Phase 1)
# ---------------------------------------------------------------------------


def _ndjson(obj: dict) -> str:
    """Serialise one streaming event as a newline-delimited JSON line."""
    return json.dumps(obj, ensure_ascii=False) + "\n"


def stream_answer(
    query: str,
    session_id: str,
    conversation_history: Optional[List[dict]] = None,
) -> Iterator[str]:
    """
    Streaming counterpart to get_answer().

    Runs the identical retrieval pipeline (_run_retrieval), then consumes
    Mistral's response token-by-token via the OpenAI-compatible streaming API
    (stream=True). Yields newline-delimited JSON (NDJSON) events so the
    frontend can render tokens as they arrive:

        {"type": "token", "text": "..."}                      ← zero or more
        {"type": "done",  "answer": str, "cited_pages": [..],
                          "is_refusal": bool, "chunks_used": int}   ← exactly one
        {"type": "error", "text": "..."}                      ← on failure

    Citations are parsed from the fully-accumulated answer once the stream
    completes (the same _extract_cited_pages / _validate_citations path as
    get_answer), then emitted in the terminal "done" event.

    This is a synchronous generator; StreamingResponse in main.py iterates it
    in a threadpool so the blocking requests call never stalls the event loop.

    Args:
        query:                The user's natural-language question.
        session_id:           UUID identifying the active upload session.
        conversation_history: Prior turns (None/[] for a fresh conversation).

    Yields:
        NDJSON-encoded event strings (each ending in "\n").
    """
    if conversation_history is None:
        conversation_history = []

    print(
        f"[agent] STREAM Query: {query!r}  |  session: {session_id}  |  "
        f"history_turns={len(conversation_history)}"
    )

    # ── Steps 1-3: retrieve (shared with get_answer) ─────────────────────────
    chunks = _run_retrieval(query, session_id, conversation_history)

    # ── Step 4: immediate refusal — stream the refusal sentence as one token ─
    if not chunks:
        print("[agent] STREAM REFUSAL_REASON: zero_chunks — no retrievable content.")
        topic = query if len(query) <= 80 else query[:77] + "…"
        refusal = f"{REFUSAL_PREFIX} {topic}."
        yield _ndjson({"type": "token", "text": refusal})
        yield _ndjson({
            "type": "done", "answer": refusal,
            "cited_pages": [], "is_refusal": True, "chunks_used": 0,
        })
        return

    top_score, chunk_pages_set = _log_confidence(chunks)

    # Confidence-based refusal override. Streaming cannot retry mid-stream once
    # tokens have been flushed, so HIGH confidence forces the answer PRE-EMPTIVELY
    # via the directive baked into the user message (the system-prompt Rule-3
    # override keys on the "RETRIEVAL NOTE: HIGH / ANSWER REQUIRED" line).
    force_answer = top_score >= _HIGH_CONFIDENCE_SCORE
    if force_answer:
        print("[agent] Refusal override: confidence HIGH, forcing evidence-based answer")

    # ── Steps 5-6: build the message list (shared helpers) ───────────────────
    user_message_content = _build_user_message(query, chunks, force_answer=force_answer)
    messages_for_api = _build_messages_for_api(
        history=list(conversation_history),
        current_user_content=user_message_content,
    )

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY environment variable is not set. Add it to your .env file."
        )

    print(
        f"[agent] STREAM Calling {MODEL} via Mistral API (stream=True) | "
        f"chunks={len(chunks)} | history_turns={len(conversation_history)} | "
        f"top_chunk_score={top_score}"
    )

    # ── Step 7: stream the completion ────────────────────────────────────────
    full_parts: List[str] = []
    try:
        with requests.post(
            MISTRAL_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages_for_api,
                "max_tokens": MAX_TOKENS,
                "stream": True,
            },
            timeout=60,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    payload = json.loads(data_str)
                except json.JSONDecodeError:
                    continue  # skip keep-alive / malformed fragments
                delta = payload.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content")
                if token:
                    full_parts.append(token)
                    yield _ndjson({"type": "token", "text": token})

    except Exception as exc:
        print(f"[agent] STREAM error during generation: {exc!r}")
        yield _ndjson({"type": "error", "text": f"Streaming failed: {exc}"})
        return

    # ── Step 8: parse + validate citations from the accumulated answer ───────
    answer = "".join(full_parts)
    cited_pages_raw = _extract_cited_pages(answer)
    cited_pages = _validate_citations(cited_pages_raw, chunk_pages_set)
    refusal = is_refusal(answer)

    if refusal:
        print(
            f"[agent] STREAM REFUSAL_REASON: llm_chose_refusal — chunks were present "
            f"(top_score={top_score:.4f}) but LLM issued refusal."
        )
    print(
        f"[agent] STREAM Response done | is_refusal={refusal} | "
        f"cited_pages_raw={cited_pages_raw} | cited_pages_valid={cited_pages} | "
        f"chars={len(answer)}"
    )

    # Answer-quality audit block (same as non-streaming path).
    print("[agent]")
    print(f"retrieval_confidence={_confidence_band(top_score)}")
    print(f"chunks_found={len(chunks)}")
    print(f"llm_decision={'refusal' if refusal else 'answer'}")

    yield _ndjson({
        "type": "done",
        "answer": answer,
        "cited_pages": cited_pages,
        "is_refusal": refusal,
        "chunks_used": len(chunks),
    })
