"""
citation_service.py — Parse, validate, and classify answer citations.

Two responsibilities, both pure (no I/O):
  • Extract [Page N] citations from an answer and drop any page the model was
    never shown (anti-hallucination — the response text is left untouched).
  • Detect refusals, including LLM-paraphrased variants of the canonical
    refusal sentence.
"""

from __future__ import annotations

import re
from typing import List

# The prefix used for every out-of-scope refusal. is_refusal() matches on this
# exact string, so keep it in sync with the system prompt's Rule 3.
REFUSAL_PREFIX = "I'm sorry, but the uploaded document does not contain information about"

# Catches LLM-paraphrased refusals that don't use the exact REFUSAL_PREFIX wording,
# e.g. "I'm sorry, but I cannot find information about…". Intentionally narrow
# (requires "I'm/I am sorry, but") to avoid flagging legitimate partial answers.
_REFUSAL_RE = re.compile(
    r"^I(?:'m| am) sorry[,.]?\s+but\s+(?:I |the |this |there )",
    re.IGNORECASE,
)

_PAGE_RE = re.compile(r"\[Page\s+(\d+)\]", flags=re.IGNORECASE)


def extract_cited_pages(text: str) -> List[int]:
    """Return deduplicated, sorted [Page N] numbers parsed from the answer."""
    return sorted({int(m) for m in _PAGE_RE.findall(text)})


def validate_citations(cited_pages: List[int], valid_pages: set) -> List[int]:
    """
    Keep only citations to pages that actually appeared in the retrieved chunks.

    The response text is not modified — scrubbing prose is fragile — so callers
    rely on the returned list being the trustworthy subset.
    """
    valid = [p for p in cited_pages if p in valid_pages]
    hallucinated = [p for p in cited_pages if p not in valid_pages]
    if hallucinated:
        print(
            f"[citation] Removed hallucinated page(s) {hallucinated} "
            f"(not in retrieved chunks: pages {sorted(valid_pages)})"
        )
    return valid


def is_refusal(answer: str) -> bool:
    """True if the answer is an out-of-scope refusal (exact prefix or paraphrase)."""
    stripped = answer.strip()
    return stripped.startswith(REFUSAL_PREFIX) or bool(_REFUSAL_RE.match(stripped))
