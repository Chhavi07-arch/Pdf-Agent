"""
query_builder.py — Build standalone retrieval queries from user turns.

Turns a (possibly contextual) user message into the best query for vector
search: normalize punctuation/meta-prefixes, resolve pronouns/ordinals via a
lightweight LLM rewrite, enrich with keywords from the last answer, and expand
with domain synonyms. The rewritten query is used ONLY for retrieval — the
original question always goes to the LLM prompt (retrieval-drift prevention).
"""

from __future__ import annotations

import re
from typing import List

from app.interfaces.llm_client import LLMClient
from app.services.citation_service import is_refusal

# Synonym expansion sets (domain vocabulary) keyed by trigger word.
_EXPANSIONS = {
    "outside":     "outside besides alternative constructor before class method other",
    "where":       "where location place method constructor build class",
    "besides":     "besides outside alternative also another additionally other",
    "alternative": "alternative option other approach method way instead",
    "difference":  "difference comparison contrast versus between",
    "compare":     "compare comparison contrast difference versus similar",
    "summary":     "summary overview main points key aspects",
    "summarize":   "summarize summary overview main points key aspects all",
    "pitfall":     "pitfall pitfalls problem issue drawback limitation disadvantage risk",
    "pitfalls":    "pitfall pitfalls problem issue drawback limitation disadvantage risk",
    "drawback":    "drawback pitfall limitation disadvantage problem issue concern",
    "advantage":   "advantage benefit strength gain improvement value",
    "benefits":    "benefit advantage strength gain improvement value",
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

_STOPWORDS = frozenset({
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

_ORDINAL_WORDS = frozenset({
    "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "last", "previous", "aforementioned", "latter",
})
_PRONOUN_WORDS = frozenset({
    "it", "its", "this", "that", "they", "them", "their",
    "these", "those", "he", "she",
})
_FOLLOW_UP_PREFIXES = (
    "tell me more", "explain more", "elaborate", "go on", "what else",
    "anything else", "more details", "why is that", "how so",
)


class QueryBuilder:
    """Builds retrieval queries; uses an LLMClient for contextual rewrites/variants."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    # ── public API ────────────────────────────────────────────────────────────
    def build_retrieval_query(self, query: str, history: List[dict]) -> str:
        """Normalize → (rewrite if contextual) → enrich with last-answer keywords."""
        normalized = self._normalize(query)
        if normalized != query:
            print(f"[query] Normalization: {query!r}  →  {normalized!r}")

        if not history:
            return normalized

        retrieval_query = normalized
        steps: List[str] = []

        if self._is_contextual(query):
            rewritten = self._rewrite(query, history)
            if rewritten and rewritten != query:
                print(f"[query] LLM rewrite: {query!r}  →  {rewritten!r}")
                retrieval_query = rewritten
                steps.append("llm_rewrite")
            else:
                print("[query] LLM rewrite: no change — keeping original query.")
        else:
            print("[query] Rewrite: SKIPPED (standalone query — no API call)")

        last_assistant = next(
            (t["content"] for t in reversed(history) if t["role"] == "assistant"), ""
        )
        if last_assistant and is_refusal(last_assistant):
            print("[query] Enrichment SKIPPED: last answer was a refusal.")
        elif last_assistant:
            keywords = self._extract_keywords(last_assistant)
            if keywords:
                retrieval_query = f"{retrieval_query} {' '.join(keywords)}"
                steps.append(f"context_keywords({len(keywords)})")

        print(f"[query] Retrieval enrichment: {', '.join(steps) if steps else 'none'}")
        return retrieval_query

    def expand_query(self, query: str) -> str:
        """Append up to 2 domain-synonym sets to broaden recall."""
        query_lower = query.lower()
        added: List[str] = []
        seen: set = set()
        for keyword, synonyms in _EXPANSIONS.items():
            if keyword in query_lower and synonyms not in seen:
                added.append(synonyms)
                seen.add(synonyms)
                if len(added) >= 2:
                    break
        return f"{query} {' '.join(added)}" if added else query

    def generate_variants(self, query: str) -> List[str]:
        """Up to 2 alternative phrasings (one concise, one descriptive); [] on failure."""
        if not self._llm.available:
            return []
        prompt = (
            "Generate exactly 2 alternative document-search queries for the user's "
            "question.\n\nRules:\n- Preserve meaning\n- Use different wording\n"
            "- One query should be concise\n- One query should be descriptive\n\n"
            "Return exactly 2 lines.\nNo numbering.\nNo explanations.\n\n"
            f"User Question:\n{query}"
        )
        try:
            raw = self._llm.complete([{"role": "user", "content": prompt}], max_tokens=120, temperature=0.3)
        except Exception as exc:  # noqa: BLE001 — never break retrieval
            print(f"[query] Variant generation failed: {exc!r} — single-query fallback.")
            return []

        variants: List[str] = []
        for line in raw.strip().split("\n"):
            cleaned = line.strip().lstrip("-*0123456789.) ").strip()
            if not cleaned or cleaned.lower() == query.strip().lower():
                continue
            if cleaned not in variants:
                variants.append(cleaned)
            if len(variants) >= 2:
                break
        return variants

    # ── internals ─────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(query: str) -> str:
        q = query.strip()
        lower_q = q.lower()
        for prefix in (
            "what is meant by ", "what do you mean by ", "what does it mean by ",
            "what is the meaning of ", "define ",
        ):
            if lower_q.startswith(prefix):
                q = q[len(prefix):]
                break
        q = q.replace("/", " ")
        for ch in ('"', "'", "“", "”", "‘", "’"):
            q = q.replace(ch, "")
        return re.sub(r"\s+", " ", q).strip()

    @staticmethod
    def _is_contextual(query: str) -> bool:
        lower = query.lower().strip()
        raw_words = query.split()
        word_set = set(re.findall(r"[a-z]+", lower))
        if len(raw_words) <= 2:
            return True
        if word_set & _ORDINAL_WORDS:
            return True
        if word_set & _PRONOUN_WORDS:
            non_first = raw_words[1:]
            has_named_subject = any(tok[0].isupper() and tok.isalpha() for tok in non_first)
            if not has_named_subject:
                return True
        if any(lower == p or lower.startswith(p + " ") for p in _FOLLOW_UP_PREFIXES):
            return True
        return False

    @staticmethod
    def _extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        cleaned = re.sub(r"\[Page\s+\d+\]", "", text, flags=re.IGNORECASE)
        words = re.findall(r"[a-zA-Z]{4,}", cleaned)
        filtered = [w for w in words if w.lower() not in _STOPWORDS]
        seen: set = set()
        unique: List[str] = []
        for w in filtered:
            if w.lower() not in seen:
                seen.add(w.lower())
                unique.append(w)
        unique.sort(key=len, reverse=True)
        return unique[:max_keywords]

    def _rewrite(self, query: str, history: List[dict]) -> str:
        if not self._llm.available:
            return query
        recent = history[-4:]
        context_block = "\n".join(f"{t['role'].upper()}: {t['content'][:250]}" for t in recent)
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
            f"User message: {query}\n\nStandalone search query:"
        )
        try:
            raw = self._llm.complete([{"role": "user", "content": prompt}], max_tokens=60, temperature=0.0)
        except Exception as exc:  # noqa: BLE001 — degrade to original query
            print(f"[query] Rewrite call failed: {exc!r} — using original query.")
            return query
        rewritten = raw.split("\n")[0].strip()[:200]
        return rewritten if rewritten else query
