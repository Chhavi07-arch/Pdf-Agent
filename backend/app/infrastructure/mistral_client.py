"""
mistral_client.py — LLMClient backed by the Mistral chat completions REST API.

Mistral exposes an OpenAI-compatible endpoint, so this uses plain requests — no
SDK. Provides a blocking `complete` and a token-streaming `stream`; both raise
RuntimeError if no API key is configured.
"""

from __future__ import annotations

import json
from typing import Iterator, List, Optional

import requests  # type: ignore

from app.config import MISTRAL_API_URL, MISTRAL_MODEL
from app.interfaces.llm_client import LLMClient


class MistralClient(LLMClient):
    """Chat-completion client for Mistral (OpenAI-compatible messages)."""

    def __init__(
        self,
        api_key: str,
        model: str = MISTRAL_MODEL,
        url: str = MISTRAL_API_URL,
        timeout: int = 60,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._model = model
        self._url = url
        self._timeout = timeout

    @property
    def available(self) -> bool:
        """True if an API key is configured (callers can skip optional LLM calls)."""
        return bool(self._api_key)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _require_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY environment variable is not set. Add it to your .env file."
            )

    def complete(
        self,
        messages: List[dict],
        max_tokens: int,
        temperature: Optional[float] = None,
    ) -> str:
        self._require_key()
        payload: dict = {"model": self._model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            payload["temperature"] = temperature

        response = requests.post(self._url, headers=self._headers(), json=payload, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        print(
            f"[mistral] completion | tokens="
            f"{usage.get('prompt_tokens', '?')}in/{usage.get('completion_tokens', '?')}out"
        )
        return data["choices"][0]["message"]["content"]

    def stream(self, messages: List[dict], max_tokens: int) -> Iterator[str]:
        self._require_key()
        payload = {"model": self._model, "messages": messages, "max_tokens": max_tokens, "stream": True}
        with requests.post(
            self._url,
            headers=self._headers(),
            json=payload,
            timeout=self._timeout,
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
                    payload_chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue  # skip keep-alive / malformed fragments
                delta = payload_chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content")
                if token:
                    yield token
