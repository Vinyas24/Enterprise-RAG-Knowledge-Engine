"""OpenAI-compatible chat completion provider."""

from __future__ import annotations

import httpx

from .base import LLMProvider
from .errors import LLMResponseError, LLMUnavailable


class OpenAILLMProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("LLM API key is required")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    async def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.post(
                "chat/completions",
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                },
            )
        except httpx.TimeoutException as exc:
            raise LLMUnavailable("LLM request timed out") from exc
        except httpx.RequestError as exc:
            raise LLMUnavailable(f"LLM request failed: {exc}") from exc

        if response.status_code >= 500 or response.status_code == 429:
            raise LLMUnavailable(
                f"LLM provider error ({response.status_code}): {response.text[:200]}"
            )
        if response.status_code >= 400:
            raise LLMResponseError(
                f"LLM provider rejected request ({response.status_code}): {response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError("LLM provider returned non-JSON response") from exc

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise LLMResponseError("LLM response missing choices")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("LLM response choice missing message")

        content = message.get("content")
        if not isinstance(content, str):
            raise LLMResponseError("LLM response choice missing content")

        return content.strip()

    async def close(self) -> None:
        await self._client.aclose()
