"""Provider for any OpenAI-compatible Chat Completions API (OpenAI, Groq, OpenRouter, Together, Ollama, ...).

Uses plain httpx, no vendor SDK. The API key is read from the environment variable named by
`api_key_env` in the experiment file; the key is never written to disk by prompt-lab.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from promptlab.config import ModelConfig
from promptlab.errors import (
    ConfigError,
    MalformedResponseError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
    ProviderTimeoutError,
)
from promptlab.providers.base import CompletionRequest, CompletionResponse, Provider


class OpenAICompatibleProvider(Provider):
    type = "openai_compatible"

    def __init__(self, model: str, base_url: str, api_key: str | None = None,
                 timeout_seconds: float = 60.0, transport: httpx.AsyncBaseTransport | None = None):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        # Local servers such as Ollama need no key, so the header is optional.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout_seconds, headers=headers, transport=transport
        )

    @classmethod
    def from_config(cls, config: ModelConfig) -> OpenAICompatibleProvider:
        if not config.base_url:
            raise ConfigError(f"model '{config.id}': openai_compatible needs a base_url")
        api_key = None
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env, "").strip()
            if not api_key:
                raise ConfigError(
                    f"model '{config.id}': environment variable {config.api_key_env} is not set. "
                    "Set it in your shell or in a .env file (see .env.example)."
                )
        return cls(model=config.model, base_url=config.base_url, api_key=api_key)

    def cache_key_parts(self, request: CompletionRequest) -> dict[str, Any]:
        parts = super().cache_key_parts(request)
        parts["base_url"] = self.base_url
        return parts

    def _build_payload(self, request: CompletionRequest) -> dict[str, Any]:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        return {**request.params, "model": self.model, "messages": messages}

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        started = time.perf_counter()
        try:
            response = await self._client.post("/chat/completions", json=self._build_payload(request))
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"provider timeout: {type(exc).__name__}") from exc
        except httpx.HTTPError as exc:
            raise ProviderServerError(f"provider connection error: {type(exc).__name__}") from exc

        status = response.status_code
        if status >= 400:
            detail = f"provider returned HTTP {status}: {self._error_message(response)}"
            if status in (401, 403):
                raise ProviderAuthError(detail)
            if status == 429:
                raise ProviderRateLimitError(detail, retry_after=self._retry_after(response))
            if status >= 500:
                raise ProviderServerError(detail)
            raise ProviderRequestError(detail)

        try:
            body = response.json()
            choice = body["choices"][0]
            text = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise MalformedResponseError("provider response missing choices[0].message.content") from exc
        if text is not None and not isinstance(text, str):
            raise MalformedResponseError("provider returned non-text content")

        return CompletionResponse(
            # An empty answer is still an answer: it is scored (and fails) like any other output.
            text=text or "",
            model=body.get("model") or self.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            usage=self._usage(body.get("usage")),
            finish_reason=choice.get("finish_reason"),
        )

    @staticmethod
    def _usage(usage: Any) -> dict[str, int] | None:
        """Token counts exactly as reported. Missing or malformed usage stays None, never zero."""
        if not isinstance(usage, dict):
            return None
        counts = {key: usage.get(key) for key in ("prompt_tokens", "completion_tokens")}
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in counts.values()):
            return None
        return counts

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        try:
            return float(response.headers["retry-after"])
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        """The provider's own error text, truncated. Providers do not echo API keys in error bodies."""
        try:
            error = response.json().get("error")
            message = error.get("message") if isinstance(error, dict) else str(error)
        except (ValueError, AttributeError):
            message = response.text
        return (message or "no error message")[:200]

    async def aclose(self) -> None:
        await self._client.aclose()
