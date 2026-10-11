"""Provider registry and the retry loop. Switching provider is a change in the experiment file, not in code."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from promptlab.config import ModelConfig
from promptlab.errors import ConfigError, ProviderError, ProviderTimeoutError
from promptlab.providers.base import CompletionRequest, CompletionResponse, Provider
from promptlab.providers.mock import MockProvider
from promptlab.providers.openai_compatible import OpenAICompatibleProvider

_PROVIDERS: dict[str, type[Provider]] = {}

MAX_RETRY_DELAY_SECONDS = 30.0


def register_provider(cls: type[Provider]) -> type[Provider]:
    """Make a provider available as `provider: <cls.type>` in experiment files."""
    _PROVIDERS[cls.type] = cls
    return cls


register_provider(MockProvider)
register_provider(OpenAICompatibleProvider)


def provider_types() -> list[str]:
    return sorted(_PROVIDERS)


def build_provider(config: ModelConfig) -> Provider:
    try:
        cls = _PROVIDERS[config.provider]
    except KeyError:
        raise ConfigError(
            f"model '{config.id}': unknown provider '{config.provider}' (available: {', '.join(provider_types())})"
        ) from None
    return cls.from_config(config)


async def complete_with_retry(
    provider: Provider,
    request: CompletionRequest,
    *,
    timeout_seconds: float,
    max_retries: int = 2,
    base_delay: float = 1.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[CompletionResponse, int]:
    """Call the provider with a per-attempt timeout. Returns the response and the number of attempts.

    Retryable errors (timeouts, rate limits, 5xx) are retried with exponential backoff, waiting at
    least as long as the provider's Retry-After. Other errors are raised at once.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            response = await asyncio.wait_for(provider.complete(request), timeout=timeout_seconds)
            return response, attempt
        except TimeoutError:
            error: ProviderError = ProviderTimeoutError(f"no response within {timeout_seconds:g}s")
        except ProviderError as exc:
            error = exc
        if not error.retryable or attempt > max_retries:
            error.attempts = attempt
            raise error
        delay = base_delay * (2 ** (attempt - 1))
        retry_after = getattr(error, "retry_after", None)
        if retry_after is not None:
            delay = max(delay, retry_after)  # the provider says when capacity is back
        await sleep(min(delay, MAX_RETRY_DELAY_SECONDS))
