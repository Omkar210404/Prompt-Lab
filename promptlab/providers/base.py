"""Provider interface. The runner depends only on this module, never on a concrete provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from promptlab.config import ModelConfig


@dataclass(frozen=True)
class CompletionRequest:
    prompt: str
    system: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    # Test-case details (id, tags, expected answer). Only the mock provider reads them;
    # a real provider never sends them anywhere.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    text: str
    model: str
    latency_ms: float
    # {"prompt_tokens": int, "completion_tokens": int} when the provider reports it, else None.
    usage: dict[str, int] | None = None
    # "length" means the output was cut off at max_tokens.
    finish_reason: str | None = None


class Provider(ABC):
    """Every provider implements one async method. Failures must be raised as `ProviderError` subclasses."""

    type: ClassVar[str] = "base"

    def __init__(self, model: str):
        self.model = model

    @classmethod
    @abstractmethod
    def from_config(cls, config: ModelConfig) -> Provider:
        """Build the provider from a `models:` entry. Raise `ConfigError` for bad settings."""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        ...

    def cache_key_parts(self, request: CompletionRequest) -> dict[str, Any]:
        """Everything that decides this provider's answer to `request`. Hashed into the cache key."""
        return {
            "provider": self.type,
            "model": self.model,
            "system": request.system,
            "prompt": request.prompt,
            "params": request.params,
        }

    async def aclose(self) -> None:
        return None
