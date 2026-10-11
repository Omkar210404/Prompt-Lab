"""Deterministic, offline mock provider.

It does NOT generate text. It starts from the test case's expected answer and applies scripted
rules from the experiment file, so a run is repeatable and needs no key or network. Differences
between prompts or models in a mock run are therefore written by whoever wrote the rules. They
show how prompt-lab works, not how any real model behaves.

Rules are checked in order and the first match wins. A rule matches when all of its `when`
conditions hold:
- `prompt_matches` / `prompt_lacks`: regex (case-insensitive) searched in the prompt template and system prompt
- `input_matches`: regex searched in the test case's variable values
- `tags`: the case has any of these tags
- `case_ids`: the case id is one of these
- `rate`: a stable hash of the case id falls below this fraction (not random)

Behaviours:
- `correct`: the expected answer
- `reply`: the fixed text in `value`
- `wrong`: `value`, or a generic non-answer
- `wrong_field`: the expected object with the fields in `set` overwritten
- `drop_field`: the expected object without `field`
- `number_as_string`: the expected object with numeric `field` rendered through `format` ("$49.99")
- `prose_wrapped`: the correct answer between `prefix` and `suffix` chatter
- `fenced`: the correct answer inside a Markdown code fence
- `invalid_json`: the correct answer cut off part-way
- `change_case`: the correct answer in `to` case (upper, lower, title)
- `empty`: an empty string
- `error`: raises a provider error of `kind` (server, rate_limit, timeout, auth, bad_request),
  for the first `times` calls of each request, or always when `times` is not set
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from promptlab.config import ModelConfig
from promptlab.errors import (
    ConfigError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
    ProviderTimeoutError,
)
from promptlab.providers.base import CompletionRequest, CompletionResponse, Provider

Behaviour = Literal[
    "correct", "reply", "wrong", "wrong_field", "drop_field", "number_as_string",
    "prose_wrapped", "fenced", "invalid_json", "change_case", "empty", "error",
]

_ERRORS: dict[str, Callable[[], ProviderError]] = {
    "server": lambda: ProviderServerError("mock: simulated server error (HTTP 500)"),
    "rate_limit": lambda: ProviderRateLimitError("mock: simulated rate limit (HTTP 429)", retry_after=0),
    "timeout": lambda: ProviderTimeoutError("mock: simulated timeout"),
    "auth": lambda: ProviderAuthError("mock: simulated auth failure (HTTP 401)"),
    "bad_request": lambda: ProviderRequestError("mock: simulated bad request (HTTP 400)"),
}


class MockWhen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_matches: str | None = None
    prompt_lacks: str | None = None
    input_matches: str | None = None
    tags: list[str] = Field(default_factory=list)
    case_ids: list[str] = Field(default_factory=list)
    rate: float | None = Field(default=None, ge=0, le=1)

    @field_validator("prompt_matches", "prompt_lacks", "input_matches")
    @classmethod
    def _valid_regex(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"invalid regex '{value}': {exc}") from exc
        return value


class MockRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    when: MockWhen = Field(default_factory=MockWhen)
    behave: Behaviour
    value: Any = None
    set: dict[str, Any] = Field(default_factory=dict)
    field: str | None = None
    format: str = "${:.2f}"
    to: Literal["upper", "lower", "title"] = "upper"
    prefix: str = "Sure! Here is the information you asked for:\n\n"
    suffix: str = "\n\nLet me know if you need anything else."
    kind: Literal["server", "rate_limit", "timeout", "auth", "bad_request"] = "server"
    times: int | None = Field(default=None, ge=1)


class MockSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Behaviour when no rule matches.
    default: Behaviour = "correct"
    # Text used by a default of `reply`.
    value: Any = None
    # When the expected answer is an object, answer with this one field of it.
    answer_field: str | None = None
    # Simulated response time: a real sleep of latency_ms plus a stable per-prompt jitter.
    latency_ms: float = Field(default=0, ge=0)
    latency_jitter_ms: float = Field(default=0, ge=0)
    rules: list[MockRule] = Field(default_factory=list)


def _stable_fraction(*parts: str) -> float:
    """A number in [0, 1) that depends only on `parts`. Python's own hash() changes between runs."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x1_0000_0000


def _as_text(answer: Any) -> str:
    return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)


class MockProvider(Provider):
    type = "mock"

    def __init__(self, model: str = "mock-model", settings: MockSettings | dict | None = None,
                 responder: Callable[[CompletionRequest], str | None] | None = None):
        super().__init__(model)
        if not isinstance(settings, MockSettings):
            try:
                settings = MockSettings.model_validate(settings or {})
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
                raise ConfigError(f"mock settings for '{model}': {problems}") from exc
        self.settings = settings
        # Test hooks: `responder` overrides the rules when it returns a string; `calls` records every request.
        self._responder = responder
        self.calls: list[CompletionRequest] = []
        self._error_counts: dict[tuple[int, str], int] = {}

    @classmethod
    def from_config(cls, config: ModelConfig) -> MockProvider:
        return cls(model=config.model, settings=config.mock)

    def cache_key_parts(self, request: CompletionRequest) -> dict[str, Any]:
        # The mock's answer depends on its rules and on the case's expected answer, so both
        # belong in the cache key. Editing a rule must not return stale cached output.
        parts = super().cache_key_parts(request)
        parts["mock"] = self.settings.model_dump(mode="json")
        parts["metadata"] = request.metadata
        return parts

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        started = time.perf_counter()
        self.calls.append(request)

        delay_ms = self.settings.latency_ms
        if self.settings.latency_jitter_ms:
            delay_ms += self.settings.latency_jitter_ms * _stable_fraction("latency", request.prompt)
        if delay_ms:
            await asyncio.sleep(delay_ms / 1000)

        text = self._responder(request) if self._responder else None
        if text is None:
            text = self._apply_rules(request)

        return CompletionResponse(
            text=text,
            model=self.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            # Synthetic word counts, so the token and cost paths have something to work with.
            usage={
                "prompt_tokens": len(f"{request.system or ''} {request.prompt}".split()),
                "completion_tokens": len(text.split()),
            },
            finish_reason="stop",
        )

    def _apply_rules(self, request: CompletionRequest) -> str:
        for index, rule in enumerate(self.settings.rules):
            if not self._matches(rule, index, request):
                continue
            if rule.behave == "error":
                key = (index, request.prompt)
                seen = self._error_counts.get(key, 0)
                if rule.times is not None and seen >= rule.times:
                    continue  # this request has failed often enough; fall through to later rules
                self._error_counts[key] = seen + 1
                raise _ERRORS[rule.kind]()
            return self._behave(rule, request)
        return self._behave(MockRule(behave=self.settings.default, value=self.settings.value), request)

    @staticmethod
    def _matches(rule: MockRule, index: int, request: CompletionRequest) -> bool:
        when, meta = rule.when, request.metadata
        # Prompt conditions look at the template, so the test input cannot trigger them by accident.
        prompt_text = f"{request.system or ''}\n{meta.get('prompt_template', request.prompt)}"
        if when.prompt_matches and not re.search(when.prompt_matches, prompt_text, re.I):
            return False
        if when.prompt_lacks and re.search(when.prompt_lacks, prompt_text, re.I):
            return False
        if when.input_matches:
            values = meta.get("vars") or {"prompt": request.prompt}
            if not re.search(when.input_matches, "\n".join(_as_text(v) for v in values.values()), re.I):
                return False
        if when.tags and not set(when.tags) & set(meta.get("tags") or []):
            return False
        case_id = str(meta.get("case_id", request.prompt))
        if when.case_ids and case_id not in when.case_ids:
            return False
        if when.rate is not None and _stable_fraction(rule.name or f"rule-{index}", case_id) >= when.rate:
            return False
        return True

    def _answer(self, request: CompletionRequest) -> Any:
        expected = request.metadata.get("expected")
        if self.settings.answer_field and isinstance(expected, dict):
            expected = expected.get(self.settings.answer_field)
        if expected is None:
            raise ProviderRequestError(
                "mock: this behaviour needs the test case's expected answer, and the case has none"
            )
        return copy.deepcopy(expected)

    def _answer_object(self, rule: MockRule, request: CompletionRequest) -> dict:
        answer = self._answer(request)
        if not isinstance(answer, dict):
            raise ProviderRequestError(f"mock: '{rule.behave}' needs the expected answer to be an object")
        return answer

    def _behave(self, rule: MockRule, request: CompletionRequest) -> str:
        behave = rule.behave
        if behave == "empty":
            return ""
        if behave == "reply":
            return _as_text(rule.value if rule.value is not None else "")
        if behave == "wrong":
            return _as_text(rule.value) if rule.value is not None else "I'm not sure."
        if behave == "wrong_field":
            return _as_text({**self._answer_object(rule, request), **rule.set})
        if behave == "drop_field":
            answer = self._answer_object(rule, request)
            answer.pop(rule.field, None)
            return _as_text(answer)
        if behave == "number_as_string":
            answer = self._answer_object(rule, request)
            value = answer.get(rule.field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                answer[rule.field] = rule.format.format(value)
            return _as_text(answer)

        correct = _as_text(self._answer(request))
        if behave == "prose_wrapped":
            return f"{rule.prefix}{correct}{rule.suffix}"
        if behave == "fenced":
            return f"```json\n{correct}\n```"
        if behave == "invalid_json":
            return correct[: max(1, len(correct) * 2 // 3)]
        if behave == "change_case":
            return getattr(correct, rule.to)()
        return correct
