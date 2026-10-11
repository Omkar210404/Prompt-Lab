"""Scorer interface, the registry, and helpers shared by the built-in scorers."""

from __future__ import annotations

import json
import re
from abc import ABC
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from promptlab.config import ScorerConfig, TestCase
from promptlab.errors import ConfigError
from promptlab.providers.base import CompletionRequest, CompletionResponse


@dataclass
class ScoreResult:
    passed: bool
    # 0.0 to 1.0. Most scorers give 0 or 1; some give partial credit.
    score: float
    # One line a person can read in the report to see why.
    reason: str
    # True when the scorer could not reach a verdict (for example an unusable judge reply).
    # Such a result counts as a failure but is reported separately.
    error: bool = False


@dataclass
class ScoreContext:
    """What a scorer may need besides the output and the test case."""

    prompt: str
    system: str | None = None
    # Calls a judge model by id, through the runner's cache, retries and concurrency limits.
    complete: Callable[[str, CompletionRequest], Awaitable[CompletionResponse]] | None = None


class Scorer(ABC):
    """Subclass, set `type`, and implement `score`. Options arrive as keyword arguments to `__init__`."""

    type: ClassVar[str] = ""
    # Set by `build_scorer` from the scorer's config entry.
    name: str = ""
    required: bool = True

    def score(self, output: str, case: TestCase) -> ScoreResult:
        raise NotImplementedError

    async def ascore(self, output: str, case: TestCase, context: ScoreContext) -> ScoreResult:
        """Async entry point used by the runner. Override it only if the scorer has to await something."""
        return self.score(output, case)


_SCORERS: dict[str, type[Scorer]] = {}


def register_scorer(cls: type[Scorer]) -> type[Scorer]:
    """Class decorator: make a scorer available as `type: <cls.type>` in experiment files."""
    if not cls.type:
        raise ValueError(f"{cls.__name__} must set a `type`")
    _SCORERS[cls.type] = cls
    return cls


def scorer_types() -> dict[str, str]:
    """Registered scorer types with the first line of their docstring."""
    return {
        name: (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
        for name, cls in sorted(_SCORERS.items())
    }


def build_scorer(config: ScorerConfig, base_dir: Path) -> Scorer:
    try:
        cls = _SCORERS[config.type]
    except KeyError:
        raise ConfigError(
            f"unknown scorer type '{config.type}' (available: {', '.join(sorted(_SCORERS))})"
        ) from None
    options = config.options
    for key, value in options.items():
        # Options ending in `_file` are paths relative to the experiment file.
        if key.endswith("_file") and isinstance(value, str):
            options[key] = str(base_dir / value)
    try:
        scorer = cls(**options)
    except TypeError as exc:
        message = str(exc).split("__init__() ")[-1]
        raise ConfigError(f"scorer '{config.type}': {message}") from exc
    except ValueError as exc:
        raise ConfigError(f"scorer '{config.type}': {exc}") from exc
    scorer.name = config.name or config.type
    scorer.required = config.required
    return scorer


# --- helpers ---------------------------------------------------------------------------------

MISSING: Any = object()

_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def passed(reason: str) -> ScoreResult:
    return ScoreResult(passed=True, score=1.0, reason=reason)


def failed(reason: str) -> ScoreResult:
    return ScoreResult(passed=False, score=0.0, reason=reason)


def parse_json(text: str, allow_fenced: bool = False) -> tuple[Any, str | None]:
    """Parse the whole output as JSON. Returns (value, None) or (MISSING, why it failed).

    Strict on purpose: JSON with prose around it is not valid JSON to the program that has to
    read it. `allow_fenced` accepts one Markdown code fence around the JSON and nothing else.
    """
    candidate = text.strip()
    if not candidate:
        return MISSING, "output is empty"
    if allow_fenced:
        match = _FENCE.match(candidate)
        if match:
            candidate = match.group(1).strip()
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return MISSING, f"not valid JSON ({exc.msg} at character {exc.pos})"


def get_field(value: Any, path: str) -> Any:
    """Follow a dotted path ("customer.id") through nested objects. Returns MISSING if absent."""
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return MISSING
        value = value[part]
    return value


def expected_for(case: TestCase, expected_field: str | None = None, field: str | None = None) -> Any:
    """The reference value a scorer should compare against.

    `expected_field` picks one field of an object-valued `expected`. Otherwise, when the scorer
    looks at one `field` of the output, the same field of `expected` is used.
    """
    expected = case.expected
    path = expected_field or field
    if path and isinstance(expected, dict):
        value = get_field(expected, path)
        return None if value is MISSING else value
    return expected


def as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def output_target(output: str, field: str | None, allow_fenced: bool = False) -> tuple[Any, ScoreResult | None]:
    """The thing to score: the raw output, or one field of it when `field` is set.

    Returns (value, None), or (MISSING, a failing result) when the field cannot be read.
    """
    if not field:
        return output, None
    data, problem = parse_json(output, allow_fenced)
    if problem:
        # json_valid, if configured, reports the parse error in full; here a short reason is enough.
        return MISSING, failed(f"cannot read field '{field}': output is not valid JSON")
    value = get_field(data, field)
    if value is MISSING:
        return MISSING, failed(f"field '{field}' is missing from the output")
    return value, None


def shorten(value: Any, limit: int = 80) -> str:
    """repr() of a value, cut to a length that fits in a reason line."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
