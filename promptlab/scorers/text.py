"""Text scorers: exact_match, contains, not_contains, regex, length."""

from __future__ import annotations

import re

from promptlab.config import TestCase
from promptlab.scorers.base import (
    ScoreResult,
    Scorer,
    as_text,
    expected_for,
    failed,
    output_target,
    passed,
    register_scorer,
    shorten,
)


def _needles(value, values) -> list[str]:
    if value is not None and values is not None:
        raise ValueError("give either `value` or `values`, not both")
    if values is not None:
        if not isinstance(values, list) or not values:
            raise ValueError("`values` must be a non-empty list")
        return [as_text(v) for v in values]
    return [as_text(value)] if value is not None else []


@register_scorer
class ExactMatch(Scorer):
    """Output equals the expected value (after optional case and whitespace normalisation)."""

    type = "exact_match"

    def __init__(self, value=None, expected_field: str | None = None, field: str | None = None,
                 ignore_case: bool = False, strip: bool = True, normalize_whitespace: bool = False,
                 allow_fenced: bool = False):
        self.value = value
        self.expected_field = expected_field
        self.field = field
        self.ignore_case = ignore_case
        self.strip = strip
        self.normalize_whitespace = normalize_whitespace
        self.allow_fenced = allow_fenced

    def _normalise(self, text: str) -> str:
        if self.normalize_whitespace:
            text = " ".join(text.split())
        elif self.strip:
            text = text.strip()
        return text.casefold() if self.ignore_case else text

    def score(self, output: str, case: TestCase) -> ScoreResult:
        expected = self.value if self.value is not None else expected_for(case, self.expected_field, self.field)
        if expected is None:
            return failed("no expected value for this case")
        actual, problem = output_target(output, self.field, self.allow_fenced)
        if problem:
            return problem
        if self._normalise(as_text(actual)) == self._normalise(as_text(expected)):
            return passed("matches expected")
        label = f"{self.field}: " if self.field else ""
        return failed(f"{label}expected {shorten(expected)}, got {shorten(actual)}")


@register_scorer
class Contains(Scorer):
    """Output contains the given text (all of `values`, or any with mode: any)."""

    type = "contains"

    def __init__(self, value=None, values=None, mode: str = "all", ignore_case: bool = False,
                 field: str | None = None):
        if mode not in ("all", "any"):
            raise ValueError("`mode` must be 'all' or 'any'")
        self.needles = _needles(value, values)
        self.mode = mode
        self.ignore_case = ignore_case
        self.field = field

    def score(self, output: str, case: TestCase) -> ScoreResult:
        needles = self.needles
        if not needles:
            expected = expected_for(case, None, self.field)
            if expected is None:
                return failed("no `value` given and no expected value for this case")
            needles = [as_text(expected)]
        actual, problem = output_target(output, self.field)
        if problem:
            return problem
        haystack = as_text(actual).casefold() if self.ignore_case else as_text(actual)
        found = [n for n in needles if (n.casefold() if self.ignore_case else n) in haystack]
        missing = [n for n in needles if n not in found]
        if self.mode == "any":
            if found:
                return passed(f"contains {shorten(found[0])}")
            return failed(f"contains none of {shorten(needles)}")
        if not missing:
            return passed("contains all required text")
        return ScoreResult(False, len(found) / len(needles), f"missing {shorten(missing)}")


@register_scorer
class NotContains(Scorer):
    """Output contains none of the given text."""

    type = "not_contains"

    def __init__(self, value=None, values=None, ignore_case: bool = False, field: str | None = None):
        self.needles = _needles(value, values)
        if not self.needles:
            raise ValueError("needs `value` or `values`")
        self.ignore_case = ignore_case
        self.field = field

    def score(self, output: str, case: TestCase) -> ScoreResult:
        actual, problem = output_target(output, self.field)
        if problem:
            return problem
        haystack = as_text(actual).casefold() if self.ignore_case else as_text(actual)
        present = [n for n in self.needles if (n.casefold() if self.ignore_case else n) in haystack]
        if present:
            return failed(f"contains forbidden text {shorten(present)}")
        return passed("no forbidden text")


@register_scorer
class Regex(Scorer):
    """Output matches a regular expression (mode: search, match or fullmatch)."""

    type = "regex"

    def __init__(self, pattern: str, mode: str = "search", ignore_case: bool = False,
                 dotall: bool = False, strip: bool = True, field: str | None = None):
        if mode not in ("search", "match", "fullmatch"):
            raise ValueError("`mode` must be 'search', 'match' or 'fullmatch'")
        flags = (re.IGNORECASE if ignore_case else 0) | (re.DOTALL if dotall else 0)
        try:
            self.regex = re.compile(pattern, flags)
        except (re.error, TypeError) as exc:
            raise ValueError(f"invalid pattern {pattern!r}: {exc}") from exc
        self.mode = mode
        self.strip = strip
        self.field = field

    def score(self, output: str, case: TestCase) -> ScoreResult:
        actual, problem = output_target(output, self.field)
        if problem:
            return problem
        text = as_text(actual).strip() if self.strip else as_text(actual)
        if getattr(self.regex, self.mode)(text):
            return passed(f"matches /{self.regex.pattern}/")
        return failed(f"does not {self.mode} /{self.regex.pattern}/: {shorten(text, 60)}")


@register_scorer
class Length(Scorer):
    """Output length is within character and/or word limits."""

    type = "length"

    def __init__(self, min_chars: int | None = None, max_chars: int | None = None,
                 min_words: int | None = None, max_words: int | None = None):
        limits = (min_chars, max_chars, min_words, max_words)
        if all(limit is None for limit in limits):
            raise ValueError("needs at least one of min_chars, max_chars, min_words, max_words")
        if any(limit is not None and (not isinstance(limit, int) or limit < 0) for limit in limits):
            raise ValueError("limits must be non-negative integers")
        self.min_chars, self.max_chars = min_chars, max_chars
        self.min_words, self.max_words = min_words, max_words

    def score(self, output: str, case: TestCase) -> ScoreResult:
        text = output.strip()
        chars, words = len(text), len(text.split())
        problems = []
        if self.min_chars is not None and chars < self.min_chars:
            problems.append(f"{chars} chars, minimum {self.min_chars}")
        if self.max_chars is not None and chars > self.max_chars:
            problems.append(f"{chars} chars, maximum {self.max_chars}")
        if self.min_words is not None and words < self.min_words:
            problems.append(f"{words} words, minimum {self.min_words}")
        if self.max_words is not None and words > self.max_words:
            problems.append(f"{words} words, maximum {self.max_words}")
        if problems:
            return failed("; ".join(problems))
        return passed(f"{chars} chars, {words} words")
