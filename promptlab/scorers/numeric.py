"""Numeric scorer: a number within tolerance of the expected one."""

from __future__ import annotations

import re
from typing import Any

from promptlab.config import TestCase
from promptlab.scorers.base import (
    ScoreResult,
    Scorer,
    expected_for,
    failed,
    output_target,
    passed,
    register_scorer,
    shorten,
)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_number(value: Any, coerce: bool) -> float | None:
    """A number from a JSON value or from text. `coerce` also accepts "$1,249.50"-style strings."""
    if _is_number(value):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        pass
    if coerce:
        match = _NUMBER.search(text.replace(",", ""))
        if match:
            return float(match.group())
    return None


@register_scorer
class Numeric(Scorer):
    """A number in the output is within abs_tol / rel_tol of the expected number."""

    type = "numeric"

    def __init__(self, value: float | None = None, expected_field: str | None = None,
                 field: str | None = None, abs_tol: float = 0.0, rel_tol: float = 0.0,
                 coerce: bool = False, allow_fenced: bool = False):
        if value is not None and not _is_number(value):
            raise ValueError("`value` must be a number")
        if not _is_number(abs_tol) or not _is_number(rel_tol) or abs_tol < 0 or rel_tol < 0:
            raise ValueError("`abs_tol` and `rel_tol` must be non-negative numbers")
        self.value = value
        self.expected_field = expected_field
        self.field = field
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.coerce = coerce
        self.allow_fenced = allow_fenced

    def score(self, output: str, case: TestCase) -> ScoreResult:
        expected = self.value if self.value is not None else expected_for(case, self.expected_field, self.field)
        actual, problem = output_target(output, self.field, self.allow_fenced)
        if problem:
            return problem
        label = f"{self.field}: " if self.field else ""

        if expected is None:
            # "No number" is a legitimate expected answer for a field, such as an email with no amount.
            if self.field and actual is None:
                return passed(f"{label}null, as expected")
            if self.field:
                return failed(f"{label}expected null, got {shorten(actual)}")
            return failed("no expected number for this case")
        if not _is_number(expected):
            return failed(f"{label}expected value {shorten(expected)} is not a number")

        # A JSON field must already be a number unless `coerce` is on: "$49.99" is not 49.99
        # to the program that reads it.
        if self.field and not self.coerce:
            number = float(actual) if _is_number(actual) else None
        else:
            number = _to_number(actual, self.coerce)
        if number is None:
            return failed(f"{label}expected {expected}, got {shorten(actual)} (not a number)")

        allowed = max(self.abs_tol, self.rel_tol * abs(expected))
        difference = abs(number - expected)
        if difference <= allowed + 1e-12:
            return passed(f"{label}{number:g} is within {allowed:g} of {expected:g}")
        return failed(f"{label}expected {expected:g}, got {number:g} (off by {difference:g}, allowed {allowed:g})")
