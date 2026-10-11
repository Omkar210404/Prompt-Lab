"""JSON scorers: json_valid, json_schema, json_fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from promptlab.config import TestCase
from promptlab.scorers.base import (
    MISSING,
    ScoreResult,
    Scorer,
    failed,
    get_field,
    parse_json,
    passed,
    register_scorer,
    shorten,
)


def values_equal(actual: Any, expected: Any, ignore_case: bool = False) -> bool:
    """Equality for JSON values. True is not 1, and "49.99" is not 49.99."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual == expected
    if isinstance(actual, str) and isinstance(expected, str) and ignore_case:
        return actual.strip().casefold() == expected.strip().casefold()
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected
    return type(actual) is type(expected) and actual == expected


@register_scorer
class JsonValid(Scorer):
    """The whole output parses as JSON. Prose around the JSON fails unless allow_fenced is set."""

    type = "json_valid"

    def __init__(self, allow_fenced: bool = False, require_object: bool = False):
        self.allow_fenced = allow_fenced
        self.require_object = require_object

    def score(self, output: str, case: TestCase) -> ScoreResult:
        data, problem = parse_json(output, self.allow_fenced)
        if problem:
            return failed(problem)
        if self.require_object and not isinstance(data, dict):
            return failed(f"valid JSON but not an object (got {type(data).__name__})")
        return passed("valid JSON")


@register_scorer
class JsonSchema(Scorer):
    """The output validates against a JSON Schema (inline `schema` or a `schema_file`)."""

    type = "json_schema"

    def __init__(self, schema: dict | None = None, schema_file: str | None = None,
                 allow_fenced: bool = False):
        if (schema is None) == (schema_file is None):
            raise ValueError("give exactly one of `schema` or `schema_file`")
        if schema_file is not None:
            path = Path(schema_file)
            if not path.is_file():
                raise ValueError(f"schema file not found: {path}")
            try:
                schema = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} is not valid JSON ({exc.msg})") from exc
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError(f"invalid JSON Schema: {exc.message}") from exc
        self.validator = Draft202012Validator(schema)
        self.allow_fenced = allow_fenced

    def score(self, output: str, case: TestCase) -> ScoreResult:
        data, problem = parse_json(output, self.allow_fenced)
        if problem:
            return failed(f"cannot validate: output is {problem}")
        errors = sorted(self.validator.iter_errors(data), key=lambda e: list(e.absolute_path))
        if not errors:
            return passed("matches schema")
        shown = [
            f"{'.'.join(str(p) for p in e.absolute_path) or '(root)'}: {e.message}" for e in errors[:3]
        ]
        more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
        return failed("; ".join(shown) + more)


@register_scorer
class JsonFields(Scorer):
    """Listed fields of the JSON output equal the expected object; score is the fraction that match."""

    type = "json_fields"

    def __init__(self, fields: list[str] | None = None, ignore_case: bool = False,
                 allow_fenced: bool = False):
        if fields is not None and (not isinstance(fields, list) or not fields):
            raise ValueError("`fields` must be a non-empty list")
        self.fields = fields
        self.ignore_case = ignore_case
        self.allow_fenced = allow_fenced

    def score(self, output: str, case: TestCase) -> ScoreResult:
        if not isinstance(case.expected, dict):
            return failed("this case's expected value is not an object")
        data, problem = parse_json(output, self.allow_fenced)
        if problem:
            return failed(f"cannot compare fields: output is {problem}")
        if not isinstance(data, dict):
            return failed(f"output is JSON but not an object (got {type(data).__name__})")

        fields = self.fields or list(case.expected)
        mismatches = []
        for name in fields:
            expected = get_field(case.expected, name)
            actual = get_field(data, name)
            if expected is MISSING:
                mismatches.append(f"{name}: not in this case's expected value")
            elif actual is MISSING:
                mismatches.append(f"{name}: missing")
            elif not values_equal(actual, expected, self.ignore_case):
                mismatches.append(f"{name}: expected {shorten(expected, 40)}, got {shorten(actual, 40)}")
        if not mismatches:
            return passed(f"all {len(fields)} fields match")
        return ScoreResult(False, (len(fields) - len(mismatches)) / len(fields), "; ".join(mismatches))
