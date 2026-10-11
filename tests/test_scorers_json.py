import json

import pytest

from promptlab.errors import ConfigError
from tests.conftest import make_case, make_scorer

TICKET_SCHEMA = {
    "type": "object",
    "required": ["order_id", "category", "refund_amount"],
    "additionalProperties": False,
    "properties": {
        "order_id": {"type": ["string", "null"]},
        "category": {"enum": ["refund", "shipping", "billing"]},
        "refund_amount": {"type": ["number", "null"]},
    },
}
GOOD = {"order_id": "ORD-1", "category": "refund", "refund_amount": 49.99}


# --- json_valid ---

def test_json_valid_accepts_json():
    assert make_scorer("json_valid").score(json.dumps(GOOD), make_case()).passed
    assert make_scorer("json_valid").score("  [1, 2]\n", make_case()).passed


@pytest.mark.parametrize("output, reason", [
    ("", "output is empty"),
    ("Sure! Here it is: {\"a\": 1}", "not valid JSON"),
    ('{"a": 1', "not valid JSON"),
    ('```json\n{"a": 1}\n```', "not valid JSON"),
])
def test_json_valid_is_strict_by_default(output, reason):
    result = make_scorer("json_valid").score(output, make_case())
    assert not result.passed and reason in result.reason


def test_json_valid_allow_fenced():
    scorer = make_scorer("json_valid", allow_fenced=True)
    assert scorer.score('```json\n{"a": 1}\n```', make_case()).passed
    assert scorer.score('```\n{"a": 1}\n```', make_case()).passed
    assert not scorer.score('Here:\n```json\n{"a": 1}\n```', make_case()).passed  # prose is still prose


def test_json_valid_require_object():
    result = make_scorer("json_valid", require_object=True).score("[1]", make_case())
    assert not result.passed and "not an object" in result.reason


# --- json_schema ---

def test_json_schema_pass():
    assert make_scorer("json_schema", schema=TICKET_SCHEMA).score(json.dumps(GOOD), make_case()).passed


def test_json_schema_lists_violations_with_paths():
    bad = {"order_id": 42, "category": "complaint"}
    result = make_scorer("json_schema", schema=TICKET_SCHEMA).score(json.dumps(bad), make_case())
    assert not result.passed
    assert "'refund_amount' is a required property" in result.reason
    assert "category: 'complaint' is not one of" in result.reason
    assert "order_id: 42 is not of type" in result.reason


def test_json_schema_caps_the_violation_list():
    schema = {"type": "object", "required": ["a", "b", "c", "d", "e"]}
    result = make_scorer("json_schema", schema=schema).score("{}", make_case())
    assert result.reason.endswith("(+2 more)")


def test_json_schema_on_invalid_json():
    result = make_scorer("json_schema", schema=TICKET_SCHEMA).score("not json", make_case())
    assert not result.passed and result.reason.startswith("cannot validate: output is not valid JSON")


def test_json_schema_from_file(tmp_path):
    (tmp_path / "ticket.schema.json").write_text(json.dumps(TICKET_SCHEMA), encoding="utf-8")
    scorer = make_scorer("json_schema", base_dir=tmp_path, schema_file="ticket.schema.json")
    assert scorer.score(json.dumps(GOOD), make_case()).passed
    assert not scorer.score('{"order_id": "x"}', make_case()).passed


def test_json_schema_option_errors(tmp_path):
    with pytest.raises(ConfigError, match="exactly one of"):
        make_scorer("json_schema")
    with pytest.raises(ConfigError, match="exactly one of"):
        make_scorer("json_schema", schema={}, schema_file="x.json")
    with pytest.raises(ConfigError, match="schema file not found"):
        make_scorer("json_schema", base_dir=tmp_path, schema_file="missing.json")
    with pytest.raises(ConfigError, match="invalid JSON Schema"):
        make_scorer("json_schema", schema={"type": "not-a-type"})


# --- json_fields ---

def test_json_fields_all_match():
    result = make_scorer("json_fields").score(json.dumps(GOOD), make_case(GOOD))
    assert result.passed and result.reason == "all 3 fields match"


def test_json_fields_gives_partial_credit_and_names_each_mismatch():
    output = {"order_id": "ORD-1", "category": "billing"}
    result = make_scorer("json_fields").score(json.dumps(output), make_case(GOOD))
    assert not result.passed
    assert result.score == pytest.approx(1 / 3)
    assert "category: expected 'refund', got 'billing'" in result.reason
    assert "refund_amount: missing" in result.reason


def test_json_fields_only_checks_listed_fields():
    output = {"order_id": "ORD-1", "category": "billing", "refund_amount": 0}
    assert make_scorer("json_fields", fields=["order_id"]).score(json.dumps(output), make_case(GOOD)).passed


def test_json_fields_type_strictness():
    scorer = make_scorer("json_fields")
    assert not scorer.score('{"n": "49.99"}', make_case({"n": 49.99})).passed  # string is not a number
    assert not scorer.score('{"n": true}', make_case({"n": 1})).passed        # true is not 1
    assert scorer.score('{"n": 49}', make_case({"n": 49.0})).passed
    assert scorer.score('{"n": null}', make_case({"n": None})).passed
    assert not scorer.score('{"n": "UNKNOWN"}', make_case({"n": None})).passed


def test_json_fields_ignore_case():
    assert not make_scorer("json_fields").score('{"s": "Negative"}', make_case({"s": "negative"})).passed
    assert make_scorer("json_fields", ignore_case=True).score('{"s": "Negative"}', make_case({"s": "negative"})).passed


def test_json_fields_bad_inputs():
    scorer = make_scorer("json_fields")
    assert "not valid JSON" in scorer.score("Sure! {}", make_case(GOOD)).reason
    assert "not an object" in scorer.score("[1]", make_case(GOOD)).reason
    assert "expected value is not an object" in scorer.score("{}", make_case("text")).reason
    with pytest.raises(ConfigError, match="non-empty list"):
        make_scorer("json_fields", fields=[])
