import pytest

from promptlab.errors import ConfigError
from tests.conftest import make_case, make_scorer


# --- exact_match ---

def test_exact_match_passes_and_strips_by_default():
    result = make_scorer("exact_match").score("  positive\n", make_case("positive"))
    assert result.passed and result.score == 1.0


def test_exact_match_fails_with_a_readable_reason():
    result = make_scorer("exact_match").score("Positive", make_case("positive"))
    assert not result.passed and result.score == 0.0
    assert result.reason == "expected 'positive', got 'Positive'"


def test_exact_match_options():
    assert make_scorer("exact_match", ignore_case=True).score("POSITIVE", make_case("positive")).passed
    assert not make_scorer("exact_match", strip=False).score("positive ", make_case("positive")).passed
    assert make_scorer("exact_match", normalize_whitespace=True).score("a  b\n c", make_case("a b c")).passed


def test_exact_match_explicit_value_and_expected_field():
    assert make_scorer("exact_match", value="yes").score("yes", make_case("no")).passed
    case = make_case({"sentiment": "negative", "order_id": "ORD-1"})
    assert make_scorer("exact_match", expected_field="sentiment").score("negative", case).passed
    assert not make_scorer("exact_match", expected_field="sentiment").score("ORD-1", case).passed


def test_exact_match_on_a_json_field():
    scorer = make_scorer("exact_match", field="category")
    case = make_case({"category": "refund"})
    assert scorer.score('{"category": "refund"}', case).passed
    assert scorer.score('{"category": "billing"}', case).reason == "category: expected 'refund', got 'billing'"
    assert "is missing" in scorer.score('{"other": 1}', case).reason
    assert "not valid JSON" in scorer.score("refund", case).reason


def test_exact_match_without_expected_fails():
    assert make_scorer("exact_match").score("x", make_case(None)).reason == "no expected value for this case"


# --- contains / not_contains ---

def test_contains_all_and_any():
    case = make_case()
    all_of = make_scorer("contains", values=["refund", "ORD-1"])
    assert all_of.score("refund for ORD-1", case).passed
    partial = all_of.score("refund only", case)
    assert not partial.passed and partial.score == 0.5 and "ORD-1" in partial.reason

    any_of = make_scorer("contains", values=["refund", "return"], mode="any")
    assert any_of.score("please return it", case).passed
    assert not any_of.score("hello", case).passed


def test_contains_ignore_case_and_expected_default():
    assert make_scorer("contains", value="REFUND", ignore_case=True).score("a refund", make_case()).passed
    assert not make_scorer("contains", value="REFUND").score("a refund", make_case()).passed
    assert make_scorer("contains").score("it is positive overall", make_case("positive")).passed
    assert not make_scorer("contains").score("anything", make_case(None)).passed


def test_not_contains():
    scorer = make_scorer("not_contains", values=["Sure", "As an AI"])
    assert scorer.score('{"ok": true}', make_case()).passed
    result = scorer.score("Sure! Here you go", make_case())
    assert not result.passed and "Sure" in result.reason
    assert not make_scorer("not_contains", value="sure", ignore_case=True).score("SURE", make_case()).passed


def test_contains_option_errors():
    with pytest.raises(ConfigError, match="either `value` or `values`"):
        make_scorer("contains", value="a", values=["b"])
    with pytest.raises(ConfigError, match="needs `value` or `values`"):
        make_scorer("not_contains")
    with pytest.raises(ConfigError, match="`mode` must be"):
        make_scorer("contains", value="a", mode="most")


# --- regex ---

def test_regex_modes():
    case = make_case()
    assert make_scorer("regex", pattern=r"ORD-\d+").score("order ORD-42 here", case).passed
    assert not make_scorer("regex", pattern=r"ORD-\d+", mode="fullmatch").score("order ORD-42", case).passed
    assert make_scorer("regex", pattern=r"ORD-\d+", mode="fullmatch").score(" ORD-42\n", case).passed
    assert make_scorer("regex", pattern="^(positive|negative)$", ignore_case=True).score("Negative", case).passed


def test_regex_failure_reason_and_field():
    result = make_scorer("regex", pattern="^(positive|negative)$").score("The sentiment is negative.", make_case())
    assert not result.passed and "does not search" in result.reason
    assert make_scorer("regex", pattern=r"^ORD-\d{5}$", field="order_id").score('{"order_id": "ORD-48213"}', make_case()).passed


def test_regex_bad_options():
    with pytest.raises(ConfigError, match="invalid pattern"):
        make_scorer("regex", pattern="(unclosed")
    with pytest.raises(ConfigError, match="pattern"):
        make_scorer("regex")
    with pytest.raises(ConfigError, match="`mode` must be"):
        make_scorer("regex", pattern="a", mode="find")


# --- length ---

def test_length_limits():
    case = make_case()
    assert make_scorer("length", max_words=1).score("negative", case).passed
    too_long = make_scorer("length", max_words=1).score("it is negative", case)
    assert not too_long.passed and too_long.reason == "3 words, maximum 1"
    assert not make_scorer("length", min_chars=5).score("hey", case).passed
    assert make_scorer("length", min_chars=2, max_chars=10, min_words=1, max_words=3).score("two words", case).passed
    assert not make_scorer("length", max_chars=3).score("abcd", case).passed
    assert not make_scorer("length", min_words=2).score("", case).passed


def test_length_needs_a_limit():
    with pytest.raises(ConfigError, match="at least one of"):
        make_scorer("length")
    with pytest.raises(ConfigError, match="non-negative integers"):
        make_scorer("length", max_words=-1)


# --- common to every scorer ---

def test_name_and_required_come_from_config():
    scorer = make_scorer("length", max_words=5, name="short answer", required=False)
    assert scorer.name == "short answer" and scorer.required is False
    assert make_scorer("exact_match").name == "exact_match" and make_scorer("exact_match").required is True
