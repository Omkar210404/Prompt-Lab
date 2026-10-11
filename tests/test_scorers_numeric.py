import pytest

from promptlab.errors import ConfigError
from tests.conftest import make_case, make_scorer


def test_exact_number_with_zero_tolerance():
    assert make_scorer("numeric").score("42", make_case(42)).passed
    assert make_scorer("numeric").score(" 42.0\n", make_case(42)).passed
    assert not make_scorer("numeric").score("42.01", make_case(42)).passed


def test_absolute_tolerance():
    scorer = make_scorer("numeric", abs_tol=0.01)
    assert scorer.score("49.995", make_case(49.99)).passed
    result = scorer.score("50.5", make_case(49.99))
    assert not result.passed and "off by 0.51" in result.reason


def test_relative_tolerance():
    scorer = make_scorer("numeric", rel_tol=0.05)
    assert scorer.score("104", make_case(100)).passed
    assert not scorer.score("106", make_case(100)).passed
    # the larger of the two tolerances applies
    assert make_scorer("numeric", rel_tol=0.01, abs_tol=10).score("106", make_case(100)).passed


def test_text_that_is_not_a_number_fails():
    result = make_scorer("numeric").score("about fifty", make_case(50))
    assert not result.passed and "not a number" in result.reason


def test_coerce_reads_a_number_out_of_text():
    assert not make_scorer("numeric").score("The total is $1,249.50.", make_case(1249.5)).passed
    assert make_scorer("numeric", coerce=True).score("The total is $1,249.50.", make_case(1249.5)).passed
    assert make_scorer("numeric", coerce=True).score("-3.5 degrees", make_case(-3.5)).passed


def test_json_field_must_be_a_real_number():
    scorer = make_scorer("numeric", field="refund_amount", abs_tol=0.01)
    case = make_case({"refund_amount": 49.99})
    assert scorer.score('{"refund_amount": 49.99}', case).passed
    as_string = scorer.score('{"refund_amount": "$49.99"}', case)
    assert not as_string.passed
    assert as_string.reason == "refund_amount: expected 49.99, got '$49.99' (not a number)"
    assert not scorer.score('{"refund_amount": true}', case).passed
    assert make_scorer("numeric", field="refund_amount", coerce=True).score('{"refund_amount": "$49.99"}', case).passed


def test_expected_null_field():
    scorer = make_scorer("numeric", field="refund_amount")
    case = make_case({"refund_amount": None})
    assert scorer.score('{"refund_amount": null}', case).passed
    result = scorer.score('{"refund_amount": 0}', case)
    assert not result.passed and "expected null, got 0" in result.reason


def test_field_problems():
    scorer = make_scorer("numeric", field="refund_amount")
    case = make_case({"refund_amount": 5})
    assert "is missing" in scorer.score('{"other": 5}', case).reason
    assert "not valid JSON" in scorer.score("Sure! 5", case).reason


def test_explicit_value_and_expected_field():
    assert make_scorer("numeric", value=7).score("7", make_case("ignored")).passed
    assert make_scorer("numeric", expected_field="total").score("12", make_case({"total": 12})).passed


def test_missing_or_non_numeric_expected():
    assert make_scorer("numeric").score("1", make_case(None)).reason == "no expected number for this case"
    assert "is not a number" in make_scorer("numeric").score("1", make_case("one")).reason


def test_option_errors():
    with pytest.raises(ConfigError, match="non-negative"):
        make_scorer("numeric", abs_tol=-1)
    with pytest.raises(ConfigError, match="`value` must be a number"):
        make_scorer("numeric", value="7")
