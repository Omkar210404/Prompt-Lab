import json

import pytest

from promptlab.config import load_experiment, render_template, template_variables
from promptlab.errors import ConfigError


# --- templating ---

def test_render_substitutes_variables():
    assert render_template("Hello {name}, order {order_id}", {"name": "Asha", "order_id": 7}) == "Hello Asha, order 7"


def test_json_examples_in_a_prompt_are_left_alone():
    template = 'Example: {"order_id": "ORD-1", "nested": {"a": 1}}\nEmail: {email}'
    assert template_variables(template) == {"email"}
    assert render_template(template, {"email": "hi"}) == 'Example: {"order_id": "ORD-1", "nested": {"a": 1}}\nEmail: hi'


def test_double_braces_write_a_literal_placeholder():
    assert template_variables("Use {{name}} as is, fill {city}") == {"city"}
    assert render_template("Use {{name}} as is, fill {city}", {"city": "Pune"}) == "Use {name} as is, fill Pune"


def test_substituted_values_are_not_rendered_again():
    assert render_template("{email}", {"email": "my id is {secret}"}) == "my id is {secret}"


def test_render_missing_variable_raises():
    with pytest.raises(ConfigError, match="{city}"):
        render_template("{city}", {})


# --- loading ---

def test_load_valid_experiment(write_experiment):
    loaded = load_experiment(write_experiment())
    assert loaded.experiment.name == "test-exp"
    assert [c.id for c in loaded.cases] == ["c1", "c2", "c3"]
    assert loaded.experiment.models[0].model == "m1"  # defaults to the id
    assert len(loaded.config_hash) == 64


def test_meta_line_and_auto_ids(write_experiment):
    raw = "\n".join([
        json.dumps({"_meta": {"synthetic": True, "description": "made up"}}),
        json.dumps({"vars": {"text": "a"}, "expected": "x"}),
        "",
        json.dumps({"vars": {"text": "b"}, "expected": "y"}),
    ])
    loaded = load_experiment(write_experiment(raw_dataset=raw))
    assert loaded.dataset_meta == {"synthetic": True, "description": "made up"}
    assert [c.id for c in loaded.cases] == ["case-0001", "case-0002"]


def test_missing_variable_is_a_load_error(write_experiment):
    path = write_experiment({"prompts": [{"id": "p1", "template": "Classify {text} for {customer}"}]})
    with pytest.raises(ConfigError, match=r"prompt 'p1' uses \{customer\} but test case 'c1'"):
        load_experiment(path)


def test_duplicate_prompt_ids(write_experiment):
    path = write_experiment({"prompts": [{"id": "p", "template": "{text}"}, {"id": "p", "template": "{text}!"}]})
    with pytest.raises(ConfigError, match="duplicate prompt id: p"):
        load_experiment(path)


def test_duplicate_case_ids(write_experiment):
    cases = [{"id": "same", "vars": {"text": "a"}, "expected": "x"}] * 2
    with pytest.raises(ConfigError, match="duplicate case id: same"):
        load_experiment(write_experiment(cases=cases))


def test_unknown_scorer_type(write_experiment):
    with pytest.raises(ConfigError, match="unknown scorer type 'vibes'"):
        load_experiment(write_experiment({"scorers": [{"type": "vibes"}]}))


def test_bad_scorer_option(write_experiment):
    with pytest.raises(ConfigError, match="scorer 'regex'"):
        load_experiment(write_experiment({"scorers": [{"type": "regex", "pattern": "("}]}))
    with pytest.raises(ConfigError, match="scorer 'exact_match'.*ignore_caes"):
        load_experiment(write_experiment({"scorers": [{"type": "exact_match", "ignore_caes": True}]}))


def test_bad_check_names_the_case(write_experiment):
    cases = [{"id": "c9", "vars": {"text": "a"}, "expected": "x", "checks": [{"type": "length"}]}]
    with pytest.raises(ConfigError, match="test case 'c9'.*scorer 'length'"):
        load_experiment(write_experiment(cases=cases))


def test_case_without_any_scorer(write_experiment):
    with pytest.raises(ConfigError, match="has no scorers"):
        load_experiment(write_experiment({"scorers": []}))


def test_unknown_provider(write_experiment):
    with pytest.raises(ConfigError, match="unknown provider 'telepathy'"):
        load_experiment(write_experiment({"models": [{"id": "m", "provider": "telepathy"}]}))


def test_unknown_key_in_experiment(write_experiment):
    with pytest.raises(ConfigError, match="promts"):
        load_experiment(write_experiment({"promts": []}))


def test_bad_jsonl_line_reports_line_number(write_experiment):
    raw = json.dumps({"id": "ok", "vars": {"text": "a"}, "expected": "x"}) + "\n{not json}\n"
    with pytest.raises(ConfigError, match="cases.jsonl line 2: invalid JSON"):
        load_experiment(write_experiment(raw_dataset=raw))


def test_unknown_case_key_reports_line_number(write_experiment):
    raw = json.dumps({"id": "ok", "input": {"text": "a"}}) + "\n"
    with pytest.raises(ConfigError, match="cases.jsonl line 1: input"):
        load_experiment(write_experiment(raw_dataset=raw))


def test_missing_files(write_experiment, tmp_path):
    with pytest.raises(ConfigError, match="experiment file not found"):
        load_experiment(tmp_path / "nope.yaml")
    path = write_experiment()
    (tmp_path / "cases.jsonl").unlink()
    with pytest.raises(ConfigError, match="dataset not found"):
        load_experiment(path)


def test_unknown_judge(write_experiment):
    scorers = [{"type": "llm_judge", "rubric": "Is it polite?", "judge": "ghost"}]
    with pytest.raises(ConfigError, match="judge 'ghost' is not defined"):
        load_experiment(write_experiment({"scorers": scorers}))


def test_judge_same_as_tested_model_warns(write_experiment):
    scorers = [{"type": "llm_judge", "rubric": "Is it polite?", "judge": "m1"}]
    loaded = load_experiment(write_experiment({"scorers": scorers}))
    assert len(loaded.warnings) == 1 and "same model" in loaded.warnings[0]

    separate = {"scorers": [{"type": "llm_judge", "rubric": "Is it polite?", "judge": "j1"}],
                "judges": [{"id": "j1", "provider": "mock"}]}
    assert load_experiment(write_experiment(separate)).warnings == []


# --- config hash ---

def test_config_hash_is_stable(write_experiment):
    path = write_experiment()
    assert load_experiment(path).config_hash == load_experiment(path).config_hash


def test_config_hash_ignores_run_settings_and_line_endings(write_experiment, tmp_path):
    base = load_experiment(write_experiment()).config_hash
    assert load_experiment(write_experiment({"run": {"concurrency": 9}, "description": "x"})).config_hash == base

    dataset = tmp_path / "cases.jsonl"
    unix = dataset.read_bytes().replace(b"\r\n", b"\n")
    for line_ending in (b"\n", b"\r\n"):
        dataset.write_bytes(unix.replace(b"\n", line_ending))
        assert load_experiment(tmp_path / "exp.yaml").config_hash == base


@pytest.mark.parametrize("changes", [
    {"prompts": [{"id": "p1", "template": "Classify this: {text}"}]},
    {"models": [{"id": "m1", "provider": "mock", "params": {"temperature": 1}}]},
    {"scorers": [{"type": "exact_match", "ignore_case": True}]},
])
def test_config_hash_changes_with_what_is_measured(write_experiment, changes):
    base = load_experiment(write_experiment()).config_hash
    assert load_experiment(write_experiment(changes)).config_hash != base


def test_config_hash_changes_with_dataset_and_schema_file(write_experiment, tmp_path):
    base = load_experiment(write_experiment()).config_hash
    cases = [{"id": "c1", "vars": {"text": "different"}, "expected": "positive"}]
    assert load_experiment(write_experiment(cases=cases)).config_hash != base

    schema = tmp_path / "s.json"
    schema.write_text('{"type": "object"}', encoding="utf-8")
    with_schema = {"scorers": [{"type": "json_schema", "schema_file": "s.json"}]}
    first = load_experiment(write_experiment(with_schema)).config_hash
    schema.write_text('{"type": "array"}', encoding="utf-8")
    assert load_experiment(write_experiment(with_schema)).config_hash != first
