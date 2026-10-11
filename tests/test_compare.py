"""compare: regressions are cases that passed in run A and fail in run B."""

import json
from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from promptlab.cache import ResponseCache
from promptlab.cli import app
from promptlab.compare import compare_runs
from promptlab.config import load_experiment
from promptlab.runner import Runner
from promptlab.storage import load_run, save_run

T0 = datetime(2026, 10, 11, 9, 0, 0, tzinfo=timezone.utc)
cli = CliRunner()


def model(rules=None) -> dict:
    return {"models": [{"id": "m1", "provider": "mock", "mock": {"rules": rules or []}}]}


def wrong_for(*case_ids, prompt=None) -> dict:
    when = {"case_ids": list(case_ids)}
    if prompt:
        when["prompt_matches"] = prompt
    return {"when": when, "behave": "wrong", "value": "banana"}


async def save(write_experiment, tmp_path, name, changes=None, cases=None, hours=0):
    loaded = load_experiment(write_experiment(changes, cases=cases))
    results = await Runner(loaded, ResponseCache(tmp_path / "cache", enabled=False)).run()
    started = T0 + timedelta(hours=hours)
    return load_run(save_run(tmp_path / "runs", loaded, results, started_at=started,
                             finished_at=started + timedelta(seconds=1), name=name))


async def test_identical_runs_have_no_regressions(write_experiment, tmp_path):
    a = await save(write_experiment, tmp_path, "a")
    b = await save(write_experiment, tmp_path, "b", hours=1)
    result = compare_runs(a, b)
    assert result["same_config"] and result["same_dataset"]
    assert result["counts"] == {"compared": 6, "regressions": 0, "fixes": 0, "still_passing": 6,
                                "still_failing": 0, "only_in_a": 0, "only_in_b": 0}
    assert result["regressions"] == [] and result["warnings"] == []
    assert result["changed_prompts"] == [] and not result["changed_scorers"]


async def test_regressions_and_fixes(write_experiment, tmp_path):
    # A: c1 fails everywhere. B: c1 is fixed, but c3 now fails under p2 only.
    a = await save(write_experiment, tmp_path, "a", model([wrong_for("c1")]))
    b = await save(write_experiment, tmp_path, "b", model([wrong_for("c3", prompt="one word")]), hours=1)
    result = compare_runs(a, b)

    assert result["counts"] == {"compared": 6, "regressions": 1, "fixes": 2, "still_passing": 3,
                                "still_failing": 0, "only_in_a": 0, "only_in_b": 0}
    assert result["regressions"] == [{
        "case_id": "c3", "prompt_id": "p2", "model_id": "m1", "score_a": 1.0, "score_b": 0.0,
        "reasons": ["exact_match: expected 'neutral', got 'banana'"],  # why it fails now
    }]
    assert [(f["case_id"], f["prompt_id"]) for f in result["fixes"]] == [("c1", "p1"), ("c1", "p2")]
    assert result["fixes"][0]["reasons"] == ["exact_match: expected 'positive', got 'banana'"]  # why it used to fail
    assert not result["same_config"] and result["changed_models"] == ["m1"]

    p1, p2 = result["groups"]
    assert (p1["passed_a"], p1["passed_b"], p1["pass_rate_delta"], p1["regressions"], p1["fixes"]) == (2, 3, 0.3333, 0, 1)
    assert (p2["passed_a"], p2["passed_b"], p2["pass_rate_delta"], p2["regressions"], p2["fixes"]) == (2, 2, 0.0, 1, 1)


async def test_still_failing_is_not_a_regression(write_experiment, tmp_path):
    a = await save(write_experiment, tmp_path, "a", model([wrong_for("c2")]))
    b = await save(write_experiment, tmp_path, "b", model([wrong_for("c2")]), hours=1)
    counts = compare_runs(a, b)["counts"]
    assert counts["regressions"] == 0 and counts["still_failing"] == 2 and counts["still_passing"] == 4


async def test_a_call_error_counts_as_a_regression(write_experiment, tmp_path):
    a = await save(write_experiment, tmp_path, "a")
    failing = model([{"when": {"case_ids": ["c1"]}, "behave": "error", "kind": "auth"}])
    b = await save(write_experiment, tmp_path, "b", failing, hours=1)
    regression = compare_runs(a, b)["regressions"][0]
    assert regression["case_id"] == "c1" and regression["reasons"] == ["auth_error: mock: simulated auth failure (HTTP 401)"]


async def test_changed_prompt_is_named(write_experiment, tmp_path):
    a = await save(write_experiment, tmp_path, "a")
    edited = {"prompts": [{"id": "p1", "template": "Classify: {text}"}, {"id": "p2", "template": "One word only: {text}"}]}
    b = await save(write_experiment, tmp_path, "b", edited, hours=1)
    result = compare_runs(a, b)
    assert result["changed_prompts"] == ["p2"] and not result["same_config"] and result["same_dataset"]


async def test_added_and_removed_cells_are_counted_not_compared(write_experiment, tmp_path):
    a = await save(write_experiment, tmp_path, "a")
    extra_prompt = {"prompts": [{"id": "p1", "template": "Classify: {text}"}, {"id": "p3", "template": "New: {text}"}]}
    b = await save(write_experiment, tmp_path, "b", extra_prompt, hours=1)
    result = compare_runs(a, b)
    assert result["counts"]["compared"] == 3
    assert result["counts"]["only_in_a"] == 3 and result["counts"]["only_in_b"] == 3
    assert ["c1", "p2", "m1"] in result["only_in_a"] and ["c1", "p3", "m1"] in result["only_in_b"]
    assert [g["prompt_id"] for g in result["groups"]] == ["p1"]  # rates only over shared cells


async def test_different_test_sets_warn_and_name_changed_cases(write_experiment, tmp_path):
    a = await save(write_experiment, tmp_path, "a")
    cases = [
        {"id": "c1", "vars": {"text": "great product"}, "expected": "positive"},     # same content
        {"id": "c2", "vars": {"text": "terrible service"}, "expected": "neutral"},   # expected changed
        {"id": "c9", "vars": {"text": "brand new case"}, "expected": "positive"},
    ]
    b = await save(write_experiment, tmp_path, "b", cases=cases, hours=1)
    result = compare_runs(a, b)
    assert not result["same_dataset"] and result["changed_cases"] == ["c2"]
    assert "test sets differ" in result["warnings"][0] and "c2" in result["warnings"][0]
    assert result["counts"]["compared"] == 4  # c1 and c2 under two prompts


async def test_no_shared_cells_warns(write_experiment, tmp_path):
    a = await save(write_experiment, tmp_path, "a")
    b = await save(write_experiment, tmp_path, "b", {"models": [{"id": "other", "provider": "mock"}]}, hours=1)
    result = compare_runs(a, b)
    assert result["counts"]["compared"] == 0
    assert any("no cells in common" in warning for warning in result["warnings"])


# --- the CLI command ---

async def test_cli_compare_output_and_exit_codes(write_experiment, tmp_path):
    await save(write_experiment, tmp_path, "before")
    await save(write_experiment, tmp_path, "after", model([wrong_for("c3", prompt="one word")]), hours=1)
    runs = ["--runs-dir", str(tmp_path / "runs")]

    shown = cli.invoke(app, ["compare", "before", "after", *runs])
    assert shown.exit_code == 0  # without the flag, regressions are reported, not fatal
    assert "Regressions: passed in A, fail in B" in shown.output
    assert "c3" in shown.output and "expected 'neutral', got 'banana'" in shown.output
    assert "1 regressions, 0 fixes" in shown.output
    assert "MOCK RESULTS" in shown.output

    strict = cli.invoke(app, ["compare", "before", "after", "--fail-on-regression", *runs])
    assert strict.exit_code == 1

    clean = cli.invoke(app, ["compare", "after", "before", "--fail-on-regression", *runs])
    assert clean.exit_code == 0 and "Fixes: failed in A, pass in B" in clean.output


async def test_cli_compare_json(write_experiment, tmp_path):
    await save(write_experiment, tmp_path, "before")
    await save(write_experiment, tmp_path, "after", model([wrong_for("c1")]), hours=1)
    result = cli.invoke(app, ["compare", "before", "after", "--json", "--runs-dir", str(tmp_path / "runs")])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["counts"]["regressions"] == 2 and data["run_a"]["run_id"] == "before"


def test_cli_compare_unknown_run(tmp_path):
    result = cli.invoke(app, ["compare", "x", "y", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 2 and "no runs found" in result.output
