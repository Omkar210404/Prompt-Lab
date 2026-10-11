"""The CLI end to end, in a temp folder, with the mock provider only."""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from promptlab import __version__
from promptlab.cli import app
from promptlab.config import load_experiment

cli = CliRunner()
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run each test from an empty folder, as a user would."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def invoke(*args: str, code: int = 0):
    result = cli.invoke(app, list(args))
    assert result.exit_code == code, result.output
    return result


def test_version_and_help():
    assert f"prompt-lab {__version__}" in invoke("--version").output
    help_text = invoke("--help").output
    for command in ("init", "run", "report", "compare", "list"):
        assert command in help_text


def test_init_run_report_compare_list(workdir):
    # init
    out = invoke("init").output
    for name in ("experiment.yaml", "dataset.jsonl", "prices.yaml", ".env.example"):
        assert (workdir / name).is_file() and name in out
    assert "prompt-lab run experiment.yaml" in out

    # run
    out = invoke("run", "experiment.yaml", "--name", "first").output
    assert "2 prompts x 1 model x 6 cases = 12 calls (0 cached, 12 to make)" in out
    assert "MOCK RESULTS" in out and "Results by prompt and model" in out
    assert "Calls: 12 made, 0 from cache" in out and "Saved: runs/first" in out
    run_dir = workdir / "runs" / "first"
    assert {p.name for p in run_dir.iterdir()} == {
        "results.jsonl", "summary.json", "experiment.yaml", "dataset.jsonl", "prices.yaml"}
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["cases"] == 12 and summary["totals"]["passed"] == 6
    assert summary["config_hash"] == load_experiment(workdir / "experiment.yaml").config_hash

    # a second run is served from the cache
    out = invoke("run", "experiment.yaml", "--name", "second").output
    assert "(12 cached, 0 to make)" in out and "Calls: 0 made, 12 from cache" in out
    assert "spent this run: $0)" in out

    # report: table, failures, HTML
    out = invoke("report", "second", "--failures", "2").output
    assert "Worst failures" in out and "exact_match: expected" in out
    out = invoke("report", "latest", "--html").output
    assert "HTML report: runs/second/report.html" in out
    html = (workdir / "runs" / "second" / "report.html").read_text(encoding="utf-8")
    assert "my-first-experiment" in html and "Mock results." in html
    invoke("report", "first", "--out", "exports/first.html")
    assert (workdir / "exports" / "first.html").is_file()

    # compare: the same experiment twice has no regressions
    out = invoke("compare", "first", "second", "--fail-on-regression").output
    assert "Same config hash" in out and "0 regressions, 0 fixes" in out

    # list
    out = invoke("list").output
    assert "first" in out and "second" in out and "my-first-experiment" in out


def test_init_does_not_overwrite_without_force(workdir):
    invoke("init", "lab")
    (workdir / "lab" / "experiment.yaml").write_text("name: mine", encoding="utf-8")
    result = invoke("init", "lab", code=2)
    assert "already exist" in result.output
    assert (workdir / "lab" / "experiment.yaml").read_text(encoding="utf-8") == "name: mine"
    invoke("init", "lab", "--force")
    assert "my-first-experiment" in (workdir / "lab" / "experiment.yaml").read_text(encoding="utf-8")


def test_dry_run_calls_nothing_and_saves_nothing(workdir):
    invoke("init")
    out = invoke("run", "experiment.yaml", "--dry-run").output
    assert "Dry run: nothing was called" in out and "12 calls (0 cached, 12 to make)" in out
    assert not (workdir / "runs").exists() and not (workdir / ".promptlab").exists()


def test_run_filters_and_no_cache(workdir):
    invoke("init")
    out = invoke("run", "experiment.yaml", "--prompt", "v2_one_word", "--limit", "3", "--no-cache", "--name", "small").output
    assert "1 prompt x 1 model x 3 cases = 3 calls" in out
    assert "Partial run" in out and "3 cases of 6" in out
    assert not (workdir / ".promptlab").exists()
    assert "unknown prompt id: nope" in invoke("run", "experiment.yaml", "--prompt", "nope", code=2).output


def test_run_with_html_flag(workdir):
    invoke("init")
    out = invoke("run", "experiment.yaml", "--name", "r", "--html", "--concurrency", "2").output
    assert "HTML report: runs/r/report.html" in out and (workdir / "runs" / "r" / "report.html").is_file()


def test_errors_are_one_line_with_exit_code_2(workdir):
    result = invoke("run", "missing.yaml", code=2)
    assert "Error:" in result.output and "experiment file not found" in result.output
    assert "Traceback" not in result.output

    (workdir / "bad.yaml").write_text("name: x\ndataset: d.jsonl\nprompts: []\nmodels: []\n", encoding="utf-8")
    assert "prompts" in invoke("run", "bad.yaml", code=2).output
    assert "no runs found" in invoke("report", code=2).output
    invoke("init")
    assert "invalid run name" in invoke("run", "experiment.yaml", "--name", "../x", code=2).output


def test_missing_api_key_is_reported_before_any_call(workdir, monkeypatch):
    monkeypatch.delenv("PROMPTLAB_TEST_KEY", raising=False)
    invoke("init")
    text = (workdir / "experiment.yaml").read_text(encoding="utf-8").replace(
        "  - id: mock-model\n    provider: mock\n",
        "  - id: real\n    provider: openai_compatible\n    base_url: https://llm.example/v1\n"
        "    api_key_env: PROMPTLAB_TEST_KEY\n  - id: mock-model\n    provider: mock\n",
    )
    (workdir / "experiment.yaml").write_text(text, encoding="utf-8")
    result = invoke("run", "experiment.yaml", code=2)
    assert "environment variable PROMPTLAB_TEST_KEY is not set" in result.output
    assert not (workdir / "runs").exists()


def test_key_can_come_from_a_dotenv_file(workdir, monkeypatch):
    monkeypatch.delenv("PROMPTLAB_TEST_KEY", raising=False)
    invoke("init")
    (workdir / ".env").write_text("PROMPTLAB_TEST_KEY=sk-from-dotenv\n", encoding="utf-8")
    text = (workdir / "experiment.yaml").read_text(encoding="utf-8").replace(
        "  - id: mock-model\n    provider: mock\n",
        "  - id: real\n    provider: openai_compatible\n    base_url: https://llm.example/v1\n"
        "    api_key_env: PROMPTLAB_TEST_KEY\n  - id: mock-model\n    provider: mock\n",
    )
    (workdir / "experiment.yaml").write_text(text, encoding="utf-8")
    try:
        # --dry-run builds the providers (so the key must resolve) but calls nothing
        out = invoke("run", "experiment.yaml", "--dry-run").output
        assert "24 calls" in out
    finally:
        os.environ.pop("PROMPTLAB_TEST_KEY", None)  # load_dotenv put it in os.environ


def test_list_scorers_and_empty_list(workdir):
    out = invoke("list", "--scorers").output
    for name in ("exact_match", "contains", "not_contains", "regex", "json_valid", "json_schema",
                 "json_fields", "numeric", "length", "llm_judge"):
        assert name in out
    assert "No runs in runs yet" in invoke("list").output


# --- the shipped examples must load and run ---

@pytest.mark.parametrize("name", ["support_extraction.yaml", "support_extraction_shortened.yaml", "sentiment.yaml"])
def test_shipped_examples_run_offline(workdir, name):
    out = invoke("run", str(EXAMPLES / name), "--name", "example", "--no-cache", "--concurrency", "16").output
    assert "MOCK RESULTS" in out and "0 errors" in out
    summary = json.loads((workdir / "runs" / "example" / "summary.json").read_text(encoding="utf-8"))
    assert summary["dataset"]["cases"] == 24 and summary["dataset"]["meta"]["synthetic"] is True
    assert summary["totals"]["errors"] == 0 and summary["totals"]["cost_usd"] > 0


def test_demo_tells_the_story_it_claims(workdir):
    """The README says each prompt version passes more cases than the last, and that shortening
    v3 causes regressions. Both come from the mock rules; this keeps the claim and the files in step."""
    invoke("run", str(EXAMPLES / "support_extraction.yaml"), "--name", "demo", "--concurrency", "16")
    invoke("run", str(EXAMPLES / "support_extraction_shortened.yaml"), "--name", "short", "--concurrency", "16")
    summary = json.loads((workdir / "runs" / "demo" / "summary.json").read_text(encoding="utf-8"))
    for model in ("mock-careful", "mock-sloppy"):
        rates = [g["pass_rate"] for g in summary["groups"] if g["model_id"] == model]
        assert rates == sorted(rates) and rates[0] < rates[-1]

    result = cli.invoke(app, ["compare", "demo", "short", "--fail-on-regression", "--json"])
    assert result.exit_code == 1
    comparison = json.loads(result.output)
    assert comparison["changed_prompts"] == ["v3_fewshot"] and comparison["counts"]["fixes"] == 0
    assert {r["prompt_id"] for r in comparison["regressions"]} == {"v3_fewshot"}
