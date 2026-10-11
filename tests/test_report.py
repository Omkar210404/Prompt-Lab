"""The CLI table and the self-contained HTML report."""

import json
import re
from datetime import datetime, timedelta, timezone

from rich.console import Console

from promptlab.cache import ResponseCache
from promptlab.config import load_experiment
from promptlab.pricing import Price
from promptlab.report.html import json_for_script, render_html
from promptlab.report.table import fmt_cost, fmt_ms, fmt_rate, fmt_tokens, print_run
from promptlab.runner import Runner
from promptlab.storage import load_run, save_run
from promptlab.summary import hardest_cases, percentile, worst_failures

T0 = datetime(2026, 10, 11, 9, 0, 0, tzinfo=timezone.utc)
HOSTILE = '</script><script>alert("xss")</script><img src=x onerror=alert(1)> & "quotes"'


async def stored_run(write_experiment, tmp_path, changes=None, cases=None, prices=None, providers=None):
    loaded = load_experiment(write_experiment(changes, cases=cases))
    runner = Runner(loaded, ResponseCache(tmp_path / "cache", enabled=False), prices=prices, providers=providers)
    return load_run(save_run(tmp_path / "runs", loaded, await runner.run(), started_at=T0,
                             finished_at=T0 + timedelta(seconds=2), prices=prices, name="r1"))


def two_models(rules) -> dict:
    return {"models": [{"id": "m1", "provider": "mock"},
                       {"id": "m2", "provider": "mock", "mock": {"rules": rules}}]}


def embedded_data(html: str) -> dict:
    match = re.search(r'<script id="pl-data" type="application/json">(.*?)</script>', html, re.DOTALL)
    return json.loads(match.group(1))


def table_text(run, **kwargs) -> str:
    console = Console(width=140, record=True, force_terminal=False)
    print_run(console, run.summary, run.rows, **kwargs)
    return console.export_text()


# --- formatting ---

def test_formatters_never_turn_unknown_into_zero():
    assert fmt_cost(None) == "n/a" and fmt_cost(0) == "$0" and fmt_cost(0.000123) == "$0.000123" and fmt_cost(1.5) == "$1.5000"
    assert fmt_rate(None) == "n/a" and fmt_rate(0.8333) == "83.3%"
    assert fmt_ms(None) == "n/a" and fmt_ms(3.21) == "3.2 ms" and fmt_ms(431.7) == "432 ms"
    assert fmt_tokens(None, None) == "n/a" and fmt_tokens(12345, 678) == "12,345 / 678"


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([10], 95) == 10
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([1, 2, 3, 4, 5], 95) == 4.8


# --- CLI table ---

async def test_table_shows_groups_scorers_and_totals(write_experiment, tmp_path):
    rules = [{"when": {"case_ids": ["c3"]}, "behave": "wrong", "value": "nope"}]
    run = await stored_run(write_experiment, tmp_path, two_models(rules), prices={"m1": Price(input=1, output=2)})
    text = table_text(run, failures=5)

    assert "MOCK RESULTS" in text and "not a benchmark" in " ".join(text.split())  # the banner may wrap
    assert "Results by prompt and model" in text and "Pass rate by scorer" in text
    assert re.search(r"p1\s+m1\s+100\.0%\s+3/3", text)
    assert re.search(r"p1\s+m2\s+66\.7%\s+2/3", text)
    assert "Worst failures" in text and "exact_match: expected 'neutral', got 'nope'" in text
    assert "Cells: 12 (10 passed, 2 failed, 0 errors)" in text
    assert "Calls: 12 made, 0 from cache" in text
    assert re.search(r"m2 .* n/a\s*$", text, re.MULTILINE)  # m2 has no price entry: n/a, not $0


async def test_table_without_mock_has_no_banner(write_experiment, tmp_path):
    run = await stored_run(write_experiment, tmp_path)
    run.summary["mock_models"] = []
    assert "MOCK RESULTS" not in table_text(run)


# --- HTML report ---

async def test_html_contains_the_run(write_experiment, tmp_path):
    rules = [{"when": {"case_ids": ["c3"]}, "behave": "wrong", "value": "nope"}]
    run = await stored_run(write_experiment, tmp_path, two_models(rules))
    html = render_html(run)

    assert html.startswith("<!doctype html>") and "<title>test-exp · prompt-lab report</title>" in html
    assert "Mock results." in html and "not a benchmark" in html
    assert run.summary["config_hash"][:16] in html and "2026-10-11T09:00:00Z" in html
    for heading in ("Pass rate by prompt and model", "Pass rate by scorer", "Side by side", "Worst failures", "Prompts"):
        assert heading in html
    assert 'style="width: 66.67%"' in html and 'style="width: 100.0%"' in html  # chart bars
    assert "Classify: {text}" in html  # prompt templates
    assert "exact_match: expected &#39;neutral&#39;, got &#39;nope&#39;" in html  # worst failures, escaped

    data = embedded_data(html)
    assert data["prompts"] == ["p1", "p2"] and data["models"] == ["m1", "m2"]
    assert [c["id"] for c in data["cases"]] == ["c1", "c2", "c3"]
    assert data["cases"][0] == {"id": "c1", "vars": {"text": "great product"}, "expected": "positive", "tags": ["easy"]}
    assert len(data["rows"]) == 12
    failing = [r for r in data["rows"] if not r["passed"]]
    assert {(r["case_id"], r["model_id"]) for r in failing} == {("c3", "m2")}
    assert failing[0]["output"] == "nope" and failing[0]["scores"][0]["reason"] == "expected 'neutral', got 'nope'"


async def test_html_is_self_contained(write_experiment, tmp_path):
    html = render_html(await stored_run(write_experiment, tmp_path))
    assert not re.search(r"""(?:src|href)\s*=\s*["']?(?:https?:)?//""", html)  # no remote scripts, styles or images
    assert "<link" not in html and "@import" not in html and "url(" not in html
    assert "http://" not in html and "https://" not in html
    assert html.count("<script") == 2 and "<style>" in html  # the data block and one inline script


async def test_html_escapes_everything_that_comes_from_data(write_experiment, tmp_path):
    cases = [{"id": "c<1>", "vars": {"text": HOSTILE}, "expected": HOSTILE, "tags": ["<b>tag</b>"]}]
    changes = {
        "name": "exp <script>alert(0)</script>",
        "description": "desc <img src=x onerror=alert(2)>",
        "models": [{"id": "m1", "provider": "mock", "mock": {"default": "reply", "value": HOSTILE + " extra"}}],
    }
    html = render_html(await stored_run(write_experiment, tmp_path, changes, cases=cases))

    assert "alert(" in html  # the text is there...
    assert "<script>alert" not in html and "<img src=x" not in html and "<b>tag</b>" not in html  # ...but never as markup
    assert html.count("</script>") == 2  # nothing in the data closed a script element early
    data = embedded_data(html)  # and it still round-trips exactly
    assert data["cases"][0]["vars"]["text"] == HOSTILE
    assert data["rows"][0]["output"] == HOSTILE + " extra"


def test_json_for_script_cannot_close_the_script_tag():
    encoded = json_for_script({"a": "</script><!-- &  "})
    assert "<" not in encoded and ">" not in encoded and "&" not in encoded and " " not in encoded
    assert json.loads(encoded) == {"a": "</script><!-- &  "}


async def test_html_with_errors_and_no_cost(write_experiment, tmp_path):
    rules = [{"when": {"case_ids": ["c1"]}, "behave": "error", "kind": "auth"}]
    changes = {"models": [{"id": "m1", "provider": "mock", "mock": {"rules": rules}}]}
    run = await stored_run(write_experiment, tmp_path, changes)
    html = render_html(run)
    assert "(2 call errors)" in html
    assert "auth_error: mock: simulated auth failure (HTTP 401)" in html
    assert "n/a (spent this run: n/a)" in html  # no price table: cost is unknown, not zero
    assert embedded_data(html)["rows"][0]["error"]["type"] == "auth_error"


async def test_html_when_everything_passes(write_experiment, tmp_path):
    html = render_html(await stored_run(write_experiment, tmp_path))
    assert "No failures in this run." in html


async def test_more_than_eight_models_share_a_neutral_colour(write_experiment, tmp_path):
    models = [{"id": f"m{i}", "provider": "mock"} for i in range(1, 11)]
    html = render_html(await stored_run(write_experiment, tmp_path, {"models": models}))
    assert '<span class="key s8"></span>m8' in html
    assert '<span class="key s0"></span>m9' in html and '<span class="key s0"></span>m10' in html


# --- failure lists ---

def row(case, prompt, model, score, passed=False, status="ok"):
    return {"case_id": case, "prompt_id": prompt, "model_id": model, "score": score, "passed": passed,
            "status": status, "error": {"type": "timeout", "message": "slow"} if status == "error" else None,
            "scores": [] if status == "error" else [{"scorer": "s", "passed": passed, "reason": f"reason {score}"}]}


def test_worst_failures_lowest_score_first_then_most_surprising():
    rows = [
        row("a", "bad", "m", 0.0), row("b", "bad", "m", 0.0), row("c", "bad", "m", 0.0),
        row("a", "good", "m", 1.0, passed=True), row("b", "good", "m", 1.0, passed=True), row("c", "good", "m", 0.0),
        row("d", "good", "m", 0.5), row("d", "bad", "m", 0.0, status="error"),
    ]
    worst = worst_failures(rows, limit=3)
    # all three have score 0; the one from the prompt that usually passes comes first
    assert [(w["case_id"], w["prompt_id"]) for w in worst] == [("c", "good"), ("a", "bad"), ("b", "bad")]
    assert worst[0]["reasons"] == ["s: reason 0.0"]
    assert worst_failures(rows, limit=50)[-1]["score"] == 0.5
    errored = next(w for w in worst_failures(rows, 50) if w["status"] == "error")
    assert errored["reasons"] == ["timeout: slow"]


def test_hardest_cases_fail_under_the_most_variants():
    rows = [
        row("a", "p1", "m", 0.0), row("a", "p2", "m", 0.5), row("a", "p3", "m", 1.0, passed=True),
        row("b", "p1", "m", 0.0), row("b", "p2", "m", 1.0, passed=True), row("b", "p3", "m", 1.0, passed=True),
        row("c", "p1", "m", 1.0, passed=True),
    ]
    assert hardest_cases(rows) == [
        {"case_id": "a", "cells": 3, "failed": 2, "mean_score": 0.5},
        {"case_id": "b", "cells": 3, "failed": 1, "mean_score": 0.6667},
    ]
