"""The runner, driven by the mock provider. No test here touches the network."""

import asyncio

import pytest

from promptlab.cache import ResponseCache
from promptlab.config import TestCase, load_experiment
from promptlab.errors import ConfigError
from promptlab.providers import CompletionResponse, MockProvider
from promptlab.runner import Runner, RunOptions
from promptlab.scorers import ScoreResult, Scorer, register_scorer

JUDGE_PASS = '{"reasoning": "fine", "verdict": "pass"}'


def make_runner(path, tmp_path, cache=False, **kwargs) -> Runner:
    return Runner(load_experiment(path), ResponseCache(tmp_path / "cache", enabled=cache), **kwargs)


def mock_model(rules=None, **settings) -> dict:
    return {"id": "m1", "provider": "mock", "mock": {"rules": rules or [], **settings}}


class Tracking(MockProvider):
    """Mock that records how many calls are in flight at once."""

    def __init__(self, delay=0.02, **kwargs):
        super().__init__(**kwargs)
        self.delay, self.in_flight, self.peak = delay, 0, 0

    async def complete(self, request):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return await super().complete(request)
        finally:
            self.in_flight -= 1


async def test_runs_every_prompt_model_case_in_a_stable_order(write_experiment, tmp_path):
    models = [{"id": "m1", "provider": "mock"}, {"id": "m2", "provider": "mock"}]
    rows = await make_runner(write_experiment({"models": models}), tmp_path).run()

    assert len(rows) == 2 * 2 * 3
    assert [(r.prompt_id, r.model_id, r.case_id) for r in rows[:4]] == [
        ("p1", "m1", "c1"), ("p1", "m1", "c2"), ("p1", "m1", "c3"), ("p1", "m2", "c1"),
    ]
    assert all(r.status == "ok" and r.passed and r.score == 1.0 for r in rows)
    first = rows[0]
    assert first.output == "positive" and first.provider == "mock" and first.attempts == 1
    assert first.latency_ms is not None and first.usage == {"prompt_tokens": 3, "completion_tokens": 1}
    assert first.scores[0].model_dump() == {
        "scorer": "exact_match", "type": "exact_match", "passed": True, "score": 1.0,
        "reason": "matches expected", "required": True, "error": False,
    }


async def test_wrong_outputs_fail_with_reasons(write_experiment, tmp_path):
    rules = [{"when": {"tags": ["hard"]}, "behave": "wrong", "value": "positive"}]
    rows = await make_runner(write_experiment({"models": [mock_model(rules)]}), tmp_path).run()
    failed = [r for r in rows if not r.passed]
    assert {r.case_id for r in failed} == {"c3"} and len(failed) == 2
    assert failed[0].scores[0].reason == "expected 'neutral', got 'positive'"
    assert failed[0].score == 0.0


async def test_the_mock_sees_the_prompt_template(write_experiment, tmp_path):
    rules = [{"when": {"prompt_lacks": "one word"}, "behave": "prose_wrapped", "prefix": "It is ", "suffix": "."}]
    rows = await make_runner(write_experiment({"models": [mock_model(rules)]}), tmp_path).run()
    assert [r.passed for r in rows] == [False, False, False, True, True, True]  # p1 fails, p2 passes
    assert rows[0].output == "It is positive."


async def test_global_concurrency_limit_is_respected(write_experiment, tmp_path):
    provider = Tracking()
    path = write_experiment({"run": {"concurrency": 2}})
    rows = await make_runner(path, tmp_path, providers={"m1": provider}).run()
    assert len(rows) == 6 and provider.peak == 2

    unlimited = Tracking()
    await make_runner(path, tmp_path, providers={"m1": unlimited}, options=RunOptions(concurrency=6)).run()
    assert unlimited.peak == 6  # the limit, not the test, is what held it at 2


async def test_per_model_concurrency_limit(write_experiment, tmp_path):
    slow, fast = Tracking(), Tracking()
    models = [{"id": "m1", "provider": "mock", "concurrency": 1}, {"id": "m2", "provider": "mock"}]
    path = write_experiment({"models": models, "run": {"concurrency": 8}})
    await make_runner(path, tmp_path, providers={"m1": slow, "m2": fast}).run()
    assert slow.peak == 1 and fast.peak > 1


async def test_timeout_becomes_an_error_row_and_the_run_continues(write_experiment, tmp_path):
    class SlowOnC2(MockProvider):
        async def complete(self, request):
            if request.metadata["case_id"] == "c2":
                await asyncio.sleep(5)
            return await super().complete(request)

    path = write_experiment({"run": {"timeout_seconds": 0.05, "max_retries": 1, "retry_base_delay": 0}})
    rows = await make_runner(path, tmp_path, providers={"m1": SlowOnC2()}).run()

    errors = [r for r in rows if r.status == "error"]
    assert {r.case_id for r in errors} == {"c2"} and len(errors) == 2
    assert errors[0].error == {"type": "timeout", "message": "no response within 0.05s"}
    assert errors[0].attempts == 2 and errors[0].output is None
    assert not errors[0].passed and errors[0].score == 0.0 and errors[0].scores == []
    assert sum(r.passed for r in rows) == 4


async def test_retry_then_success(write_experiment, tmp_path):
    rules = [{"behave": "error", "kind": "server", "times": 2}]
    rows = await make_runner(write_experiment({"models": [mock_model(rules)]}), tmp_path).run()
    assert all(r.status == "ok" and r.attempts == 3 and r.passed for r in rows)


async def test_retries_are_bounded(write_experiment, tmp_path):
    rules = [{"behave": "error", "kind": "rate_limit"}]
    rows = await make_runner(write_experiment({"models": [mock_model(rules)]}), tmp_path).run()
    assert all(r.status == "error" and r.error["type"] == "rate_limited" and r.attempts == 3 for r in rows)


async def test_auth_errors_are_not_retried(write_experiment, tmp_path):
    provider = MockProvider(settings={"rules": [{"behave": "error", "kind": "auth"}]})
    rows = await make_runner(write_experiment(), tmp_path, providers={"m1": provider}).run()
    assert all(r.error["type"] == "auth_error" and r.attempts == 1 for r in rows)
    assert len(provider.calls) == 6  # one call per cell, no retries


async def test_one_failing_cell_does_not_stop_the_run(write_experiment, tmp_path):
    rules = [{"when": {"case_ids": ["c2"]}, "behave": "error", "kind": "bad_request"}]
    rows = await make_runner(write_experiment({"models": [mock_model(rules)]}), tmp_path).run()
    assert [r.status for r in rows] == ["ok", "error", "ok"] * 2
    assert rows[1].error["type"] == "bad_request"


async def test_unexpected_provider_exception_is_contained(write_experiment, tmp_path):
    class Buggy(MockProvider):
        async def complete(self, request):
            if request.metadata["case_id"] == "c1":
                raise KeyError("oops")
            return await super().complete(request)

    rows = await make_runner(write_experiment(), tmp_path, providers={"m1": Buggy()}).run()
    assert rows[0].status == "error" and rows[0].error["type"] == "unexpected_error"
    assert "KeyError" in rows[0].error["message"]
    assert rows[1].status == "ok"


async def test_crashing_scorer_is_contained(write_experiment, tmp_path):
    @register_scorer
    class Crashy(Scorer):
        type = "crashy_for_test"

        def score(self, output: str, case: TestCase) -> ScoreResult:
            raise ZeroDivisionError("boom")

    scorers = [{"type": "exact_match"}, {"type": "crashy_for_test"}]
    rows = await make_runner(write_experiment({"scorers": scorers}), tmp_path).run()
    crashed = rows[0].scores[1]
    assert crashed.error and not crashed.passed and "scorer crashed: ZeroDivisionError: boom" in crashed.reason
    assert rows[0].scores[0].passed and not rows[0].passed and rows[0].score == 0.5


async def test_filters(write_experiment, tmp_path):
    models = [{"id": "m1", "provider": "mock"}, {"id": "m2", "provider": "mock"}]
    path = write_experiment({"models": models})
    rows = await make_runner(path, tmp_path, options=RunOptions(prompt_ids=["p2"], model_ids=["m2"], limit=2)).run()
    assert [(r.prompt_id, r.model_id, r.case_id) for r in rows] == [("p2", "m2", "c1"), ("p2", "m2", "c2")]

    with pytest.raises(ConfigError, match="unknown prompt id: p9"):
        make_runner(path, tmp_path, options=RunOptions(prompt_ids=["p9"]))
    with pytest.raises(ConfigError, match="unknown model id: gpt"):
        make_runner(path, tmp_path, options=RunOptions(model_ids=["gpt"]))


async def test_per_case_checks_and_duplicate_scorer_names(write_experiment, tmp_path):
    cases = [
        {"id": "plain", "vars": {"text": "a"}, "expected": "positive"},
        {"id": "checked", "vars": {"text": "b"}, "expected": "positive", "checks": [
            {"type": "length", "max_words": 1}, {"type": "length", "max_chars": 3}]},
    ]
    rows = await make_runner(write_experiment(cases=cases), tmp_path, options=RunOptions(prompt_ids=["p1"])).run()
    assert [s.scorer for s in rows[0].scores] == ["exact_match"]
    assert [s.scorer for s in rows[1].scores] == ["exact_match", "length", "length#2"]
    assert rows[0].passed and not rows[1].passed  # "positive" is longer than 3 chars
    assert rows[1].score == pytest.approx(2 / 3)


async def test_advisory_scorer_is_reported_but_does_not_fail_the_case(write_experiment, tmp_path):
    scorers = [{"type": "exact_match"}, {"type": "length", "max_chars": 3, "required": False}]
    rows = await make_runner(write_experiment({"scorers": scorers}), tmp_path).run()
    assert rows[0].passed and rows[0].score == 0.5
    assert rows[0].scores[1].model_dump()["required"] is False and not rows[0].scores[1].passed


async def test_on_cell_is_called_for_every_cell(write_experiment, tmp_path):
    seen = []
    await make_runner(write_experiment(), tmp_path, on_cell=seen.append).run()
    assert len(seen) == 6


async def test_missing_api_key_stops_before_any_call(write_experiment, tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    mock = MockProvider()
    models = [{"id": "m1", "provider": "mock"},
              {"id": "real", "provider": "openai_compatible", "base_url": "https://llm.example/v1", "api_key_env": "NOPE_KEY"}]
    runner = make_runner(write_experiment({"models": models}), tmp_path, providers={"m1": mock})
    with pytest.raises(ConfigError, match="NOPE_KEY is not set"):
        await runner.run()
    assert mock.calls == []


def test_plan_counts_calls_and_cache_hits(write_experiment, tmp_path):
    path = write_experiment()
    plan = make_runner(path, tmp_path, cache=True).plan()
    assert (plan.total, plan.cached, plan.to_call) == (6, 0, 6)

    asyncio.run(make_runner(path, tmp_path, cache=True, options=RunOptions(prompt_ids=["p1"])).run())
    plan = make_runner(path, tmp_path, cache=True).plan()
    assert (plan.total, plan.cached, plan.to_call) == (6, 3, 3)
    assert plan.by_model == {"m1": {"total": 6, "cached": 3}}


# --- llm_judge through the runner ---

def judged_experiment(judge_settings=None, **run) -> dict:
    return {
        "scorers": [{"type": "llm_judge", "rubric": "Is it one word?", "judge": "j1"}],
        "judges": [{"id": "j1", "provider": "mock", "params": {"max_tokens": 50},
                    "mock": judge_settings or {"default": "reply", "value": JUDGE_PASS}}],
        "run": {"retry_base_delay": 0, **run},
    }


async def test_judge_runs_through_the_runner(write_experiment, tmp_path):
    judge = MockProvider(settings={"default": "reply", "value": JUDGE_PASS})
    rows = await make_runner(write_experiment(judged_experiment()), tmp_path, providers={"j1": judge}).run()

    assert all(r.passed and r.scores[0].reason == "judge said pass: fine" for r in rows)
    assert rows[0].judge.calls == 1 and rows[0].judge.cached_calls == 0
    assert rows[0].judge.prompt_tokens > 0 and rows[0].judge.completion_tokens == 4
    assert rows[0].judge.cost_usd is None  # no price table
    request = judge.calls[0]
    assert request.params == {"temperature": 0, "max_tokens": 50}
    assert "Classify: great product" in request.prompt and "=== MODEL OUTPUT ===\npositive" in request.prompt
    assert request.metadata == {}  # the judge is not handed the expected answer through the back door


async def test_judge_at_concurrency_one_does_not_deadlock(write_experiment, tmp_path):
    runner = make_runner(write_experiment(judged_experiment(concurrency=1)), tmp_path)
    rows = await asyncio.wait_for(runner.run(), timeout=10)
    assert len(rows) == 6 and all(r.passed for r in rows)


async def test_unusable_judge_reply_is_a_scorer_error(write_experiment, tmp_path):
    settings = {"default": "reply", "value": "Looks good to me!"}
    rows = await make_runner(write_experiment(judged_experiment(settings)), tmp_path).run()
    assert all(not r.passed and r.scores[0].error and r.status == "ok" for r in rows)


async def test_failing_judge_call_is_a_scorer_error_not_a_crash(write_experiment, tmp_path):
    settings = {"rules": [{"behave": "error", "kind": "auth"}]}
    rows = await make_runner(write_experiment(judged_experiment(settings)), tmp_path).run()
    assert rows[0].status == "ok" and rows[0].scores[0].error
    assert "judge call failed (auth_error)" in rows[0].scores[0].reason


async def test_custom_provider_response_model_name_is_recorded(write_experiment, tmp_path):
    class Renaming(MockProvider):
        async def complete(self, request):
            response: CompletionResponse = await super().complete(request)
            response.model = "served-model-2026"
            return response

    rows = await make_runner(write_experiment(), tmp_path, providers={"m1": Renaming()}).run()
    assert rows[0].model == "served-model-2026" and rows[0].model_id == "m1"
