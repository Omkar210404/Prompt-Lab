"""Response cache: a re-run must not repeat calls that were already paid for."""

from promptlab.cache import ResponseCache, cache_key
from promptlab.config import load_experiment
from promptlab.providers import CompletionResponse, MockProvider
from promptlab.runner import Runner


async def run_with(path, tmp_path, provider, **cache_options):
    cache = ResponseCache(tmp_path / "cache", **cache_options)
    return await Runner(load_experiment(path), cache, providers={"m1": provider}).run()


def cache_files(tmp_path) -> list:
    return sorted((tmp_path / "cache").rglob("*.json"))


# --- the cache itself ---

def test_key_is_stable_and_order_independent():
    assert cache_key({"a": 1, "b": {"x": 1, "y": 2}}) == cache_key({"b": {"y": 2, "x": 1}, "a": 1})
    assert cache_key({"a": 1}) != cache_key({"a": 2})
    assert len(cache_key({})) == 64


def test_round_trip(tmp_path):
    cache = ResponseCache(tmp_path)
    response = CompletionResponse(text="héllo", model="m", latency_ms=12.5,
                                  usage={"prompt_tokens": 3, "completion_tokens": 1}, finish_reason="stop")
    assert cache.get("ab12") is None and not cache.has("ab12")
    cache.put("ab12", response)
    assert cache.get("ab12") == response and cache.has("ab12")
    assert (tmp_path / "ab" / "ab12.json").is_file()
    assert not list(tmp_path.rglob("*.tmp"))


def test_disabled_cache_neither_reads_nor_writes(tmp_path):
    ResponseCache(tmp_path).put("ab12", CompletionResponse(text="x", model="m", latency_ms=1))
    disabled = ResponseCache(tmp_path, enabled=False)
    assert disabled.get("ab12") is None
    disabled.put("cd34", CompletionResponse(text="y", model="m", latency_ms=1))
    assert not (tmp_path / "cd").exists()


def test_corrupt_entry_is_a_miss(tmp_path):
    cache = ResponseCache(tmp_path)
    (tmp_path / "ab").mkdir()
    (tmp_path / "ab" / "ab12.json").write_text("{half written", encoding="utf-8")
    assert cache.get("ab12") is None


# --- through the runner ---

async def test_second_run_makes_no_provider_calls(write_experiment, tmp_path):
    path, provider = write_experiment(), MockProvider()
    first = await run_with(path, tmp_path, provider)
    assert len(provider.calls) == 6 and not any(r.cached for r in first)
    assert len(cache_files(tmp_path)) == 6

    second = await run_with(path, tmp_path, provider)
    assert len(provider.calls) == 6  # nothing new
    assert all(r.cached and r.attempts == 0 for r in second)
    assert [r.output for r in second] == [r.output for r in first]
    assert [r.latency_ms for r in second] == [r.latency_ms for r in first]  # latency of the original call
    assert [r.usage for r in second] == [r.usage for r in first]
    assert all(r.passed for r in second)  # cached outputs are still scored


async def test_changed_prompt_or_params_is_a_miss(write_experiment, tmp_path):
    provider = MockProvider()
    await run_with(write_experiment(), tmp_path, provider)

    reworded = {"prompts": [{"id": "p1", "template": "Classify this: {text}"}, {"id": "p2", "template": "Reply with one word. Classify: {text}"}]}
    await run_with(write_experiment(reworded), tmp_path, provider)
    assert len(provider.calls) == 6 + 3  # only the reworded prompt is called again

    with_params = {"models": [{"id": "m1", "provider": "mock", "params": {"temperature": 0.7}}]}
    await run_with(write_experiment(with_params), tmp_path, provider)
    assert len(provider.calls) == 9 + 6


async def test_different_model_is_a_miss(write_experiment, tmp_path):
    path = write_experiment()
    await run_with(path, tmp_path, MockProvider(model="model-a"))
    other = MockProvider(model="model-b")
    await run_with(path, tmp_path, other)
    assert len(other.calls) == 6


async def test_editing_a_mock_rule_is_a_miss(write_experiment, tmp_path):
    path = write_experiment()
    await run_with(path, tmp_path, MockProvider())
    edited = MockProvider(settings={"rules": [{"when": {"tags": ["hard"]}, "behave": "empty"}]})
    rows = await run_with(path, tmp_path, edited)
    assert len(edited.calls) == 6 and not any(r.cached for r in rows)
    assert [r.output for r in rows if r.case_id == "c3"] == ["", ""]


async def test_errors_are_not_cached(write_experiment, tmp_path):
    path = write_experiment()
    failing = MockProvider(settings={"rules": [{"behave": "error", "kind": "bad_request"}]})
    await run_with(path, tmp_path, failing)
    assert cache_files(tmp_path) == []
    await run_with(path, tmp_path, failing)
    assert len(failing.calls) == 12


async def test_no_cache_calls_again_and_writes_nothing(write_experiment, tmp_path):
    path, provider = write_experiment(), MockProvider()
    await run_with(path, tmp_path, provider, enabled=False)
    rows = await run_with(path, tmp_path, provider, enabled=False)
    assert len(provider.calls) == 12 and not any(r.cached for r in rows)
    assert cache_files(tmp_path) == []


async def test_refresh_calls_again_and_overwrites(write_experiment, tmp_path):
    path = write_experiment()
    await run_with(path, tmp_path, MockProvider())
    replacement = MockProvider(responder=lambda request: "fresh answer")
    rows = await run_with(path, tmp_path, replacement, refresh=True)
    assert len(replacement.calls) == 6 and not any(r.cached for r in rows)

    after = await run_with(path, tmp_path, MockProvider())
    assert all(r.cached and r.output == "fresh answer" for r in after)


async def test_scoring_is_not_cached(write_experiment, tmp_path):
    provider = MockProvider(settings={"rules": [{"behave": "change_case", "to": "title"}]})
    strict = await run_with(write_experiment(), tmp_path, provider)
    assert not any(r.passed for r in strict)  # "Positive" != "positive"

    lenient = await run_with(write_experiment({"scorers": [{"type": "exact_match", "ignore_case": True}]}), tmp_path, provider)
    assert len(provider.calls) == 6           # no new calls
    assert all(r.cached and r.passed for r in lenient)  # same outputs, new verdicts


async def test_judge_calls_are_cached_too(write_experiment, tmp_path):
    judge = MockProvider(settings={"default": "reply", "value": '{"reasoning": "ok", "verdict": "pass"}'})
    path = write_experiment({
        "scorers": [{"type": "llm_judge", "rubric": "One word?", "judge": "j1"}],
        "judges": [{"id": "j1", "provider": "mock"}],
    })
    cache = ResponseCache(tmp_path / "cache")
    providers = {"m1": MockProvider(), "j1": judge}
    await Runner(load_experiment(path), cache, providers=providers).run()
    rows = await Runner(load_experiment(path), cache, providers=providers).run()
    assert len(judge.calls) == 6
    assert all(r.judge.calls == 1 and r.judge.cached_calls == 1 for r in rows)
