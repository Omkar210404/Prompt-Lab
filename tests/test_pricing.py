"""Cost estimation. An unknown cost must stay unknown, never become zero."""

import pytest

from promptlab.cache import ResponseCache
from promptlab.config import ModelConfig, load_experiment
from promptlab.errors import ConfigError
from promptlab.pricing import Price, estimate_cost, load_prices, price_for
from promptlab.providers import CompletionResponse, MockProvider
from promptlab.runner import Runner
from promptlab.summary import summarize

PRICE = Price(input=0.10, output=0.40)


class FixedUsage(MockProvider):
    """Mock that reports a fixed token count, or none at all."""

    def __init__(self, usage, **kwargs):
        super().__init__(**kwargs)
        self._usage = usage

    async def complete(self, request):
        response: CompletionResponse = await super().complete(request)
        response.usage = self._usage
        return response


def test_cost_is_tokens_times_price_per_million():
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}
    assert estimate_cost(usage, PRICE) == pytest.approx((1000 * 0.10 + 500 * 0.40) / 1_000_000)
    assert estimate_cost(usage, PRICE) == pytest.approx(0.0003)
    assert estimate_cost({"prompt_tokens": 1_000_000, "completion_tokens": 0}, PRICE) == pytest.approx(0.10)


def test_free_model_costs_zero_which_is_not_the_same_as_unknown():
    assert estimate_cost({"prompt_tokens": 10, "completion_tokens": 10}, Price(input=0, output=0)) == 0.0


@pytest.mark.parametrize("usage, price", [
    (None, PRICE),
    ({"prompt_tokens": 10, "completion_tokens": 5}, None),
    ({"prompt_tokens": 10}, PRICE),
    ({}, PRICE),
])
def test_missing_usage_or_price_gives_none(usage, price):
    assert estimate_cost(usage, price) is None


def test_load_prices(tmp_path):
    path = tmp_path / "prices.yaml"
    path.write_text(
        "model-a: {input: 0.10, output: 0.40, note: fictional}\n"
        "vendor/model-b:\n  input: 3\n  output: 15\n  source: https://example.com/pricing\n",
        encoding="utf-8",
    )
    prices = load_prices(path)
    assert prices["model-a"].input == 0.10 and prices["model-a"].output == 0.40
    assert prices["vendor/model-b"].output == 15
    assert prices["model-a"].model_extra == {"note": "fictional"}
    assert load_prices(None) == {}


@pytest.mark.parametrize("content, message", [
    ("model-a: {input: 0.1}", "price for 'model-a': output"),
    ("model-a: {input: -1, output: 1}", "price for 'model-a': input"),
    ("model-a: cheap", "price for 'model-a'"),
    ("- just\n- a list", "expected a mapping"),
    ("model-a: {input: [", "invalid YAML"),
])
def test_bad_price_files(tmp_path, content, message):
    path = tmp_path / "prices.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_prices(path)


def test_missing_price_file(tmp_path):
    with pytest.raises(ConfigError, match="prices file not found"):
        load_prices(tmp_path / "nope.yaml")


def test_price_lookup_by_model_name_then_id():
    by_name = ModelConfig(id="fast", provider="mock", model="vendor/model-b")
    assert price_for({"vendor/model-b": PRICE}, by_name) is PRICE
    assert price_for({"fast": PRICE}, by_name) is PRICE
    assert price_for({"other": PRICE}, by_name) is None


# --- through the runner and the summary ---

async def run(path, tmp_path, provider, prices, cache=False):
    runner = Runner(load_experiment(path), ResponseCache(tmp_path / "cache", enabled=cache),
                    prices=prices, providers={"m1": provider})
    return [row.model_dump() for row in await runner.run()]


async def test_rows_carry_cost_from_reported_tokens(write_experiment, tmp_path):
    provider = FixedUsage({"prompt_tokens": 2000, "completion_tokens": 100})
    rows = await run(write_experiment(), tmp_path, provider, {"m1": PRICE})
    per_call = (2000 * 0.10 + 100 * 0.40) / 1_000_000
    assert all(row["cost_usd"] == pytest.approx(per_call) for row in rows)

    totals = summarize(rows)["totals"]
    assert totals["cost_usd"] == pytest.approx(6 * per_call)
    assert totals["cost_usd_spent"] == pytest.approx(6 * per_call)
    assert totals["prompt_tokens"] == 12000 and totals["completion_tokens"] == 600


async def test_no_price_entry_means_no_cost(write_experiment, tmp_path):
    rows = await run(write_experiment(), tmp_path, MockProvider(), {"some-other-model": PRICE})
    assert all(row["cost_usd"] is None for row in rows)
    totals = summarize(rows)["totals"]
    assert totals["cost_usd"] is None and totals["cost_usd_spent"] is None
    assert totals["prompt_tokens"] > 0  # tokens were reported; only the price is unknown


async def test_no_reported_usage_means_no_tokens_and_no_cost(write_experiment, tmp_path):
    rows = await run(write_experiment(), tmp_path, FixedUsage(None), {"m1": PRICE})
    assert all(row["usage"] is None and row["cost_usd"] is None for row in rows)
    totals = summarize(rows)["totals"]
    assert totals["prompt_tokens"] is None and totals["completion_tokens"] is None
    assert totals["cost_usd"] is None


async def test_cached_calls_are_excluded_from_spend(write_experiment, tmp_path):
    path = write_experiment()
    provider = FixedUsage({"prompt_tokens": 1000, "completion_tokens": 1000})
    per_call = (1000 * 0.10 + 1000 * 0.40) / 1_000_000

    first = summarize(await run(path, tmp_path, provider, {"m1": PRICE}, cache=True))["totals"]
    assert first["cost_usd_spent"] == pytest.approx(6 * per_call) and first["cached"] == 0

    second = summarize(await run(path, tmp_path, provider, {"m1": PRICE}, cache=True))["totals"]
    assert second["cached"] == 6
    assert second["cost_usd"] == pytest.approx(6 * per_call)  # what these outputs cost to produce
    assert second["cost_usd_spent"] == 0.0                    # what this run added


async def test_error_rows_have_no_cost(write_experiment, tmp_path):
    provider = MockProvider(settings={"rules": [{"when": {"case_ids": ["c1"]}, "behave": "error", "kind": "auth"}]})
    rows = await run(write_experiment(), tmp_path, provider, {"m1": PRICE})
    assert [row["cost_usd"] is None for row in rows] == [True, False, False] * 2


async def test_judge_cost_is_tracked_separately(write_experiment, tmp_path):
    path = write_experiment({
        "scorers": [{"type": "llm_judge", "rubric": "One word?", "judge": "j1"}],
        "judges": [{"id": "j1", "provider": "mock", "model": "judge-model"}],
    })
    judge = FixedUsage({"prompt_tokens": 500, "completion_tokens": 50},
                       settings={"default": "reply", "value": '{"reasoning": "ok", "verdict": "pass"}'})
    runner = Runner(load_experiment(path), ResponseCache(tmp_path / "cache", enabled=False),
                    prices={"judge-model": Price(input=1.0, output=2.0)},
                    providers={"m1": MockProvider(), "j1": judge})
    rows = [row.model_dump() for row in await runner.run()]

    per_judge_call = (500 * 1.0 + 50 * 2.0) / 1_000_000
    assert rows[0]["judge"]["cost_usd"] == pytest.approx(per_judge_call)
    assert rows[0]["cost_usd"] is None  # the model under test has no price entry
    totals = summarize(rows)["totals"]
    assert totals["judge"] == {"calls": 6, "cached_calls": 0, "prompt_tokens": 3000,
                               "completion_tokens": 300, "cost_usd": pytest.approx(6 * per_judge_call)}
    assert totals["cost_usd"] is None
