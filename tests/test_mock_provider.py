import json

import pytest

from promptlab.config import ModelConfig
from promptlab.errors import (
    ConfigError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
)
from promptlab.providers import CompletionRequest, MockProvider, build_provider

EXPECTED = {"order_id": "ORD-1", "category": "refund", "amount": 49.99}


def request(prompt="Extract: hello", expected=EXPECTED, **meta) -> CompletionRequest:
    metadata = {"case_id": meta.pop("case_id", "c1"), "expected": expected, "tags": meta.pop("tags", []), **meta}
    return CompletionRequest(prompt=prompt, system=meta.get("system"), metadata=metadata)


async def reply(settings: dict, req: CompletionRequest | None = None) -> str:
    return (await MockProvider(settings=settings).complete(req or request())).text


async def test_default_is_the_correct_answer():
    assert json.loads(await reply({})) == EXPECTED
    assert await reply({}, request(expected="positive")) == "positive"


async def test_answer_field_picks_one_field():
    assert await reply({"answer_field": "category"}) == "refund"


@pytest.mark.parametrize("rule, check", [
    ({"behave": "reply", "value": "hello"}, lambda t: t == "hello"),
    ({"behave": "wrong"}, lambda t: t == "I'm not sure."),
    ({"behave": "wrong", "value": {"order_id": None}}, lambda t: json.loads(t) == {"order_id": None}),
    ({"behave": "wrong_field", "set": {"category": "complaint"}},
     lambda t: json.loads(t) == {**EXPECTED, "category": "complaint"}),
    ({"behave": "drop_field", "field": "amount"}, lambda t: "amount" not in json.loads(t)),
    ({"behave": "number_as_string", "field": "amount"}, lambda t: json.loads(t)["amount"] == "$49.99"),
    ({"behave": "prose_wrapped"}, lambda t: t.startswith("Sure!") and json.dumps(EXPECTED) in t),
    ({"behave": "fenced"}, lambda t: t.startswith("```json\n") and t.endswith("\n```")),
    ({"behave": "empty"}, lambda t: t == ""),
])
async def test_behaviours(rule, check):
    assert check(await reply({"rules": [rule]}))


async def test_invalid_json_is_really_invalid():
    text = await reply({"rules": [{"behave": "invalid_json"}]})
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


async def test_change_case():
    settings = {"rules": [{"behave": "change_case", "to": "title"}]}
    assert await reply(settings, request(expected="negative")) == "Negative"


async def test_first_matching_rule_wins():
    settings = {"rules": [
        {"when": {"tags": ["hard"]}, "behave": "reply", "value": "first"},
        {"behave": "reply", "value": "second"},
    ]}
    assert await reply(settings, request(tags=["hard"])) == "first"
    assert await reply(settings, request(tags=["easy"])) == "second"


async def test_prompt_conditions_use_the_template_not_the_input():
    settings = {"rules": [{"when": {"prompt_lacks": "JSON"}, "behave": "prose_wrapped"}]}
    bare = request(prompt="Extract: please send JSON", prompt_template="Extract: {email}")
    strict = request(prompt="Return only JSON. Extract: hi", prompt_template="Return only JSON. Extract: {email}")
    assert (await reply(settings, bare)).startswith("Sure!")       # "JSON" is only in the email
    assert json.loads(await reply(settings, strict)) == EXPECTED

    matches = {"rules": [{"when": {"prompt_matches": "only json"}, "behave": "empty"}]}
    assert await reply(matches, strict) == ""
    assert await reply(matches, bare) != ""


async def test_input_and_case_id_conditions():
    settings = {"rules": [
        {"when": {"input_matches": "refund"}, "behave": "reply", "value": "by input"},
        {"when": {"case_ids": ["c7"]}, "behave": "reply", "value": "by id"},
    ]}
    assert await reply(settings, request(vars={"email": "I want a REFUND"})) == "by input"
    assert await reply(settings, request(case_id="c7", vars={"email": "hi"})) == "by id"
    assert json.loads(await reply(settings, request(case_id="c8", vars={"email": "hi"}))) == EXPECTED


async def test_rate_is_deterministic_and_roughly_proportional():
    settings = {"rules": [{"name": "half", "when": {"rate": 0.5}, "behave": "empty"}]}
    ids = [f"case-{i}" for i in range(200)]

    async def affected() -> list[str]:
        return [i for i in ids if await reply(settings, request(case_id=i)) == ""]

    first = await affected()
    assert first == await affected()  # same cases every time
    assert 70 <= len(first) <= 130    # about half of them

    never = {"rules": [{"when": {"rate": 0}, "behave": "empty"}]}
    always = {"rules": [{"when": {"rate": 1}, "behave": "empty"}]}
    assert await reply(never) != "" and await reply(always) == ""


@pytest.mark.parametrize("kind, error", [
    ("server", ProviderServerError), ("rate_limit", ProviderRateLimitError),
    ("auth", ProviderAuthError), ("bad_request", ProviderRequestError),
])
async def test_error_behaviour(kind, error):
    provider = MockProvider(settings={"rules": [{"behave": "error", "kind": kind}]})
    with pytest.raises(error):
        await provider.complete(request())


async def test_error_times_then_falls_through():
    provider = MockProvider(settings={"rules": [{"behave": "error", "times": 2}]})
    for _ in range(2):
        with pytest.raises(ProviderServerError):
            await provider.complete(request())
    assert json.loads((await provider.complete(request())).text) == EXPECTED


async def test_correct_without_expected_is_an_error():
    with pytest.raises(ProviderRequestError, match="expected answer"):
        await MockProvider().complete(CompletionRequest(prompt="hi"))


async def test_reports_synthetic_usage_and_records_calls():
    provider = MockProvider(model="mock-x", settings={"default": "reply", "value": "one two three"})
    response = await provider.complete(CompletionRequest(prompt="four words in prompt", system="sys"))
    assert response.usage == {"prompt_tokens": 5, "completion_tokens": 3}
    assert response.model == "mock-x" and response.latency_ms >= 0
    assert len(provider.calls) == 1


async def test_responder_hook_overrides_rules():
    provider = MockProvider(responder=lambda req: "scripted" if "x" in req.prompt else None)
    assert (await provider.complete(request(prompt="x"))).text == "scripted"
    assert json.loads((await provider.complete(request(prompt="y"))).text) == EXPECTED


def test_bad_settings_are_config_errors():
    with pytest.raises(ConfigError, match="mock settings"):
        MockProvider(settings={"rules": [{"behave": "sing"}]})
    with pytest.raises(ConfigError, match="invalid regex"):
        MockProvider(settings={"rules": [{"when": {"prompt_lacks": "("}, "behave": "empty"}]})
    with pytest.raises(ConfigError, match="mock settings"):
        build_provider(ModelConfig(id="m", provider="mock", mock={"defualt": "correct"}))


def test_cache_key_covers_rules_and_expected_answer():
    plain, ruled = MockProvider(), MockProvider(settings={"rules": [{"behave": "empty"}]})
    assert plain.cache_key_parts(request()) != ruled.cache_key_parts(request())
    assert plain.cache_key_parts(request()) != plain.cache_key_parts(request(expected="other"))
    assert plain.cache_key_parts(request()) == MockProvider().cache_key_parts(request())
