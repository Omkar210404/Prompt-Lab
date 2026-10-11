"""OpenAI-compatible provider and the retry loop, exercised through httpx.MockTransport (no network)."""

import asyncio
import json

import httpx
import pytest

from promptlab.config import ModelConfig
from promptlab.errors import (
    ConfigError,
    MalformedResponseError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
    ProviderTimeoutError,
)
from promptlab.providers import (
    CompletionRequest,
    MockProvider,
    OpenAICompatibleProvider,
    build_provider,
    complete_with_retry,
)

REQUEST = CompletionRequest(prompt="Say hi", system="Be brief", params={"temperature": 0, "max_tokens": 20},
                            metadata={"expected": "never sent"})


def completion(content="hi", usage=None, **extra) -> dict:
    body = {"model": "served-model", "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    if usage is not None:
        body["usage"] = usage
    return {**body, **extra}


def provider_with(handler, api_key="sk-test") -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        model="test-model", base_url="https://llm.example/v1/", api_key=api_key,
        transport=httpx.MockTransport(handler),
    )


async def test_success_sends_the_right_request_and_reads_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion("hello", {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}))

    response = await provider_with(handler).complete(REQUEST)

    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"] == {
        "model": "test-model", "temperature": 0, "max_tokens": 20,
        "messages": [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Say hi"}],
    }
    assert response.text == "hello"
    assert response.model == "served-model"
    assert response.usage == {"prompt_tokens": 12, "completion_tokens": 3}
    assert response.finish_reason == "stop"
    assert response.latency_ms >= 0


async def test_missing_usage_stays_none():
    response = await provider_with(lambda r: httpx.Response(200, json=completion())).complete(REQUEST)
    assert response.usage is None
    partial = provider_with(lambda r: httpx.Response(200, json=completion(usage={"prompt_tokens": 5})))
    assert (await partial.complete(REQUEST)).usage is None


async def test_null_content_becomes_an_empty_output():
    response = await provider_with(lambda r: httpx.Response(200, json=completion(None))).complete(REQUEST)
    assert response.text == ""


async def test_no_key_means_no_auth_header():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=completion())

    await provider_with(handler, api_key=None).complete(REQUEST)
    assert seen["auth"] is None


@pytest.mark.parametrize("status, error, retryable", [
    (401, ProviderAuthError, False),
    (403, ProviderAuthError, False),
    (400, ProviderRequestError, False),
    (404, ProviderRequestError, False),
    (429, ProviderRateLimitError, True),
    (500, ProviderServerError, True),
    (503, ProviderServerError, True),
])
async def test_http_errors_map_to_typed_errors(status, error, retryable):
    handler = lambda r: httpx.Response(status, json={"error": {"message": "model exploded"}})  # noqa: E731
    with pytest.raises(error, match=f"HTTP {status}: model exploded") as info:
        await provider_with(handler).complete(REQUEST)
    assert info.value.retryable is retryable


async def test_rate_limit_carries_retry_after():
    handler = lambda r: httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow down"})  # noqa: E731
    with pytest.raises(ProviderRateLimitError) as info:
        await provider_with(handler).complete(REQUEST)
    assert info.value.retry_after == 7.0


async def test_timeout_and_connection_errors():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderTimeoutError):
        await provider_with(timeout).complete(REQUEST)
    with pytest.raises(ProviderServerError, match="connection error"):
        await provider_with(refused).complete(REQUEST)


@pytest.mark.parametrize("response", [
    httpx.Response(200, text="<html>not json</html>"),
    httpx.Response(200, json={"choices": []}),
    httpx.Response(200, json={"unexpected": True}),
    httpx.Response(200, json=completion({"not": "text"})),
])
async def test_malformed_bodies(response):
    with pytest.raises(MalformedResponseError):
        await provider_with(lambda r: response).complete(REQUEST)


async def test_error_text_never_contains_the_key():
    handler = lambda r: httpx.Response(401, json={"error": {"message": "Invalid API Key"}})  # noqa: E731
    with pytest.raises(ProviderAuthError) as info:
        await provider_with(handler, api_key="sk-super-secret").complete(REQUEST)
    assert "sk-super-secret" not in str(info.value)


# --- building from config: keys come only from the environment ---

def test_key_is_read_from_the_named_env_var(monkeypatch):
    monkeypatch.setenv("MY_LLM_KEY", "sk-from-env")
    config = ModelConfig(id="m", provider="openai_compatible", model="x",
                         base_url="https://llm.example/v1", api_key_env="MY_LLM_KEY")
    provider = build_provider(config)
    assert provider._client.headers["authorization"] == "Bearer sk-from-env"
    assert provider.cache_key_parts(REQUEST)["base_url"] == "https://llm.example/v1"
    assert "sk-from-env" not in json.dumps(provider.cache_key_parts(REQUEST))


def test_missing_key_is_a_config_error_before_any_call(monkeypatch):
    monkeypatch.delenv("MY_LLM_KEY", raising=False)
    config = ModelConfig(id="m", provider="openai_compatible", base_url="https://llm.example/v1",
                         api_key_env="MY_LLM_KEY")
    with pytest.raises(ConfigError, match="environment variable MY_LLM_KEY is not set"):
        build_provider(config)


def test_missing_base_url_is_a_config_error():
    with pytest.raises(ConfigError, match="needs a base_url"):
        build_provider(ModelConfig(id="m", provider="openai_compatible"))


# --- retry loop ---

class Recorder:
    """Stands in for asyncio.sleep so retries are instant and the delays can be checked."""

    def __init__(self):
        self.delays = []

    async def __call__(self, seconds):
        self.delays.append(seconds)


async def test_retries_then_succeeds():
    statuses = iter([500, 429, 200])

    def handler(request):
        status = next(statuses)
        if status == 200:
            return httpx.Response(200, json=completion("finally"))
        return httpx.Response(status, json={"error": "try later"})

    sleep = Recorder()
    response, attempts = await complete_with_retry(
        provider_with(handler), REQUEST, timeout_seconds=5, max_retries=2, base_delay=1.0, sleep=sleep)
    assert response.text == "finally" and attempts == 3
    assert sleep.delays == [1.0, 2.0]  # exponential backoff


async def test_retry_waits_for_retry_after():
    responses = iter([httpx.Response(429, headers={"retry-after": "9"}, json={}), httpx.Response(200, json=completion())])
    sleep = Recorder()
    await complete_with_retry(provider_with(lambda r: next(responses)), REQUEST,
                              timeout_seconds=5, max_retries=2, base_delay=1.0, sleep=sleep)
    assert sleep.delays == [9.0]


async def test_gives_up_after_max_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json={})

    with pytest.raises(ProviderServerError) as info:
        await complete_with_retry(provider_with(handler), REQUEST, timeout_seconds=5,
                                  max_retries=2, base_delay=0, sleep=Recorder())
    assert len(calls) == 3 and info.value.attempts == 3


async def test_auth_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={})

    with pytest.raises(ProviderAuthError) as info:
        await complete_with_retry(provider_with(handler), REQUEST, timeout_seconds=5,
                                  max_retries=3, base_delay=0, sleep=Recorder())
    assert len(calls) == 1 and info.value.attempts == 1


async def test_slow_call_times_out():
    class Slow(MockProvider):
        async def complete(self, request):
            await asyncio.sleep(5)

    with pytest.raises(ProviderTimeoutError, match="no response within 0.05s") as info:
        await complete_with_retry(Slow(), REQUEST, timeout_seconds=0.05, max_retries=1,
                                  base_delay=0, sleep=Recorder())
    assert info.value.attempts == 2
