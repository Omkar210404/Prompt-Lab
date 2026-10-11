"""Run the prompt x model x test case matrix: concurrency, timeouts, retries, caching and scoring.

Each combination is one cell. A cell that fails is recorded as an error and the run carries on;
one bad call never throws away the rest of a run.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Literal

from pydantic import BaseModel

from promptlab.cache import ResponseCache, cache_key
from promptlab.config import LoadedExperiment, ModelConfig, PromptConfig, TestCase, render_template
from promptlab.errors import ConfigError, ProviderError
from promptlab.pricing import Price, estimate_cost, price_for
from promptlab.providers import CompletionRequest, CompletionResponse, Provider, build_provider, complete_with_retry
from promptlab.scorers import ScoreContext, ScoreResult, Scorer, build_scorer
from promptlab.scorers.llm_judge import LLMJudge

# A judge should give the same verdict twice, so it runs at temperature 0 unless its config says otherwise.
JUDGE_DEFAULT_PARAMS = {"temperature": 0}


class ScoreRow(BaseModel):
    scorer: str
    type: str
    passed: bool
    score: float
    reason: str
    required: bool = True
    error: bool = False


class JudgeUsage(BaseModel):
    """Judge calls made while scoring one cell. Kept apart from the usage of the model under test."""

    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None


class CellResult(BaseModel):
    """One line of results.jsonl."""

    case_id: str
    prompt_id: str
    model_id: str
    provider: str
    model: str
    status: Literal["ok", "error"]
    output: str | None = None
    error: dict[str, str] | None = None
    latency_ms: float | None = None
    # True when the output came from the cache. latency_ms and usage are then those of the original call.
    cached: bool = False
    attempts: int = 0
    usage: dict[str, int] | None = None
    cost_usd: float | None = None
    finish_reason: str | None = None
    passed: bool = False
    score: float = 0.0
    scores: list[ScoreRow] = []
    judge: JudgeUsage | None = None


@dataclass
class RunOptions:
    """Narrow a run without editing the experiment file."""

    prompt_ids: list[str] = field(default_factory=list)
    model_ids: list[str] = field(default_factory=list)
    limit: int | None = None
    concurrency: int | None = None


@dataclass
class Plan:
    """What a run would do, worked out without calling any model."""

    total: int = 0
    cached: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def to_call(self) -> int:
        return self.total - self.cached


@dataclass
class _Call:
    response: CompletionResponse
    cached: bool
    attempts: int


def _select(items: list, wanted: list[str], label: str) -> list:
    if not wanted:
        return items
    known = {item.id for item in items}
    unknown = [i for i in wanted if i not in known]
    if unknown:
        raise ConfigError(f"unknown {label} id: {', '.join(unknown)} (available: {', '.join(sorted(known))})")
    return [item for item in items if item.id in wanted]


def _name_scorers(scorers: list[Scorer]) -> list[Scorer]:
    """Give scorers that share a name distinct labels: contains, contains#2, ..."""
    seen: dict[str, int] = {}
    for scorer in scorers:
        seen[scorer.name] = seen.get(scorer.name, 0) + 1
        if seen[scorer.name] > 1:
            scorer.name = f"{scorer.name}#{seen[scorer.name]}"
    return scorers


class Runner:
    def __init__(
        self,
        loaded: LoadedExperiment,
        cache: ResponseCache,
        prices: dict[str, Price] | None = None,
        options: RunOptions | None = None,
        providers: dict[str, Provider] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_cell: Callable[[CellResult], None] | None = None,
    ):
        self.loaded = loaded
        self.cache = cache
        self.prices = prices or {}
        self.options = options or RunOptions()
        self._sleep = sleep
        self._on_cell = on_cell

        experiment = loaded.experiment
        self.run_config = experiment.run
        self.prompts: list[PromptConfig] = _select(experiment.prompts, self.options.prompt_ids, "prompt")
        self.models: list[ModelConfig] = _select(experiment.models, self.options.model_ids, "model")
        self.cases: list[TestCase] = loaded.cases[: self.options.limit] if self.options.limit else loaded.cases

        self._scorers: dict[str, list[Scorer]] = {
            case.id: _name_scorers(
                [build_scorer(cfg, loaded.base_dir) for cfg in [*experiment.scorers, *case.checks]]
            )
            for case in self.cases
        }
        judge_ids = {
            scorer.judge for scorers in self._scorers.values() for scorer in scorers
            if isinstance(scorer, LLMJudge)
        }
        self._configs: dict[str, ModelConfig] = {model.id: model for model in self.models}
        for judge_id in sorted(judge_ids):
            self._configs.setdefault(judge_id, experiment.judge(judge_id))

        # `providers` lets tests inject their own. Anything missing is built from the experiment file.
        self._providers: dict[str, Provider] = dict(providers or {})
        self._owned: list[Provider] = []
        self._semaphore: asyncio.Semaphore | None = None
        self._model_semaphores: dict[str, asyncio.Semaphore] = {}

    # --- setup ---

    def _ensure_providers(self) -> None:
        """Build every provider before the first call, so a missing API key stops the run at once."""
        for model_id, config in self._configs.items():
            if model_id not in self._providers:
                provider = build_provider(config)
                self._providers[model_id] = provider
                self._owned.append(provider)

    def cells(self) -> list[tuple[PromptConfig, ModelConfig, TestCase]]:
        return [(prompt, model, case) for prompt in self.prompts for model in self.models for case in self.cases]

    def _request(self, prompt: PromptConfig, model: ModelConfig, case: TestCase) -> CompletionRequest:
        return CompletionRequest(
            prompt=render_template(prompt.template, case.vars),
            system=render_template(prompt.system, case.vars) if prompt.system else None,
            params=model.params,
            metadata={
                "case_id": case.id,
                "tags": case.tags,
                "expected": case.expected,
                "vars": case.vars,
                "prompt_id": prompt.id,
                "prompt_template": prompt.template,
            },
        )

    def plan(self) -> Plan:
        """Count the calls a run would make and how many are already cached. Judge calls are not
        included: what the judge is asked depends on outputs that do not exist yet."""
        self._ensure_providers()
        plan = Plan()
        for prompt, model, case in self.cells():
            key = cache_key(self._providers[model.id].cache_key_parts(self._request(prompt, model, case)))
            counts = plan.by_model.setdefault(model.id, {"total": 0, "cached": 0})
            hit = self.cache.has(key)
            counts["total"] += 1
            counts["cached"] += hit
            plan.total += 1
            plan.cached += hit
        return plan

    # --- running ---

    async def run(self) -> list[CellResult]:
        self._ensure_providers()
        self._semaphore = asyncio.Semaphore(self.options.concurrency or self.run_config.concurrency)
        self._model_semaphores = {
            model_id: asyncio.Semaphore(config.concurrency)
            for model_id, config in self._configs.items() if config.concurrency
        }
        try:
            return list(await asyncio.gather(*(self._run_cell(*cell) for cell in self.cells())))
        finally:
            for provider in self._owned:
                await provider.aclose()

    async def _call(self, model_id: str, request: CompletionRequest) -> _Call:
        """One model call through the cache, the concurrency limits and the retry loop."""
        provider = self._providers[model_id]
        key = cache_key(provider.cache_key_parts(request))
        stored = self.cache.get(key)
        if stored is not None:
            return _Call(stored, cached=True, attempts=0)
        # The slot is held only for the call itself, never while scoring, so a judge call made
        # during scoring cannot deadlock against the cells waiting on it.
        async with self._semaphore, self._model_semaphores.get(model_id, contextlib.nullcontext()):
            response, attempts = await complete_with_retry(
                provider, request,
                timeout_seconds=self.run_config.timeout_seconds,
                max_retries=self.run_config.max_retries,
                base_delay=self.run_config.retry_base_delay,
                sleep=self._sleep,
            )
        self.cache.put(key, response)
        return _Call(response, cached=False, attempts=attempts)

    async def _run_cell(self, prompt: PromptConfig, model: ModelConfig, case: TestCase) -> CellResult:
        request = self._request(prompt, model, case)
        result = CellResult(
            case_id=case.id, prompt_id=prompt.id, model_id=model.id,
            provider=model.provider, model=model.model, status="ok",
        )
        try:
            call = await self._call(model.id, request)
        except ProviderError as exc:
            result.status = "error"
            result.error = {"type": exc.error_type, "message": str(exc)}
            result.attempts = exc.attempts
        except Exception as exc:  # a provider bug must not cost the rest of the run
            result.status = "error"
            result.error = {"type": "unexpected_error", "message": f"{type(exc).__name__}: {exc}"}
            result.attempts = 1
        else:
            response = call.response
            result.output = response.text
            result.model = response.model or model.model
            result.latency_ms = response.latency_ms
            result.cached = call.cached
            result.attempts = call.attempts
            result.usage = response.usage
            result.finish_reason = response.finish_reason
            result.cost_usd = estimate_cost(response.usage, price_for(self.prices, model))
            await self._score(result, request, case)

        if self._on_cell:
            self._on_cell(result)
        return result

    async def _score(self, result: CellResult, request: CompletionRequest, case: TestCase) -> None:
        judge_usage = JudgeUsage()

        async def ask_judge(judge_id: str, judge_request: CompletionRequest) -> CompletionResponse:
            config = self._configs[judge_id]
            call = await self._call(
                judge_id, replace(judge_request, params={**JUDGE_DEFAULT_PARAMS, **config.params})
            )
            judge_usage.calls += 1
            judge_usage.cached_calls += call.cached
            usage = call.response.usage
            if usage:
                judge_usage.prompt_tokens = (judge_usage.prompt_tokens or 0) + usage["prompt_tokens"]
                judge_usage.completion_tokens = (judge_usage.completion_tokens or 0) + usage["completion_tokens"]
            cost = estimate_cost(usage, price_for(self.prices, config))
            if cost is not None:
                judge_usage.cost_usd = (judge_usage.cost_usd or 0.0) + cost
            return call.response

        context = ScoreContext(prompt=request.prompt, system=request.system, complete=ask_judge)
        for scorer in self._scorers[case.id]:
            try:
                outcome = await scorer.ascore(result.output or "", case, context)
            except Exception as exc:  # a broken custom scorer must not lose outputs already paid for
                outcome = ScoreResult(False, 0.0, f"scorer crashed: {type(exc).__name__}: {exc}", error=True)
            result.scores.append(ScoreRow(
                scorer=scorer.name, type=scorer.type, passed=outcome.passed, score=outcome.score,
                reason=outcome.reason, required=scorer.required, error=outcome.error,
            ))

        required = [row for row in result.scores if row.required]
        result.passed = all(row.passed for row in required)
        result.score = sum(row.score for row in result.scores) / len(result.scores) if result.scores else 0.0
        if judge_usage.calls:
            result.judge = judge_usage


def run_sync(runner: Runner) -> list[CellResult]:
    return asyncio.run(runner.run())


