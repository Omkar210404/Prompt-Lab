"""Local web app: try an input, edit prompts, run the test set, browse and compare runs.

It is a thin layer over the same config, runner, storage and compare modules the CLI uses.
The server is meant for your own machine: it binds to 127.0.0.1 by default and refuses
requests that come from another site's pages.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from promptlab import __version__
from promptlab.cache import ResponseCache
from promptlab.compare import compare_runs
from promptlab.config import LoadedExperiment, TestCase, load_experiment, override_experiment
from promptlab.errors import ConfigError, PromptLabError, RunNotFoundError
from promptlab.pricing import load_prices
from promptlab.providers import build_provider
from promptlab.report.html import render_html
from promptlab.runner import Runner, RunOptions
from promptlab.storage import SUMMARY_FILE, list_runs, load_run, save_run, utc_now, validate_run_name

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
# A single typed input is not part of the test set, so it gets an id that cannot clash with one.
TRY_CASE_ID = "typed-input"
FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#1a1a19"/>'
    '<rect x="7" y="8" width="18" height="4" rx="2" fill="#3987e5"/>'
    '<rect x="7" y="14" width="12" height="4" rx="2" fill="#d95926"/>'
    '<rect x="7" y="20" width="15" height="4" rx="2" fill="#199e70"/></svg>'
)


class PromptIn(BaseModel):
    id: str = Field(min_length=1)
    template: str = Field(min_length=1)
    system: str | None = None


class Selection(BaseModel):
    """Which experiment, with which prompts (as edited in the browser) and which models."""

    experiment: str
    prompts: list[PromptIn] | None = None
    models: list[str] = Field(default_factory=list)
    use_cache: bool = True


class TryRequest(Selection):
    vars: dict[str, Any] = Field(default_factory=dict)
    expected: Any = None
    tags: list[str] = Field(default_factory=list)


class RunRequest(Selection):
    name: str | None = None


@dataclass
class Job:
    """A full test-set run in progress."""

    id: str
    total: int
    cached: int
    done: int = 0
    status: str = "running"      # running | done | error
    run_id: str | None = None
    error: str | None = None
    task: asyncio.Task | None = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "status": self.status, "done": self.done, "total": self.total,
                "cached": self.cached, "run_id": self.run_id, "error": self.error}


def _host_of(value: str) -> str:
    """'http://localhost:8321' or 'localhost:8321' -> 'localhost'."""
    host = value.split("://", 1)[-1].split("/", 1)[0]
    if host.startswith("["):  # IPv6 literal, such as [::1]:8321
        return host.split("]", 1)[0] + "]"
    return host.split(":", 1)[0]


def create_app(root: str | Path = ".", runs_dir: str | Path | None = None,
               cache_dir: str | Path | None = None) -> FastAPI:
    root = Path(root).resolve()
    runs_dir = Path(runs_dir) if runs_dir is not None else root / "runs"
    cache_dir = Path(cache_dir) if cache_dir is not None else root / ".promptlab" / "cache"
    jobs: dict[str, Job] = {}

    app = FastAPI(title="prompt-lab", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # This server can spend money on model calls, so only pages it served may talk to it.
        # A wrong Host means DNS rebinding; a foreign Origin means another site's page is calling.
        if _host_of(request.headers.get("host", "")) not in LOCAL_HOSTS:
            return JSONResponse({"detail": "prompt-lab only answers on localhost"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and _host_of(origin) not in LOCAL_HOSTS:
            return JSONResponse({"detail": "cross-site requests are refused"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(RunNotFoundError)
    async def not_found(_: Request, exc: RunNotFoundError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(PromptLabError)
    async def bad_request(_: Request, exc: PromptLabError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # --- helpers ---

    def experiment_path(experiment_id: str) -> Path:
        path = (root / experiment_id).resolve()
        if root not in path.parents or path.suffix not in (".yaml", ".yml") or not path.is_file():
            raise HTTPException(status_code=404, detail=f"experiment not found: {experiment_id}")
        return path

    def candidates() -> list[Path]:
        """Experiment files: YAML in the working folder and in its examples/ folder."""
        found = []
        for folder in (root, root / "examples"):
            if folder.is_dir():
                found += sorted(p for p in folder.iterdir() if p.suffix in (".yaml", ".yml") and p.is_file())
        return found

    def prepare(selection: Selection, cases: list[TestCase] | None = None) -> LoadedExperiment:
        loaded = load_experiment(experiment_path(selection.experiment))
        prompts = [p.model_dump() for p in selection.prompts] if selection.prompts is not None else None
        if prompts is not None or cases is not None:
            loaded = override_experiment(loaded, prompts=prompts, cases=cases)
        return loaded

    def runner_for(loaded: LoadedExperiment, selection: Selection) -> Runner:
        cache = ResponseCache(cache_dir, enabled=loaded.experiment.run.cache and selection.use_cache)
        options = RunOptions(model_ids=selection.models)
        return Runner(loaded, cache, prices=load_prices(loaded.prices_path), options=options)

    async def model_status(loaded: LoadedExperiment) -> list[dict[str, Any]]:
        """Each model with whether it can be called right now (for example, is its API key set?)."""
        models = []
        for config in loaded.experiment.models:
            problem = None
            try:
                await build_provider(config).aclose()
            except ConfigError as exc:
                problem = str(exc)
            models.append({"id": config.id, "provider": config.provider, "model": config.model,
                           "available": problem is None, "problem": problem})
        return models

    def run_dir(run_id: str) -> Path:
        """The folder of a saved run. Only exact names directly under the runs folder are accepted."""
        try:
            validate_run_name(run_id)
        except ConfigError:
            raise RunNotFoundError(f"no run named '{run_id}'") from None
        path = runs_dir / run_id
        if not (path / SUMMARY_FILE).is_file():
            raise RunNotFoundError(f"no run named '{run_id}'")
        return path

    def run_overview(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
        totals = summary["totals"]
        return {
            "run_id": path.name, "experiment": summary["experiment"], "started_at": summary["started_at"],
            "config_hash": summary["config_hash"], "cells": totals["cases"], "passed": totals["passed"],
            "pass_rate": totals["pass_rate"], "errors": totals["errors"], "cost_usd": totals["cost_usd"],
            "mock": bool(summary.get("mock_models")), "partial": bool(summary.get("filters")),
        }

    # --- pages ---

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (resources.files("promptlab") / "ui" / "static" / "app.html").read_text(encoding="utf-8")

    @app.get("/favicon.svg")
    async def favicon() -> Response:
        return Response(FAVICON, media_type="image/svg+xml")

    @app.get("/runs/{run_id}/report", response_class=HTMLResponse)
    async def report(run_id: str) -> str:
        return render_html(load_run(run_dir(run_id)), app_nav=True)

    # --- API ---

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return {"version": __version__, "root": str(root), "runs_dir": str(runs_dir)}

    @app.get("/api/experiments")
    async def experiments() -> list[dict[str, Any]]:
        listed = []
        for path in candidates():
            experiment_id = path.relative_to(root).as_posix()
            try:
                loaded = load_experiment(path)
            except ConfigError as exc:
                # Not every YAML file is an experiment (a price table is not). Only report files
                # that look like one and fail to load.
                if "prompts" in path.read_text(encoding="utf-8", errors="replace"):
                    listed.append({"id": experiment_id, "name": path.stem, "error": str(exc)})
                continue
            listed.append({
                "id": experiment_id, "name": loaded.experiment.name,
                "description": loaded.experiment.description, "cases": len(loaded.cases),
                "mock": all(m.provider == "mock" for m in loaded.experiment.models),
            })
        return listed

    @app.get("/api/experiment")
    async def experiment(id: str) -> dict[str, Any]:
        # Pick up a key added to .env while the app is running. Variables already set are kept.
        load_dotenv(root / ".env")
        loaded = load_experiment(experiment_path(id))
        exp = loaded.experiment
        variables = sorted(set().union(*(p.variables() for p in exp.prompts)))
        return {
            "id": id, "name": exp.name, "description": exp.description, "config_hash": loaded.config_hash,
            "warnings": loaded.warnings,
            "prompts": [p.model_dump() for p in exp.prompts],
            "models": await model_status(loaded),
            "scorers": [{"name": s.name or s.type, "type": s.type, "required": s.required} for s in exp.scorers],
            "variables": variables,
            "dataset": {"file": loaded.dataset_path.name, "cases": len(loaded.cases),
                        "synthetic": bool(loaded.dataset_meta.get("synthetic")),
                        "description": loaded.dataset_meta.get("description", "")},
            "cases": [{"id": c.id, "vars": c.vars, "expected": c.expected, "tags": c.tags} for c in loaded.cases],
        }

    @app.post("/api/try")
    async def try_one(request: TryRequest) -> dict[str, Any]:
        """Run one typed input through the chosen prompts and models. Nothing is saved."""
        case = TestCase(id=TRY_CASE_ID, vars=request.vars, expected=request.expected, tags=request.tags)
        loaded = prepare(request, cases=[case])
        results = await runner_for(loaded, request).run()
        return {"rows": [r.model_dump(mode="json") for r in results], "warnings": loaded.warnings}

    @app.post("/api/plan")
    async def plan(request: Selection) -> dict[str, Any]:
        """How many calls a full run would make, and how many are already cached."""
        runner = runner_for(prepare(request), request)
        try:
            planned = runner.plan()
        finally:
            await runner.aclose()
        return {"total": planned.total, "cached": planned.cached, "to_call": planned.to_call,
                "prompts": len(runner.prompts), "models": len(runner.models), "cases": len(runner.cases)}

    @app.post("/api/runs")
    async def start_run(request: RunRequest) -> dict[str, Any]:
        """Start a run of the whole test set in the background and return a job to poll."""
        name = request.name.strip() if request.name and request.name.strip() else None
        if name is not None:
            validate_run_name(name)  # reject a bad name now, not after the calls have been made
        loaded = prepare(request)
        runner = runner_for(loaded, request)
        planned = runner.plan()  # also fails here, before anything is called, if a key is missing
        job = Job(id=uuid.uuid4().hex[:12], total=planned.total, cached=planned.cached)
        jobs[job.id] = job

        def advance(_: Any) -> None:
            job.done += 1

        runner.on_cell = advance

        async def work() -> None:
            started_at = utc_now()
            try:
                results = await runner.run()
                path = save_run(runs_dir, loaded, results, started_at=started_at, finished_at=utc_now(),
                                prices=runner.prices, options=runner.options, name=name)
                job.run_id, job.status = path.name, "done"
            except Exception as exc:  # reported to the page; the server keeps running
                job.status, job.error = "error", str(exc) or type(exc).__name__

        job.task = asyncio.create_task(work())
        return job.public()

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> dict[str, Any]:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="no such job")
        return jobs[job_id].public()

    @app.get("/api/runs")
    async def runs() -> list[dict[str, Any]]:
        return [run_overview(path, summary) for path, summary in reversed(list_runs(runs_dir))]

    @app.get("/api/runs/{run_id}")
    async def run_summary(run_id: str) -> dict[str, Any]:
        return load_run(run_dir(run_id)).summary

    @app.get("/api/compare")
    async def compare(a: str, b: str) -> dict[str, Any]:
        return compare_runs(load_run(run_dir(a)), load_run(run_dir(b)))

    return app
