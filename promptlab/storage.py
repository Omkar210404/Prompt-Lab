"""Runs on disk. One directory per run:

    results.jsonl     one line per prompt x model x test case
    summary.json      totals, config hash, timestamps, prices used
    experiment.yaml   the experiment exactly as it was run
    dataset.jsonl     the test set exactly as it was run
    report.html       written by `prompt-lab report --html`

The snapshot files make a run directory self-contained: `prompt-lab run <run>/experiment.yaml`
repeats it and arrives at the same config hash.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from promptlab import __version__
from promptlab.config import LoadedExperiment
from promptlab.errors import ConfigError, RunNotFoundError
from promptlab.pricing import Price, price_for
from promptlab.runner import CellResult, RunOptions
from promptlab.summary import summarize

RESULTS_FILE = "results.jsonl"
SUMMARY_FILE = "summary.json"
SNAPSHOT_FILE = "experiment.yaml"
DATASET_FILE = "dataset.jsonl"
PRICES_FILE = "prices.yaml"
REPORT_FILE = "report.html"

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass
class StoredRun:
    path: Path
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    cases: list[dict[str, Any]]
    experiment: dict[str, Any]

    @property
    def run_id(self) -> str:
        return self.summary["run_id"]


class _BlockDumper(yaml.SafeDumper):
    """Writes multi-line strings (prompt templates) as readable `|` blocks."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|" if "\n" in data else None)


_BlockDumper.add_representer(str, _represent_str)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "run"


def make_run_id(experiment_name: str, config_hash: str, started_at: datetime) -> str:
    return f"{started_at:%Y%m%dT%H%M%SZ}-{slugify(experiment_name)}-{config_hash[:8]}"


def _is_run_dir(path: Path) -> bool:
    return (path / SUMMARY_FILE).is_file()


def _prepare_run_dir(runs_dir: Path, run_id: str, named: bool) -> Path:
    run_dir = runs_dir / run_id
    if run_dir.exists():
        if not named:  # two runs started in the same second
            suffix = 2
            while (runs_dir / f"{run_id}-{suffix}").exists():
                suffix += 1
            run_dir = runs_dir / f"{run_id}-{suffix}"
        elif any(run_dir.iterdir()) and not _is_run_dir(run_dir):
            raise ConfigError(f"refusing to write into {run_dir}: it exists and is not a prompt-lab run")
    run_dir.mkdir(parents=True, exist_ok=True)
    # A named run replaces the previous one with that name; its old report would describe old results.
    (run_dir / REPORT_FILE).unlink(missing_ok=True)
    return run_dir


def _copy_scorer_files(loaded: LoadedExperiment, run_dir: Path, snapshot: dict[str, Any]) -> None:
    """Copy files that scorers read (JSON Schemas) next to the snapshot, keeping their relative paths."""
    base = loaded.base_dir.resolve()

    def copy(relative: str) -> str:
        source = (base / relative).resolve()
        if not source.is_file():
            return relative
        if base not in source.parents:
            return str(source)  # outside the experiment folder: point at it instead of copying
        target = run_dir / source.relative_to(base)
        if source != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        return source.relative_to(base).as_posix()

    def is_file_option(key: str, value: Any) -> bool:
        return key.endswith("_file") and isinstance(value, str)

    for scorer in snapshot.get("scorers", []):
        for key, value in scorer.items():
            if is_file_option(key, value):
                scorer[key] = copy(value)
    for case in loaded.cases:
        for check in case.checks:
            for key, value in check.options.items():
                if is_file_option(key, value):
                    copy(value)


def build_summary(
    loaded: LoadedExperiment,
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    prices: dict[str, Price] | None = None,
    options: RunOptions | None = None,
) -> dict[str, Any]:
    experiment = loaded.experiment
    options = options or RunOptions()
    prices = prices or {}
    run_model_ids = {row["model_id"] for row in rows}
    run_prompt_ids = {row["prompt_id"] for row in rows}
    models = [m for m in experiment.models if m.id in run_model_ids]
    judges = []
    if any(row.get("judge") for row in rows):
        configs = [*experiment.scorers, *(check for case in loaded.cases for check in case.checks)]
        judge_ids = dict.fromkeys(cfg.options["judge"] for cfg in configs if cfg.type == "llm_judge")
        judges = [experiment.judge(judge_id) for judge_id in judge_ids]

    used_prices = {}
    for model in [*models, *judges]:
        price = price_for(prices, model)
        if price is not None:
            used_prices[model.model] = price.model_dump()

    filters = {
        "prompts": options.prompt_ids or None,
        "models": options.model_ids or None,
        "limit": options.limit,
    }
    return {
        "run_id": run_id,
        "experiment": experiment.name,
        "description": experiment.description,
        "config_hash": loaded.config_hash,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "promptlab_version": __version__,
        "python_version": platform.python_version(),
        "dataset": {
            "file": loaded.dataset_path.name,
            "sha256": loaded.dataset_sha256,
            "cases": len({row["case_id"] for row in rows}),
            "total_cases": len(loaded.cases),
            "meta": loaded.dataset_meta,
        },
        "prompts": [p.id for p in experiment.prompts if p.id in run_prompt_ids],
        "models": [{"id": m.id, "provider": m.provider, "model": m.model} for m in models],
        "judges": [{"id": j.id, "provider": j.provider, "model": j.model} for j in judges],
        # Models answered by the scripted mock provider. A run with any of these is a demo, not a benchmark.
        "mock_models": [m.id for m in [*models, *judges] if m.provider == "mock"],
        # Set when the run covered only part of the experiment (--prompt, --model, --limit).
        "filters": filters if any(filters.values()) else None,
        "warnings": loaded.warnings,
        # The price table entries used for this run's cost estimates (USD per 1M tokens).
        "prices": used_prices,
        **summarize(rows),
    }


def save_run(
    runs_dir: str | Path,
    loaded: LoadedExperiment,
    results: list[CellResult],
    *,
    started_at: datetime,
    finished_at: datetime,
    prices: dict[str, Price] | None = None,
    options: RunOptions | None = None,
    name: str | None = None,
) -> Path:
    """Write a run directory and return its path. `name` gives it a fixed folder name instead of a timestamp."""
    if name is not None and not _SAFE_NAME.fullmatch(name):
        raise ConfigError(f"invalid run name '{name}': use letters, digits, dot, dash and underscore")
    run_id = name or make_run_id(loaded.experiment.name, loaded.config_hash, started_at)
    run_dir = _prepare_run_dir(Path(runs_dir), run_id, named=name is not None)
    run_id = run_dir.name

    rows = [result.model_dump(mode="json") for result in results]
    with (run_dir / RESULTS_FILE).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(loaded, rows, run_id=run_id, started_at=started_at, finished_at=finished_at,
                            prices=prices, options=options)
    (run_dir / SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    snapshot = loaded.experiment.model_dump(mode="json", exclude_defaults=True)
    snapshot["dataset"] = DATASET_FILE
    if (run_dir / DATASET_FILE).resolve() != loaded.dataset_path.resolve():
        shutil.copyfile(loaded.dataset_path, run_dir / DATASET_FILE)
    if loaded.prices_path is not None:
        snapshot["prices_file"] = PRICES_FILE
        if (run_dir / PRICES_FILE).resolve() != loaded.prices_path.resolve():
            shutil.copyfile(loaded.prices_path, run_dir / PRICES_FILE)
    _copy_scorer_files(loaded, run_dir, snapshot)
    header = (
        f"# Snapshot of the experiment as it was run. Run: {run_id}. Config hash: {loaded.config_hash}\n"
        f"# Source file: {loaded.path.name}\n"
    )
    (run_dir / SNAPSHOT_FILE).write_text(
        header + yaml.dump(snapshot, Dumper=_BlockDumper, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8", newline="\n",
    )
    return run_dir


def load_run(path: str | Path) -> StoredRun:
    path = Path(path)
    if not _is_run_dir(path):
        raise RunNotFoundError(f"not a prompt-lab run (no {SUMMARY_FILE}): {path}")
    try:
        summary = json.loads((path / SUMMARY_FILE).read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (path / RESULTS_FILE).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        cases = []
        if (path / DATASET_FILE).is_file():
            for line in (path / DATASET_FILE).read_text(encoding="utf-8").splitlines():
                record = json.loads(line) if line.strip() else None
                if isinstance(record, dict) and "_meta" not in record:
                    record.setdefault("id", f"case-{len(cases) + 1:04d}")
                    cases.append(record)
        experiment = {}
        if (path / SNAPSHOT_FILE).is_file():
            experiment = yaml.safe_load((path / SNAPSHOT_FILE).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise RunNotFoundError(f"cannot read run {path}: {exc}") from exc
    return StoredRun(path=path, summary=summary, rows=rows, cases=cases, experiment=experiment)


def list_runs(runs_dir: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every run under `runs_dir` as (path, summary), oldest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    runs = []
    for child in runs_dir.iterdir():
        if child.is_dir() and _is_run_dir(child):
            try:
                runs.append((child, json.loads((child / SUMMARY_FILE).read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue  # an unreadable summary should not hide the other runs
    return sorted(runs, key=lambda item: (item[1].get("started_at", ""), item[0].name))


def resolve_run(runs_dir: str | Path, reference: str) -> Path:
    """Find a run by path, exact id, unique id prefix, or the word `latest`."""
    as_path = Path(reference)
    if as_path.is_dir() and _is_run_dir(as_path):
        return as_path
    runs = list_runs(runs_dir)
    if not runs:
        raise RunNotFoundError(f"no runs found in {Path(runs_dir)}. Run an experiment first: prompt-lab run <file>")
    if reference == "latest":
        return runs[-1][0]
    names = [path.name for path, _ in runs]
    if reference in names:
        return Path(runs_dir) / reference
    matches = [name for name in names if name.startswith(reference)]
    if len(matches) == 1:
        return Path(runs_dir) / matches[0]
    if matches:
        raise RunNotFoundError(f"'{reference}' matches several runs: {', '.join(matches)}")
    raise RunNotFoundError(f"no run matches '{reference}' in {Path(runs_dir)} (try: prompt-lab list)")
