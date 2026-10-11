"""Command line: init, run, report, compare, list."""

from __future__ import annotations

import asyncio
import functools
import json
import shutil
import webbrowser
from importlib import resources
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from promptlab import __version__
from promptlab.cache import ResponseCache
from promptlab.compare import compare_runs
from promptlab.config import load_experiment
from promptlab.errors import ConfigError, PromptLabError
from promptlab.pricing import load_prices
from promptlab.report.table import make_console, print_comparison, print_run, runs_table
from promptlab.runner import Runner, RunOptions
from promptlab.scorers import scorer_types
from promptlab.storage import REPORT_FILE, list_runs, load_run, resolve_run, save_run, utc_now

app = typer.Typer(
    help="Run prompt variants over a test set, score the outputs, and compare the results.",
    no_args_is_help=True,
    add_completion=False,
)
console = make_console()
errors = make_console(stderr=True)

RunsDir = Annotated[Path, typer.Option("--runs-dir", help="Where runs are saved.", envvar="PROMPTLAB_RUNS_DIR")]

# Scaffold files written by `init`, as (file inside the package, name on disk).
SCAFFOLD = [
    ("experiment.yaml", "experiment.yaml"),
    ("dataset.jsonl", "dataset.jsonl"),
    ("prices.yaml", "prices.yaml"),
    ("env.example", ".env.example"),
]


def handle_errors(command):
    """Print expected errors as one line and exit with code 2, instead of a stack trace."""

    @functools.wraps(command)
    def wrapper(*args, **kwargs):
        try:
            return command(*args, **kwargs)
        except PromptLabError as exc:
            errors.print(f"[bold red]Error:[/bold red] {exc}", highlight=False)
            raise typer.Exit(code=2) from exc

    return wrapper


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _version(show: bool) -> None:
    if show:
        console.print(f"prompt-lab {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True,
                                          help="Show the version and exit.")] = False,
) -> None:
    # API keys come from the environment only. A .env file in the working directory is loaded
    # for convenience; variables already set in the shell win.
    load_dotenv(Path.cwd() / ".env")


@app.command()
@handle_errors
def init(
    directory: Annotated[Path, typer.Argument(help="Folder to create the starter files in.")] = Path("."),
    force: Annotated[bool, typer.Option("--force", help="Overwrite files that already exist.")] = False,
) -> None:
    """Write a starter experiment, test set, price table and .env.example."""
    scaffold = resources.files("promptlab") / "scaffold"
    existing = [name for _, name in SCAFFOLD if (directory / name).exists()]
    if existing and not force:
        raise ConfigError(
            f"{', '.join(existing)} already exist in {directory}. Use --force to overwrite, or pick another folder."
        )
    directory.mkdir(parents=True, exist_ok=True)
    for source, name in SCAFFOLD:
        with resources.as_file(scaffold / source) as path:
            shutil.copyfile(path, directory / name)
        console.print(f"  wrote {directory / name}", highlight=False)
    experiment = directory / "experiment.yaml"
    console.print(f"\nNext: prompt-lab run {experiment.as_posix()}", highlight=False)


@app.command()
@handle_errors
def run(
    experiment: Annotated[Path, typer.Argument(help="Experiment YAML file.")],
    name: Annotated[Optional[str], typer.Option(
        "--name", help="Fixed folder name for this run (replaces an earlier run with that name).")] = None,
    runs_dir: RunsDir = Path("runs"),
    cache_dir: Annotated[Path, typer.Option("--cache-dir", help="Response cache folder.")] = Path(".promptlab/cache"),
    concurrency: Annotated[Optional[int], typer.Option(
        "--concurrency", min=1, help="Parallel calls (overrides the experiment file).")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Neither read nor write the cache.")] = False,
    refresh: Annotated[bool, typer.Option("--refresh", help="Ignore cached responses and overwrite them.")] = False,
    limit: Annotated[Optional[int], typer.Option("--limit", min=1, help="Only the first N test cases.")] = None,
    prompt: Annotated[Optional[list[str]], typer.Option("--prompt", help="Only this prompt id (repeatable).")] = None,
    model: Annotated[Optional[list[str]], typer.Option("--model", help="Only this model id (repeatable).")] = None,
    dry_run: Annotated[bool, typer.Option(
        "--dry-run", help="Show how many calls would be made and how many are cached, then stop.")] = False,
    html: Annotated[bool, typer.Option("--html", help="Also write the HTML report.")] = False,
) -> None:
    """Run every prompt x model x test case, score the outputs and save the run."""
    loaded = load_experiment(experiment)
    prices = load_prices(loaded.prices_path)
    options = RunOptions(prompt_ids=prompt or [], model_ids=model or [], limit=limit, concurrency=concurrency)
    cache = ResponseCache(cache_dir, enabled=loaded.experiment.run.cache and not no_cache, refresh=refresh)
    runner = Runner(loaded, cache, prices=prices, options=options)

    plan = runner.plan()
    console.print(
        f"{loaded.experiment.name}: {_count(len(runner.prompts), 'prompt')} x {_count(len(runner.models), 'model')} x "
        f"{_count(len(runner.cases), 'case')} = {_count(plan.total, 'call')} ({plan.cached} cached, {plan.to_call} to make)",
        highlight=False,
    )
    if dry_run:
        table = Table(title="Dry run: nothing was called", title_justify="left")
        for column in ("Model", "Calls", "Cached", "To make"):
            table.add_column(column, justify="left" if column == "Model" else "right")
        for model_id, counts in plan.by_model.items():
            table.add_row(model_id, str(counts["total"]), str(counts["cached"]),
                          str(counts["total"] - counts["cached"]))
        console.print(table)
        if runner.judge_ids:
            console.print("Judge calls are not counted: they depend on the outputs.", highlight=False)
        return

    started_at = utc_now()
    with Progress(TextColumn("Running"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=errors, transient=True) as progress:
        task = progress.add_task("run", total=plan.total)
        runner.on_cell = lambda _: progress.advance(task)
        results = asyncio.run(runner.run())
    finished_at = utc_now()

    run_dir = save_run(runs_dir, loaded, results, started_at=started_at, finished_at=finished_at,
                       prices=prices, options=options, name=name)
    stored = load_run(run_dir)
    console.print()
    print_run(console, stored.summary)
    console.print(f"\nSaved: {run_dir.as_posix()}", highlight=False)
    if html:
        _write_html(stored, None)
    else:
        console.print(f"Next:  prompt-lab report {run_dir.name} --html", highlight=False)


def _write_html(stored, out: Path | None) -> Path:
    from promptlab.report.html import render_html

    target = out or stored.path / REPORT_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(stored), encoding="utf-8", newline="\n")
    console.print(f"HTML report: {target.as_posix()}", highlight=False)
    return target


@app.command()
@handle_errors
def report(
    run_ref: Annotated[str, typer.Argument(metavar="RUN", help="Run id, unique prefix, path, or 'latest'.")] = "latest",
    runs_dir: RunsDir = Path("runs"),
    html: Annotated[bool, typer.Option("--html", help="Write the self-contained HTML report.")] = False,
    out: Annotated[Optional[Path], typer.Option("--out", help="Where to write the HTML report.")] = None,
    open_browser: Annotated[bool, typer.Option("--open", help="Open the HTML report in a browser.")] = False,
    failures: Annotated[int, typer.Option("--failures", min=0, help="Also list the N worst failures.")] = 0,
) -> None:
    """Show a saved run as a table, and optionally write the HTML report."""
    stored = load_run(resolve_run(runs_dir, run_ref))
    print_run(console, stored.summary, stored.rows, failures=failures)
    if html or out or open_browser:
        console.print()
        target = _write_html(stored, out)
        if open_browser:
            webbrowser.open(target.resolve().as_uri())


@app.command()
@handle_errors
def compare(
    run_a: Annotated[str, typer.Argument(metavar="RUN_A", help="The earlier run (before).")],
    run_b: Annotated[str, typer.Argument(metavar="RUN_B", help="The later run (after).")],
    runs_dir: RunsDir = Path("runs"),
    fail_on_regression: Annotated[bool, typer.Option(
        "--fail-on-regression", help="Exit with code 1 if any case passed in A and fails in B.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the comparison as JSON.")] = False,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Rows shown per table.")] = 20,
) -> None:
    """Diff two runs and list regressions: cases that passed before and fail now."""
    comparison = compare_runs(load_run(resolve_run(runs_dir, run_a)), load_run(resolve_run(runs_dir, run_b)))
    if as_json:
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
    else:
        print_comparison(console, comparison, limit=limit)
    if fail_on_regression and comparison["counts"]["regressions"]:
        raise typer.Exit(code=1)


@app.command("list")
@handle_errors
def list_command(
    runs_dir: RunsDir = Path("runs"),
    scorers: Annotated[bool, typer.Option("--scorers", help="List the available scorer types instead.")] = False,
) -> None:
    """List saved runs, or the available scorers."""
    if scorers:
        table = Table(title="Scorers", title_justify="left")
        table.add_column("type")
        table.add_column("what it checks")
        for name, description in scorer_types().items():
            table.add_row(name, description)
        console.print(table)
        return
    runs = list_runs(runs_dir)
    if not runs:
        console.print(f"No runs in {runs_dir.as_posix()} yet. Start with: prompt-lab run <experiment.yaml>",
                      highlight=False)
        return
    console.print(runs_table(runs))
