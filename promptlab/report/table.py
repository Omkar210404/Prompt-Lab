"""Terminal output: the summary table, the per-scorer table, failures, run lists and comparisons."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from promptlab.summary import worst_failures

MOCK_NOTE = (
    "These outputs come from the scripted mock provider, not from a real model. "
    "They show how prompt-lab works. They are not a benchmark."
)
MOCK_BANNER = f"MOCK RESULTS. {MOCK_NOTE}"


def fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def fmt_cost(value: float | None) -> str:
    """Unknown cost is 'n/a', never $0."""
    if value is None:
        return "n/a"
    return "$0" if value == 0 else f"${value:.6f}" if value < 0.01 else f"${value:.4f}"


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f} ms" if value < 10 else f"{value:.0f} ms"


def make_console(stderr: bool = False) -> Console:
    """A console that does not squeeze tables to 80 columns when the output is piped to a file."""
    console = Console(stderr=stderr)
    if not console.is_terminal:
        console = Console(stderr=stderr, width=max(console.width, 130))
    return console


def _id_column(table: Table, header: str) -> None:
    # Ids are what people copy into --prompt and --model, so they wrap instead of being cut off.
    table.add_column(header, overflow="fold", min_width=10)


def fmt_tokens(prompt: int | None, completion: int | None) -> str:
    return "n/a" if prompt is None or completion is None else f"{prompt:,} / {completion:,}"


def fmt_delta(value: float) -> Text:
    text = f"{value * 100:+.1f} pts" if value else "0.0 pts"
    return Text(text, style="red" if value < 0 else "green" if value > 0 else "dim")


def _rate_style(value: float | None) -> str:
    if value is None:
        return "dim"
    return "green" if value >= 0.9 else "yellow" if value >= 0.6 else "red"


def _table(title: str, **kwargs: Any) -> Table:
    return Table(title=title, title_justify="left", title_style="bold", box=box.SIMPLE_HEAD, **kwargs)


def mock_banner(summary: dict[str, Any]) -> Text | None:
    if not summary.get("mock_models"):
        return None
    return Text(MOCK_BANNER, style="bold yellow")


def run_header(summary: dict[str, Any]) -> Text:
    dataset = summary["dataset"]
    lines = [
        f"Run:         {summary['run_id']}",
        f"Experiment:  {summary['experiment']}",
        f"Started:     {summary['started_at']}   ({summary['duration_seconds']}s)",
        f"Config hash: {summary['config_hash'][:12]}",
        f"Test set:    {dataset['file']}, {dataset['cases']} cases"
        + (f" of {dataset['total_cases']}" if dataset["cases"] != dataset["total_cases"] else "")
        + (" (synthetic)" if dataset.get("meta", {}).get("synthetic") else ""),
    ]
    if summary.get("filters"):
        lines.append("Partial run: filtered with --prompt, --model or --limit")
    return Text("\n".join(lines))


def summary_table(summary: dict[str, Any]) -> Table:
    table = _table("Results by prompt and model")
    _id_column(table, "Prompt")
    _id_column(table, "Model")
    for name in ("Pass rate", "Passed", "Score", "Errors", "p50", "Tokens in / out", "Est. cost"):
        table.add_column(name, justify="right")
    for group in summary["groups"]:
        table.add_row(
            group["prompt_id"],
            group["model_id"],
            Text(fmt_rate(group["pass_rate"]), style=_rate_style(group["pass_rate"])),
            f"{group['passed']}/{group['cases']}",
            f"{group['mean_score']:.3f}",
            str(group["errors"]),
            fmt_ms(group["latency_ms_p50"]),
            fmt_tokens(group["prompt_tokens"], group["completion_tokens"]),
            fmt_cost(group["cost_usd"]),
        )
    return table


def scorer_table(summary: dict[str, Any]) -> Table:
    """Pass rate of each scorer per prompt and model: it shows which check a prompt fails."""
    names = list(dict.fromkeys(entry["scorer"] for entry in summary["scorers"]))
    rates = {(e["prompt_id"], e["model_id"], e["scorer"]): e["pass_rate"] for e in summary["scorers"]}
    table = _table("Pass rate by scorer")
    _id_column(table, "Prompt")
    _id_column(table, "Model")
    for name in names:
        table.add_column(name, justify="right")
    for group in summary["groups"]:
        cells = []
        for name in names:
            rate = rates.get((group["prompt_id"], group["model_id"], name))
            cells.append(Text(fmt_rate(rate), style=_rate_style(rate)))
        table.add_row(group["prompt_id"], group["model_id"], *cells)
    return table


def totals_line(summary: dict[str, Any]) -> Text:
    totals = summary["totals"]
    made = totals["cases"] - totals["cached"]
    parts = [
        f"Cells: {totals['cases']} ({totals['passed']} passed, {totals['failed']} failed, {totals['errors']} errors)",
        f"Calls: {made} made, {totals['cached']} from cache",
        f"Est. cost: {fmt_cost(totals['cost_usd'])} (spent this run: {fmt_cost(totals['cost_usd_spent'])})",
    ]
    judge = totals.get("judge")
    if judge:
        parts.append(
            f"Judge: {judge['calls']} calls ({judge['cached_calls']} cached), est. cost {fmt_cost(judge['cost_usd'])}"
        )
    if totals["scorer_errors"]:
        parts.append(f"Scorer errors: {totals['scorer_errors']} cells (see reasons; counted as failures)")
    return Text("\n".join(parts))


def failures_table(rows: list[dict[str, Any]], limit: int) -> Table:
    table = _table(f"Worst failures (lowest score first, top {limit})")
    table.add_column("Case")
    _id_column(table, "Prompt")
    _id_column(table, "Model")
    table.add_column("Score", justify="right")
    table.add_column("Why", overflow="fold")
    for failure in worst_failures(rows, limit):
        table.add_row(failure["case_id"], failure["prompt_id"], failure["model_id"],
                      f"{failure['score']:.2f}", "\n".join(failure["reasons"]))
    return table


def print_run(console: Console, summary: dict[str, Any], rows: list[dict[str, Any]] | None = None,
              failures: int = 0) -> None:
    banner = mock_banner(summary)
    parts: list[Any] = [run_header(summary), ""]
    if banner:
        parts += [banner, ""]
    for warning in summary.get("warnings", []):
        parts += [Text(f"Warning: {warning}", style="yellow"), ""]
    parts += [summary_table(summary), scorer_table(summary)]
    if failures and rows is not None and summary["totals"]["failed"]:
        parts.append(failures_table(rows, failures))
    parts.append(totals_line(summary))
    console.print(Group(*parts))


def runs_table(runs: list[tuple[Any, dict[str, Any]]]) -> Table:
    table = _table("Saved runs (oldest first)")
    for name in ("Run", "Experiment", "Started (UTC)", "Config"):
        table.add_column(name)
    for name in ("Cells", "Pass rate", "Est. cost"):
        table.add_column(name, justify="right")
    table.add_column("Mock")
    for path, summary in runs:
        totals = summary["totals"]
        table.add_row(
            path.name, summary["experiment"], summary["started_at"].replace("+00:00", "Z"),
            summary["config_hash"][:8], str(totals["cases"]), fmt_rate(totals["pass_rate"]),
            fmt_cost(totals["cost_usd"]), "yes" if summary.get("mock_models") else "no",
        )
    return table


def _changes_table(title: str, changes: list[dict[str, Any]], why: str, limit: int) -> Table:
    table = _table(title + (f" (showing {limit} of {len(changes)})" if len(changes) > limit else ""))
    table.add_column("Case")
    _id_column(table, "Prompt")
    _id_column(table, "Model")
    table.add_column("Score A -> B", justify="right")
    table.add_column(why, overflow="fold")
    for change in changes[:limit]:
        table.add_row(change["case_id"], change["prompt_id"], change["model_id"],
                      f"{change['score_a']:.2f} -> {change['score_b']:.2f}", "\n".join(change["reasons"]))
    return table


def print_comparison(console: Console, comparison: dict[str, Any], limit: int = 20) -> None:
    a, b, counts = comparison["run_a"], comparison["run_b"], comparison["counts"]
    parts: list[Any] = [
        Text(f"A (before): {a['run_id']}   {a['started_at']}   config {a['config_hash'][:8]}"),
        Text(f"B (after):  {b['run_id']}   {b['started_at']}   config {b['config_hash'][:8]}"),
    ]

    edits = []
    if comparison["changed_prompts"]:
        edits.append(f"prompts changed: {', '.join(comparison['changed_prompts'])}")
    if comparison["changed_models"]:
        edits.append(f"models changed: {', '.join(comparison['changed_models'])}")
    if comparison["changed_scorers"]:
        edits.append("scorers changed")
    if comparison["same_config"]:
        parts.append(Text("Same config hash: any difference comes from the models, not from your files."))
    elif edits:
        parts.append(Text("Between A and B: " + "; ".join(edits)))
    parts.append("")

    if a["mock_models"] or b["mock_models"]:
        parts += [Text(MOCK_BANNER, style="bold yellow"), ""]
    for warning in comparison["warnings"]:
        parts += [Text(f"Warning: {warning}", style="yellow"), ""]

    table = _table("Pass rate by prompt and model (cells present in both runs)")
    _id_column(table, "Prompt")
    _id_column(table, "Model")
    for name in ("A", "B", "Change", "Regressions", "Fixes"):
        table.add_column(name, justify="right")
    for group in comparison["groups"]:
        table.add_row(
            group["prompt_id"], group["model_id"],
            f"{fmt_rate(group['pass_rate_a'])} ({group['passed_a']}/{group['cases']})",
            f"{fmt_rate(group['pass_rate_b'])} ({group['passed_b']}/{group['cases']})",
            fmt_delta(group["pass_rate_delta"]),
            Text(str(group["regressions"]), style="red" if group["regressions"] else "dim"),
            Text(str(group["fixes"]), style="green" if group["fixes"] else "dim"),
        )
    parts.append(table)

    if comparison["regressions"]:
        parts.append(_changes_table("Regressions: passed in A, fail in B", comparison["regressions"],
                                    "Why it fails now", limit))
    if comparison["fixes"]:
        parts.append(_changes_table("Fixes: failed in A, pass in B", comparison["fixes"],
                                    "Why it failed before", limit))

    verdict = Text(
        f"Compared {counts['compared']} cells: {counts['regressions']} regressions, {counts['fixes']} fixes, "
        f"{counts['still_passing']} still passing, {counts['still_failing']} still failing.",
        style="bold red" if counts["regressions"] else "bold green" if counts["compared"] else "bold yellow",
    )
    parts.append(verdict)
    if counts["only_in_a"] or counts["only_in_b"]:
        parts.append(Text(f"Not compared: {counts['only_in_a']} cells only in A, {counts['only_in_b']} only in B."))
    console.print(Group(*parts))
