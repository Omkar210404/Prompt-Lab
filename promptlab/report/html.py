"""Self-contained HTML report: one file, with CSS, JavaScript and data inline, so it opens offline.

The summary, chart and failure lists are rendered here in Python. The side-by-side grid and the
per-case drill-down are built in the browser from the embedded data. Everything that comes from
a model or a test set is inserted as text, never as HTML.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from promptlab import __version__
from promptlab.report.table import MOCK_NOTE, fmt_cost, fmt_ms, fmt_rate, fmt_tokens
from promptlab.storage import StoredRun
from promptlab.summary import hardest_cases, worst_failures

# Categorical palette for models, in a fixed order. Past eight, models share a neutral colour
# and are told apart by their labels, which every bar carries anyway.
MAX_SERIES = 8

_ROW_FIELDS = (
    "case_id", "prompt_id", "model_id", "status", "output", "error", "latency_ms", "cached",
    "attempts", "usage", "cost_usd", "finish_reason", "passed", "score", "scores",
)

_environment = Environment(
    loader=PackageLoader("promptlab.report", "templates"),
    autoescape=select_autoescape(default=True, default_for_string=True),
    trim_blocks=True,
    lstrip_blocks=True,
)
_environment.filters.update(rate=fmt_rate, cost=fmt_cost, ms=fmt_ms)


def json_for_script(data: Any) -> str:
    """JSON that is safe inside a <script> element: no sequence in it can close the tag."""
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    )


def _series(model_ids: list[str]) -> dict[str, int]:
    """Colour slot per model: 1-8 in experiment order, 0 (neutral) past the eighth."""
    return {model_id: index + 1 if index < MAX_SERIES else 0 for index, model_id in enumerate(model_ids)}


def build_context(run: StoredRun) -> dict[str, Any]:
    summary = run.summary
    totals = summary["totals"]
    model_ids = [m["id"] for m in summary["models"]]
    series = _series(model_ids)

    chart: dict[str, list[dict[str, Any]]] = {}
    for group in summary["groups"]:
        rate = group["pass_rate"] or 0.0
        chart.setdefault(group["prompt_id"], []).append({
            "model_id": group["model_id"],
            "series": series.get(group["model_id"], 0),
            "pct": round(rate * 100, 2),
            "label": fmt_rate(group["pass_rate"]),
            "detail": f"{group['passed']}/{group['cases']}",
            "tip": json.dumps({
                "value": f"{fmt_rate(group['pass_rate'])} pass rate",
                "detail": f"{group['passed']} of {group['cases']} cases passed, mean score "
                          f"{group['mean_score']:.3f}, {group['errors']} errors",
                "prompt": group["prompt_id"],
                "model": group["model_id"],
                "series": series.get(group["model_id"], 0),
            }),
        })

    scorer_names = list(dict.fromkeys(entry["scorer"] for entry in summary["scorers"]))
    rates = {(e["prompt_id"], e["model_id"], e["scorer"]): e for e in summary["scorers"]}
    scorer_rows = [
        {
            "prompt_id": group["prompt_id"],
            "model_id": group["model_id"],
            "cells": [rates.get((group["prompt_id"], group["model_id"], name)) for name in scorer_names],
        }
        for group in summary["groups"]
    ]

    run_prompts = set(summary["prompts"])
    prompts = [p for p in run.experiment.get("prompts", []) if p.get("id") in run_prompts]
    used_cases = {row["case_id"] for row in run.rows}
    dataset = summary["dataset"]

    data = {
        "prompts": summary["prompts"],
        "models": model_ids,
        "cases": [
            {"id": c["id"], "vars": c.get("vars", {}), "expected": c.get("expected"), "tags": c.get("tags", [])}
            for c in run.cases if c["id"] in used_cases
        ],
        "rows": [{key: row.get(key) for key in _ROW_FIELDS} for row in run.rows],
    }

    return {
        "summary": summary,
        "totals": totals,
        "dataset": dataset,
        "calls_made": totals["cases"] - totals["cached"],
        "tokens": fmt_tokens(totals["prompt_tokens"], totals["completion_tokens"]),
        "groups": [
            {**g, "series": series.get(g["model_id"], 0),
             "tokens": fmt_tokens(g["prompt_tokens"], g["completion_tokens"])}
            for g in summary["groups"]
        ],
        "mock_note": MOCK_NOTE if summary.get("mock_models") else None,
        "chart": [{"prompt_id": prompt_id, "bars": bars} for prompt_id, bars in chart.items()],
        "legend": [{"id": model_id, "series": series[model_id]} for model_id in model_ids],
        "scorer_names": scorer_names,
        "scorer_rows": scorer_rows,
        "hardest": hardest_cases(run.rows, 10),
        "worst": worst_failures(run.rows, 10),
        "prompts": prompts,
        "data_json": json_for_script(data),
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def render_html(run: StoredRun, app_nav: bool = False) -> str:
    """pp_nav adds a link back to the web app, for reports the app serves itself."""
    return _environment.get_template("report.html.j2").render(**build_context(run), app_nav=app_nav)
