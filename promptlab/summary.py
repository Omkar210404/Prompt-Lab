"""Aggregate result rows into the numbers shown in reports. Pure functions over plain dicts."""

from __future__ import annotations

from typing import Any


def percentile(values: list[float], q: float) -> float | None:
    """q-th percentile (0-100) with linear interpolation between the two nearest values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _sum_or_none(values: list[float | int | None]) -> float | int | None:
    """Sum of the known values, or None when none are known. An unknown cost is not a cost of zero."""
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    with_usage = [r["usage"] for r in rows if r.get("usage")]
    costs = [r.get("cost_usd") for r in rows]
    spent = _sum_or_none([r.get("cost_usd") for r in rows if not r.get("cached")])
    return {
        "cases": cases,
        "passed": passed,
        "failed": cases - passed,
        "pass_rate": round(passed / cases, 4) if cases else None,
        "mean_score": round(sum(r["score"] for r in rows) / cases, 4) if cases else None,
        # Calls that never produced an output (timeout, HTTP error). They count as failures.
        "errors": sum(1 for r in rows if r["status"] == "error"),
        # Cells where a scorer could not reach a verdict, such as an unusable judge reply.
        "scorer_errors": sum(1 for r in rows if any(s.get("error") for s in r.get("scores", []))),
        "cached": sum(1 for r in rows if r.get("cached")),
        "latency_ms_p50": _round(percentile(latencies, 50), 2),
        "latency_ms_p95": _round(percentile(latencies, 95), 2),
        "prompt_tokens": _sum_or_none([u.get("prompt_tokens") for u in with_usage]),
        "completion_tokens": _sum_or_none([u.get("completion_tokens") for u in with_usage]),
        # Estimated cost of every call in the run, at the price table's rates.
        "cost_usd": _round(_sum_or_none(costs), 8),
        # The part of it actually incurred by this run: cache hits cost nothing the second time.
        "cost_usd_spent": _round(0.0 if spent is None and _sum_or_none(costs) is not None else spent, 8),
    }


def _judge_totals(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    judged = [r["judge"] for r in rows if r.get("judge")]
    if not judged:
        return None
    return {
        "calls": sum(j["calls"] for j in judged),
        "cached_calls": sum(j["cached_calls"] for j in judged),
        "prompt_tokens": _sum_or_none([j.get("prompt_tokens") for j in judged]),
        "completion_tokens": _sum_or_none([j.get("completion_tokens") for j in judged]),
        "cost_usd": _round(_sum_or_none([j.get("cost_usd") for j in judged]), 8),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals, one entry per prompt x model, and a pass rate per scorer within each of those."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["prompt_id"], row["model_id"]), []).append(row)

    groups, scorers = [], []
    for (prompt_id, model_id), members in grouped.items():
        groups.append({"prompt_id": prompt_id, "model_id": model_id, **stats(members)})

        by_scorer: dict[str, list[dict[str, Any]]] = {}
        for row in members:
            for score in row.get("scores", []):
                by_scorer.setdefault(score["scorer"], []).append(score)
        for name, results in by_scorer.items():
            passed = sum(1 for s in results if s["passed"])
            scorers.append({
                "scorer": name,
                "prompt_id": prompt_id,
                "model_id": model_id,
                "cases": len(results),
                "passed": passed,
                "pass_rate": round(passed / len(results), 4),
                "errors": sum(1 for s in results if s.get("error")),
            })

    return {"totals": {**stats(rows), "judge": _judge_totals(rows)}, "groups": groups, "scorers": scorers}
