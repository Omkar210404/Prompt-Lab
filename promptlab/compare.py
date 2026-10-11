"""Diff two runs. The question it answers: what passed before and fails now?

Cells are matched on (case, prompt, model). Pass rates are computed over the cells both runs
have, so a run with extra cases cannot look better or worse just for having them.
"""

from __future__ import annotations

from typing import Any

from promptlab.storage import StoredRun
from promptlab.summary import failure_reasons

Key = tuple[str, str, str]


def _key(row: dict[str, Any]) -> Key:
    return row["case_id"], row["prompt_id"], row["model_id"]


def _describe(run: StoredRun) -> dict[str, Any]:
    summary = run.summary
    return {
        "run_id": summary["run_id"],
        "experiment": summary["experiment"],
        "started_at": summary["started_at"],
        "config_hash": summary["config_hash"],
        "dataset_sha256": summary["dataset"]["sha256"],
        "mock_models": summary.get("mock_models", []),
    }


def _changed_cases(a: StoredRun, b: StoredRun) -> list[str]:
    """Ids of cases present in both test sets whose input or expected answer differs."""
    def content(case: dict[str, Any]) -> tuple:
        return case.get("vars"), case.get("expected"), case.get("checks")

    in_b = {case["id"]: case for case in b.cases}
    return sorted(case["id"] for case in a.cases if case["id"] in in_b and content(case) != content(in_b[case["id"]]))


def _changed_by_id(a: list[dict], b: list[dict]) -> list[str]:
    """Ids of prompts (or models) defined in both runs with different contents."""
    in_b = {item["id"]: item for item in b}
    return [item["id"] for item in a if item["id"] in in_b and item != in_b[item["id"]]]


def _change(key: Key, row_a: dict[str, Any], row_b: dict[str, Any], reasons_from: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": key[0], "prompt_id": key[1], "model_id": key[2],
        "score_a": round(row_a["score"], 4), "score_b": round(row_b["score"], 4),
        # For a regression: why it fails now. For a fix: why it used to fail.
        "reasons": failure_reasons(reasons_from),
    }


def compare_runs(a: StoredRun, b: StoredRun) -> dict[str, Any]:
    rows_a = {_key(row): row for row in a.rows}
    rows_b = {_key(row): row for row in b.rows}
    shared = [key for key in rows_a if key in rows_b]

    regressions, fixes, still_passing, still_failing = [], [], 0, 0
    for key in shared:
        before, after = rows_a[key], rows_b[key]
        if before["passed"] and not after["passed"]:
            regressions.append(_change(key, before, after, reasons_from=after))
        elif not before["passed"] and after["passed"]:
            fixes.append(_change(key, before, after, reasons_from=before))
        elif after["passed"]:
            still_passing += 1
        else:
            still_failing += 1

    groups: dict[tuple[str, str], list[Key]] = {}
    for key in shared:
        groups.setdefault((key[1], key[2]), []).append(key)
    group_rows = []
    for (prompt_id, model_id), keys in groups.items():
        cases = len(keys)
        passed_a = sum(1 for k in keys if rows_a[k]["passed"])
        passed_b = sum(1 for k in keys if rows_b[k]["passed"])
        score_a = sum(rows_a[k]["score"] for k in keys) / cases
        score_b = sum(rows_b[k]["score"] for k in keys) / cases
        group_rows.append({
            "prompt_id": prompt_id, "model_id": model_id, "cases": cases,
            "passed_a": passed_a, "passed_b": passed_b,
            "pass_rate_a": round(passed_a / cases, 4), "pass_rate_b": round(passed_b / cases, 4),
            "pass_rate_delta": round((passed_b - passed_a) / cases, 4),
            "mean_score_a": round(score_a, 4), "mean_score_b": round(score_b, 4),
            "regressions": sum(1 for r in regressions if (r["prompt_id"], r["model_id"]) == (prompt_id, model_id)),
            "fixes": sum(1 for f in fixes if (f["prompt_id"], f["model_id"]) == (prompt_id, model_id)),
        })

    info_a, info_b = _describe(a), _describe(b)
    changed_cases = _changed_cases(a, b)
    warnings = []
    if info_a["experiment"] != info_b["experiment"]:
        warnings.append(
            f"The runs are from different experiments ('{info_a['experiment']}' and '{info_b['experiment']}')."
        )
    if info_a["dataset_sha256"] != info_b["dataset_sha256"]:
        detail = f" {len(changed_cases)} shared case(s) changed content: {', '.join(changed_cases[:5])}" \
                 f"{', ...' if len(changed_cases) > 5 else ''}." if changed_cases else ""
        warnings.append("The test sets differ. Only cases with the same id are compared." + detail)
    if not shared:
        warnings.append("The runs have no cells in common (no matching case, prompt and model ids).")
    if bool(info_a["mock_models"]) != bool(info_b["mock_models"]):
        warnings.append("One run used the mock provider and the other did not.")

    return {
        "run_a": info_a,
        "run_b": info_b,
        "same_config": info_a["config_hash"] == info_b["config_hash"],
        "same_dataset": info_a["dataset_sha256"] == info_b["dataset_sha256"],
        # What was edited between the runs, read from the two experiment snapshots.
        "changed_prompts": _changed_by_id(a.experiment.get("prompts", []), b.experiment.get("prompts", [])),
        "changed_models": _changed_by_id(a.experiment.get("models", []), b.experiment.get("models", [])),
        "changed_scorers": a.experiment.get("scorers") != b.experiment.get("scorers"),
        "changed_cases": changed_cases,
        "warnings": warnings,
        "counts": {
            "compared": len(shared),
            "regressions": len(regressions),
            "fixes": len(fixes),
            "still_passing": still_passing,
            "still_failing": still_failing,
            "only_in_a": len(rows_a) - len(shared),
            "only_in_b": len(rows_b) - len(shared),
        },
        "regressions": regressions,
        "fixes": fixes,
        "groups": group_rows,
        "only_in_a": [list(key) for key in rows_a if key not in rows_b],
        "only_in_b": [list(key) for key in rows_b if key not in rows_a],
    }
