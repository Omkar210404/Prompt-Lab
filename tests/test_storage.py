"""Saving and loading runs."""

import json
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from promptlab.cache import ResponseCache
from promptlab.config import load_experiment
from promptlab.errors import ConfigError, RunNotFoundError
from promptlab.pricing import Price
from promptlab.runner import Runner, RunOptions
from promptlab.storage import list_runs, load_run, make_run_id, resolve_run, save_run, slugify

T0 = datetime(2026, 10, 11, 14, 30, 22, tzinfo=timezone.utc)
PRICES = {"m1": Price(input=0.10, output=0.40, note="fictional")}


async def make_run(path, tmp_path, name=None, started_at=T0, options=None, rules=None):
    loaded = load_experiment(path)
    cache = ResponseCache(tmp_path / "cache", enabled=False)
    results = await Runner(loaded, cache, prices=PRICES, options=options).run()
    return save_run(tmp_path / "runs", loaded, results, started_at=started_at,
                    finished_at=started_at + timedelta(seconds=1.5), prices=PRICES, options=options, name=name)


def test_run_id_has_timestamp_slug_and_hash():
    assert make_run_id("Support Email: v2!", "abcdef1234567890", T0) == "20261011T143022Z-support-email-v2-abcdef12"
    assert slugify("  ") == "run"


async def test_round_trip(write_experiment, tmp_path):
    run_dir = await make_run(write_experiment(), tmp_path)
    assert run_dir.name.startswith("20261011T143022Z-test-exp-")
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "dataset.jsonl", "experiment.yaml", "results.jsonl", "summary.json"]

    run = load_run(run_dir)
    assert run.run_id == run_dir.name and len(run.rows) == 6
    assert [c["id"] for c in run.cases] == ["c1", "c2", "c3"]
    assert run.rows[0]["case_id"] == "c1" and run.rows[0]["scores"][0]["scorer"] == "exact_match"
    assert [p["id"] for p in run.experiment["prompts"]] == ["p1", "p2"]

    # results.jsonl is plain JSON Lines: one object per line
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 and all(isinstance(json.loads(line), dict) for line in lines)


async def test_summary_records_what_makes_a_run_reproducible(write_experiment, tmp_path):
    path = write_experiment()
    loaded = load_experiment(path)
    summary = load_run(await make_run(path, tmp_path)).summary

    assert summary["experiment"] == "test-exp"
    assert summary["config_hash"] == loaded.config_hash
    assert summary["started_at"] == "2026-10-11T14:30:22+00:00"
    assert summary["finished_at"] == "2026-10-11T14:30:23+00:00" and summary["duration_seconds"] == 1.5
    assert summary["dataset"] == {"file": "cases.jsonl", "sha256": loaded.dataset_sha256, "cases": 3,
                                  "total_cases": 3, "meta": {}}
    assert summary["prompts"] == ["p1", "p2"]
    assert summary["models"] == [{"id": "m1", "provider": "mock", "model": "m1"}]
    assert summary["mock_models"] == ["m1"]
    assert summary["filters"] is None
    assert summary["prices"] == {"m1": {"input": 0.10, "output": 0.40, "note": "fictional"}}
    assert summary["promptlab_version"] and summary["python_version"]
    assert summary["totals"]["cases"] == 6 and summary["totals"]["pass_rate"] == 1.0
    assert [(g["prompt_id"], g["model_id"], g["passed"]) for g in summary["groups"]] == [("p1", "m1", 3), ("p2", "m1", 3)]
    assert summary["scorers"][0] == {"scorer": "exact_match", "prompt_id": "p1", "model_id": "m1",
                                     "cases": 3, "passed": 3, "pass_rate": 1.0, "errors": 0}


async def test_partial_run_is_marked(write_experiment, tmp_path):
    run_dir = await make_run(write_experiment(), tmp_path, options=RunOptions(prompt_ids=["p2"], limit=2))
    summary = load_run(run_dir).summary
    assert summary["filters"] == {"prompts": ["p2"], "models": None, "limit": 2}
    assert summary["prompts"] == ["p2"]
    assert summary["dataset"]["cases"] == 2 and summary["dataset"]["total_cases"] == 3


async def test_snapshot_can_be_rerun_with_the_same_config_hash(write_experiment, tmp_path):
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "s.json").write_text('{"type": "string"}', encoding="utf-8")
    (tmp_path / "prices.yaml").write_text("m1: {input: 1, output: 2}\n", encoding="utf-8")
    path = write_experiment({
        "prompts": [{"id": "p1", "template": "Line one\nJSON: {\"a\": 1}\nClassify: {text}"}],
        "scorers": [{"type": "exact_match"}, {"type": "json_schema", "schema_file": "schemas/s.json", "required": False}],
        "prices_file": "prices.yaml",
    })
    original = load_experiment(path)
    run_dir = await make_run(path, tmp_path)

    assert (run_dir / "schemas" / "s.json").is_file() and (run_dir / "prices.yaml").is_file()
    snapshot_text = (run_dir / "experiment.yaml").read_text(encoding="utf-8")
    assert snapshot_text.startswith("# Snapshot of the experiment as it was run.")
    assert original.config_hash in snapshot_text
    snapshot = yaml.safe_load(snapshot_text)
    assert snapshot["dataset"] == "dataset.jsonl" and snapshot["prices_file"] == "prices.yaml"
    assert snapshot["prompts"][0]["template"] == original.experiment.prompts[0].template

    reloaded = load_experiment(run_dir / "experiment.yaml")
    assert reloaded.config_hash == original.config_hash
    assert reloaded.prices_path.resolve() == (run_dir / "prices.yaml").resolve()


async def test_named_run_replaces_the_previous_one(write_experiment, tmp_path):
    path = write_experiment()
    first = await make_run(path, tmp_path, name="demo")
    (first / "report.html").write_text("old report", encoding="utf-8")
    second = await make_run(path, tmp_path, name="demo", started_at=T0 + timedelta(hours=1))
    assert first == second and second.name == "demo"
    assert load_run(second).summary["started_at"].startswith("2026-10-11T15:30")
    assert not (second / "report.html").exists()  # it described the old results


async def test_named_run_refuses_a_folder_that_is_not_a_run(write_experiment, tmp_path):
    other = tmp_path / "runs" / "photos"
    other.mkdir(parents=True)
    (other / "holiday.txt").write_text("not a run", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a prompt-lab run"):
        await make_run(write_experiment(), tmp_path, name="photos")
    assert (other / "holiday.txt").read_text(encoding="utf-8") == "not a run"


@pytest.mark.parametrize("name", ["../escape", "a/b", "", ".hidden", "with space"])
async def test_run_name_must_be_a_plain_folder_name(write_experiment, tmp_path, name):
    with pytest.raises(ConfigError, match="invalid run name"):
        await make_run(write_experiment(), tmp_path, name=name)


async def test_two_runs_in_the_same_second_do_not_collide(write_experiment, tmp_path):
    path = write_experiment()
    first, second = await make_run(path, tmp_path), await make_run(path, tmp_path)
    assert first != second and second.name == first.name + "-2"
    assert load_run(second).run_id == second.name


async def test_list_and_resolve(write_experiment, tmp_path):
    path = write_experiment()
    old = await make_run(path, tmp_path, started_at=T0)
    new = await make_run(path, tmp_path, started_at=T0 + timedelta(days=1))
    named = await make_run(path, tmp_path, name="baseline", started_at=T0 - timedelta(days=1))
    runs_dir = tmp_path / "runs"
    (runs_dir / "not-a-run").mkdir()

    assert [p.name for p, _ in list_runs(runs_dir)] == [named.name, old.name, new.name]
    assert resolve_run(runs_dir, "latest") == new
    assert resolve_run(runs_dir, old.name) == old
    assert resolve_run(runs_dir, "20261012") == new          # unique prefix
    assert resolve_run(runs_dir, "base") == named
    assert resolve_run(runs_dir, str(old)) == old            # a path works too
    with pytest.raises(RunNotFoundError, match="matches several runs"):
        resolve_run(runs_dir, "2026101")
    with pytest.raises(RunNotFoundError, match="no run matches 'nope'"):
        resolve_run(runs_dir, "nope")


def test_empty_or_missing_runs_dir(tmp_path):
    assert list_runs(tmp_path / "missing") == []
    with pytest.raises(RunNotFoundError, match="no runs found"):
        resolve_run(tmp_path / "missing", "latest")
    with pytest.raises(RunNotFoundError, match="not a prompt-lab run"):
        load_run(tmp_path)
