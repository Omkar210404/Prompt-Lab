"""Shared fixtures. Every test runs offline: real network connections are refused."""

import copy
import json
import socket
from pathlib import Path

import pytest
import yaml

from promptlab.config import ScorerConfig, TestCase
from promptlab.scorers import build_scorer

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that tries to leave the machine. Loopback stays open: asyncio needs it on Windows."""
    real_connect = socket.socket.connect

    def guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK:
            raise RuntimeError(f"test tried to open a network connection to {address!r}")
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


DEFAULT_CASES = [
    {"id": "c1", "vars": {"text": "great product"}, "expected": "positive", "tags": ["easy"]},
    {"id": "c2", "vars": {"text": "terrible service"}, "expected": "negative", "tags": ["easy"]},
    {"id": "c3", "vars": {"text": "it is fine I guess"}, "expected": "neutral", "tags": ["hard"]},
]

DEFAULT_EXPERIMENT = {
    "name": "test-exp",
    "dataset": "cases.jsonl",
    "prompts": [
        {"id": "p1", "template": "Classify: {text}"},
        {"id": "p2", "template": "Reply with one word. Classify: {text}"},
    ],
    "models": [{"id": "m1", "provider": "mock"}],
    "scorers": [{"type": "exact_match"}],
    "run": {"retry_base_delay": 0},
}


@pytest.fixture
def write_experiment(tmp_path):
    """Write an experiment file and its test set into tmp_path; returns the YAML path.

    `changes` is merged over the default experiment, `cases` replaces the default cases.
    """

    def _write(changes: dict | None = None, cases: list | None = None, name: str = "exp.yaml",
               raw_dataset: str | None = None) -> Path:
        experiment = copy.deepcopy(DEFAULT_EXPERIMENT)
        experiment.update(changes or {})
        dataset = tmp_path / experiment["dataset"]
        if raw_dataset is not None:
            dataset.write_text(raw_dataset, encoding="utf-8")
        else:
            lines = [json.dumps(case) for case in (DEFAULT_CASES if cases is None else cases)]
            dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path = tmp_path / name
        path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
        return path

    return _write


def make_case(expected=None, **fields) -> TestCase:
    return TestCase(id=fields.pop("id", "t1"), expected=expected, **fields)


def make_scorer(scorer_type: str, base_dir: Path | None = None, **options):
    return build_scorer(ScorerConfig(type=scorer_type, **options), base_dir or Path("."))
