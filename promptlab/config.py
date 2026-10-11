"""Experiment definition: Pydantic models, YAML and JSONL loading, prompt templating and the config hash."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from promptlab.errors import ConfigError

# `{name}` is a variable. `{{name}}` is an escaped, literal `{name}`. Anything else in braces,
# such as a JSON example inside a few-shot prompt, is left exactly as written.
_PLACEHOLDER = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}")


def template_variables(template: str) -> set[str]:
    """Names of the variables a template needs."""
    return {match.group(2) for match in _PLACEHOLDER.finditer(template) if match.group(2)}


def render_template(template: str, variables: dict[str, Any]) -> str:
    """Fill `{name}` placeholders. Substituted values are not scanned again."""

    def replace(match: re.Match) -> str:
        if match.group(1):
            return "{" + match.group(1) + "}"
        name = match.group(2)
        if name not in variables:
            raise ConfigError(f"template variable {{{name}}} has no value")
        value = variables[name]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    return _PLACEHOLDER.sub(replace, template)


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    template: str = Field(min_length=1)
    system: str | None = None

    def variables(self) -> set[str]:
        return template_variables(self.template) | template_variables(self.system or "")


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    provider: str
    # Name sent to the provider. Defaults to `id`.
    model: str = ""
    base_url: str | None = None
    # NAME of the environment variable holding the key. The key itself never goes in a file.
    api_key_env: str | None = None
    # Sent to the provider as-is (temperature, max_tokens, ...).
    params: dict[str, Any] = Field(default_factory=dict)
    # Optional limit on parallel calls to this model, for strict free tiers.
    concurrency: int | None = Field(default=None, ge=1)
    # Rules for the mock provider. Ignored by other providers.
    mock: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_model_name(self) -> ModelConfig:
        if not self.model:
            self.model = self.id
        return self


class ScorerConfig(BaseModel):
    """`type` picks the scorer; every other key is passed to it as an option."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(min_length=1)
    # Label used in results. Defaults to the type.
    name: str | None = None
    # An advisory scorer (`required: false`) is reported but does not fail the case.
    required: bool = True

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concurrency: int = Field(default=4, ge=1)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    retry_base_delay: float = Field(default=1.0, ge=0)
    cache: bool = True


class TestCase(BaseModel):
    __test__ = False  # not a pytest class, despite the name

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    vars: dict[str, Any] = Field(default_factory=dict)
    expected: Any = None
    # Extra scorers for this case only, on top of the experiment-level ones.
    checks: list[ScorerConfig] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    dataset: str
    prompts: list[PromptConfig] = Field(min_length=1)
    models: list[ModelConfig] = Field(min_length=1)
    # Models used only by `llm_judge` scorers. They are not run over the test set.
    judges: list[ModelConfig] = Field(default_factory=list)
    scorers: list[ScorerConfig] = Field(default_factory=list)
    run: RunConfig = Field(default_factory=RunConfig)
    prices_file: str | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> Experiment:
        for label, ids in (
            ("prompt", [p.id for p in self.prompts]),
            ("model", [m.id for m in self.models]),
            ("judge", [j.id for j in self.judges]),
        ):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            if duplicates:
                raise ValueError(f"duplicate {label} id: {', '.join(duplicates)}")
        return self

    def prompt(self, prompt_id: str) -> PromptConfig:
        return next(p for p in self.prompts if p.id == prompt_id)

    def judge(self, judge_id: str) -> ModelConfig | None:
        """A judge is looked up in `judges` first, then in `models`."""
        for candidate in [*self.judges, *self.models]:
            if candidate.id == judge_id:
                return candidate
        return None


@dataclass
class LoadedExperiment:
    """An experiment file resolved against its test set, ready to run."""

    experiment: Experiment
    path: Path
    base_dir: Path
    cases: list[TestCase]
    dataset_path: Path
    dataset_meta: dict[str, Any]
    dataset_sha256: str
    config_hash: str
    prices_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"{location}: {message}" if location else message)
    return "; ".join(lines)


def sha256_text_file(path: Path) -> str:
    """SHA-256 of a text file with line endings normalised, so Windows and Linux checkouts agree."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_dataset(path: Path) -> tuple[list[TestCase], dict[str, Any]]:
    """Read a JSONL test set. A line of the form {"_meta": {...}} describes the set and is not a case."""
    if not path.is_file():
        raise ConfigError(f"dataset not found: {path}")

    cases: list[TestCase] = []
    meta: dict[str, Any] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path.name} line {line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise ConfigError(f"{path.name} line {line_number}: expected a JSON object")
        if "_meta" in record:
            if isinstance(record["_meta"], dict):
                meta.update(record["_meta"])
            continue
        record.setdefault("id", f"case-{len(cases) + 1:04d}")
        try:
            cases.append(TestCase.model_validate(record))
        except ValidationError as exc:
            raise ConfigError(f"{path.name} line {line_number}: {_format_validation_error(exc)}") from exc

    if not cases:
        raise ConfigError(f"{path.name}: the test set has no cases")
    ids = [case.id for case in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ConfigError(f"{path.name}: duplicate case id: {', '.join(duplicates)}")
    return cases, meta


def _file_options(scorers: list[ScorerConfig]) -> list[str]:
    return [value for cfg in scorers for key, value in cfg.options.items()
            if key.endswith("_file") and isinstance(value, str)]


def compute_config_hash(experiment: Experiment, cases: list[TestCase], dataset_sha256: str,
                        base_dir: Path) -> str:
    """Hash of everything that decides the outputs and the scores.

    Covers prompts, models, params, mock rules, judges, scorers, the test set and any files the
    scorers read. Run settings (concurrency, timeouts), prices and the description are left out,
    because they do not change what a run measures.
    """
    payload = experiment.model_dump(mode="json", exclude={"run", "prices_file", "description", "dataset"})
    payload["dataset_sha256"] = dataset_sha256
    all_scorers = [*experiment.scorers, *(check for case in cases for check in case.checks)]
    payload["files"] = {
        name: sha256_text_file(base_dir / name)
        for name in sorted(set(_file_options(all_scorers)))
        if (base_dir / name).is_file()
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_variables(experiment: Experiment, cases: list[TestCase]) -> None:
    for prompt in experiment.prompts:
        needed = prompt.variables()
        for case in cases:
            missing = sorted(needed - set(case.vars))
            if missing:
                names = ", ".join("{" + name + "}" for name in missing)
                raise ConfigError(
                    f"prompt '{prompt.id}' uses {names} but test case '{case.id}' does not provide it. "
                    "Add it to the case's vars, or write {{name}} for a literal brace."
                )


def _check_scorers(experiment: Experiment, cases: list[TestCase], base_dir: Path) -> list[str]:
    """Build every scorer once so bad options fail at load time, not halfway through a run."""
    from promptlab.scorers import build_scorer

    warnings: list[str] = []
    for cfg in experiment.scorers:
        build_scorer(cfg, base_dir)
    for case in cases:
        if not experiment.scorers and not case.checks:
            raise ConfigError(
                f"test case '{case.id}' has no scorers: add experiment-level `scorers` or per-case `checks`"
            )
        for cfg in case.checks:
            try:
                build_scorer(cfg, base_dir)
            except ConfigError as exc:
                raise ConfigError(f"test case '{case.id}': {exc}") from exc

    tested = {(m.provider, m.model, m.base_url) for m in experiment.models}
    judge_configs = [*experiment.scorers, *(check for case in cases for check in case.checks)]
    for judge_id in sorted({cfg.options.get("judge") for cfg in judge_configs if cfg.type == "llm_judge"}):
        judge = experiment.judge(judge_id)
        if judge is None:
            raise ConfigError(f"llm_judge: judge '{judge_id}' is not defined under `judges` or `models`")
        if (judge.provider, judge.model, judge.base_url) in tested:
            warnings.append(
                f"judge '{judge_id}' is the same model as one being tested ({judge.model}). "
                "Models tend to rate their own output favourably; prefer a different judge."
            )
    return warnings


def _check_providers(experiment: Experiment) -> None:
    from promptlab.providers import provider_types

    known = provider_types()
    for model in [*experiment.models, *experiment.judges]:
        if model.provider not in known:
            raise ConfigError(
                f"model '{model.id}': unknown provider '{model.provider}' (available: {', '.join(known)})"
            )


def load_experiment(path: str | Path) -> LoadedExperiment:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"experiment file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: invalid YAML ({exc})") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: expected a YAML mapping at the top level")
    try:
        experiment = Experiment.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path.name}: {_format_validation_error(exc)}") from exc

    base_dir = path.resolve().parent
    dataset_path = base_dir / experiment.dataset
    cases, meta = load_dataset(dataset_path)

    _check_variables(experiment, cases)
    _check_providers(experiment)
    warnings = _check_scorers(experiment, cases, base_dir)

    prices_path = None
    if experiment.prices_file:
        prices_path = base_dir / experiment.prices_file
        if not prices_path.is_file():
            raise ConfigError(f"prices file not found: {prices_path}")

    dataset_sha256 = sha256_text_file(dataset_path)
    return LoadedExperiment(
        experiment=experiment,
        path=path.resolve(),
        base_dir=base_dir,
        cases=cases,
        dataset_path=dataset_path,
        dataset_meta=meta,
        dataset_sha256=dataset_sha256,
        config_hash=compute_config_hash(experiment, cases, dataset_sha256, base_dir),
        prices_path=prices_path,
        warnings=warnings,
    )


def override_experiment(loaded: LoadedExperiment, prompts: list[dict[str, Any]] | None = None,
                        cases: list[TestCase] | None = None) -> LoadedExperiment:
    """A copy of a loaded experiment with different prompts and/or test cases, validated the same way.

    The web UI uses this: prompts are edited in the browser, and a single typed input is a test case.
    """
    data = loaded.experiment.model_dump(mode="json")
    if prompts is not None:
        data["prompts"] = prompts
    try:
        experiment = Experiment.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc
    cases = loaded.cases if cases is None else cases
    _check_variables(experiment, cases)
    warnings = _check_scorers(experiment, cases, loaded.base_dir)
    return replace(
        loaded,
        experiment=experiment,
        cases=cases,
        config_hash=compute_config_hash(experiment, cases, loaded.dataset_sha256, loaded.base_dir),
        warnings=warnings,
    )
