"""Estimated cost from a user-editable price table.

prompt-lab ships no real prices: they change, and a stale number in a repo is a false claim.
Cost is an estimate from the token counts a provider reports and the prices you wrote down.
If either is missing the cost is None ("n/a"), never zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from promptlab.config import ModelConfig
from promptlab.errors import ConfigError

TOKENS_PER_PRICE_UNIT = 1_000_000


class Price(BaseModel):
    """USD per 1M tokens. Extra keys (note, source, checked) are kept for the reader."""

    model_config = ConfigDict(extra="allow")

    input: float = Field(ge=0)
    output: float = Field(ge=0)


def load_prices(path: str | Path | None) -> dict[str, Price]:
    """Read a price table: a YAML mapping of model name to {input, output}."""
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"prices file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: invalid YAML ({exc})") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: expected a mapping of model name to prices")
    prices: dict[str, Price] = {}
    for name, entry in raw.items():
        try:
            prices[str(name)] = Price.model_validate(entry)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
            raise ConfigError(f"{path.name}: price for '{name}': {problems}") from exc
    return prices


def price_for(prices: dict[str, Price], model: ModelConfig) -> Price | None:
    """Look a model up by the name sent to the provider, then by its id in the experiment."""
    return prices.get(model.model) or prices.get(model.id)


def estimate_cost(usage: dict[str, Any] | None, price: Price | None) -> float | None:
    if usage is None or price is None:
        return None
    prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    return (prompt_tokens * price.input + completion_tokens * price.output) / TOKENS_PER_PRICE_UNIT
