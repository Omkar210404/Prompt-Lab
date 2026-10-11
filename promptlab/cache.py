"""On-disk response cache, so re-running an experiment does not repeat paid calls.

Only successful responses are stored. Scores are never cached: scoring is cheap and local, so
changing a scorer and re-running costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from promptlab.providers.base import CompletionResponse


def cache_key(parts: dict[str, Any]) -> str:
    """SHA-256 of a provider's `cache_key_parts`, independent of dict ordering."""
    canonical = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """`enabled=False` neither reads nor writes. `refresh=True` ignores stored entries and overwrites them."""

    def __init__(self, directory: str | Path, enabled: bool = True, refresh: bool = False):
        self.directory = Path(directory)
        self.enabled = enabled
        self.refresh = refresh

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def get(self, key: str) -> CompletionResponse | None:
        if not self.enabled or self.refresh:
            return None
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
            return CompletionResponse(
                text=data["text"],
                model=data["model"],
                latency_ms=data["latency_ms"],
                usage=data.get("usage"),
                finish_reason=data.get("finish_reason"),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None  # missing or unreadable entry: treat as a miss and call the provider

    def put(self, key: str, response: CompletionResponse) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "text": response.text,
            "model": response.model,
            "latency_ms": response.latency_ms,
            "usage": response.usage,
            "finish_reason": response.finish_reason,
            "cached_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        # Write then rename, so a run killed mid-write cannot leave a half-written entry.
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
