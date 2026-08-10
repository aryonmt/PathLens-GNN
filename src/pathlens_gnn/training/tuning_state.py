from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_state(
    state: dict[str, Any], search_space_sha256: str, max_trials: int, max_wall_seconds: int
) -> None:
    if not state:
        return
    registered = (
        state.get("search_space_sha256"),
        state.get("max_trials"),
        state.get("max_wall_seconds"),
    )
    current = (search_space_sha256, max_trials, max_wall_seconds)
    if registered != current:
        raise ValueError("Refusing to resume with a changed registered budget or search space")


def write_state(
    path: Path,
    *,
    search_space_sha256: str,
    max_trials: int,
    max_wall_seconds: int,
    active_elapsed_seconds: float,
    completed_trials: int,
    status: str,
) -> None:
    write_json(
        path,
        {
            "schema_version": "1.0",
            "search_space_sha256": search_space_sha256,
            "max_trials": max_trials,
            "max_wall_seconds": max_wall_seconds,
            "active_elapsed_seconds": active_elapsed_seconds,
            "completed_trials": completed_trials,
            "status": status,
        },
    )


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
