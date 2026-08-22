from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pathlens import METHODS_DIR, method_ids


def load_method_config(method_id: str) -> dict[str, Any]:
    if method_id not in method_ids():
        raise KeyError(f"Unknown method {method_id!r}. Known: {', '.join(method_ids())}")
    path = Path(METHODS_DIR) / method_id / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    if raw.get("method") != method_id:
        raise ValueError(f"{path} method field must equal the folder name {method_id!r}")
    return raw
