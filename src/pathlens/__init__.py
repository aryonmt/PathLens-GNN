"""Shared toolkit for leakage-safe BioSNAP DTI comparison."""

from __future__ import annotations

from pathlib import Path

__all__ = ["REPO_ROOT", "METHODS_DIR", "method_ids"]

REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS_DIR = REPO_ROOT / "methods"


def method_ids() -> tuple[str, ...]:
    if not METHODS_DIR.is_dir():
        return ()
    return tuple(
        sorted(
            path.name
            for path in METHODS_DIR.iterdir()
            if path.is_dir() and (path / "METHOD.md").is_file()
        )
    )
