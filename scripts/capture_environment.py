from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def capture_environment(repository: Path) -> dict[str, Any]:
    packages = {
        name: _package_version(distribution)
        for name, distribution in {
            "numpy": "numpy",
            "scipy": "scipy",
            "scikit_learn": "scikit-learn",
            "torch": "torch",
        }.items()
    }
    torch_details: dict[str, Any] = {"cuda_available": False, "cuda_version": None, "gpu": None}
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        torch_details = {
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        }
    except ImportError:
        pass

    return {
        "schema_version": "1.0",
        "captured_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch": torch_details,
        "git": {
            "commit": _git(repository, "rev-parse", "HEAD"),
            "ref": _git(repository, "describe", "--always", "--dirty"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the Kaggle training environment.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    payload = capture_environment(args.repository)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(repository: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


if __name__ == "__main__":
    main()
