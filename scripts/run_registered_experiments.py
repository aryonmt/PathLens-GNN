from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered seed-13 baseline suite.")
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/experiments/registered.yaml")
    )
    args = parser.parse_args()
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_heuristics.py",
            "--processed",
            str(args.processed),
            "--output",
            str(args.output / "heuristics.json"),
        ],
        check=True,
    )
    for experiment in registry["models"]:
        subprocess.run(
            [
                sys.executable,
                "scripts/train.py",
                "--processed",
                str(args.processed),
                "--output",
                str(args.output / experiment["name"]),
                "--config",
                experiment["config"],
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
