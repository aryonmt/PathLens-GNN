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
    heuristics_output = args.output / "heuristics.json"
    if not heuristics_output.exists():
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_heuristics.py",
                "--processed",
                str(args.processed),
                "--output",
                str(heuristics_output),
            ],
            check=True,
        )
    for experiment in registry["models"]:
        experiment_output = args.output / experiment["name"]
        if (experiment_output / "metrics.json").exists() and (
            experiment_output / "checkpoint.pt"
        ).exists():
            continue
        subprocess.run(
            [
                sys.executable,
                "scripts/train.py",
                "--processed",
                str(args.processed),
                "--output",
                str(experiment_output),
                "--config",
                experiment["config"],
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
