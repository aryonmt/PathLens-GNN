from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from pathlens_gnn.evaluation.heuristics import evaluate_heuristics
from pathlens_gnn.model.pathlens import PathLensConfig
from pathlens_gnn.training.runner import TrainingConfig, train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered seed-13 baseline suite.")
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/experiments/registered.yaml")
    )
    args = parser.parse_args()
    registry: dict[str, Any] = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    heuristics_output = args.output / "heuristics.json"
    if not heuristics_output.exists():
        evaluate_heuristics(args.processed, heuristics_output)
    for experiment in registry["models"]:
        experiment_output = args.output / experiment["name"]
        if (experiment_output / "metrics.json").exists() and (
            experiment_output / "checkpoint.pt"
        ).exists():
            continue
        raw_config: dict[str, Any] = yaml.safe_load(
            Path(experiment["config"]).read_text(encoding="utf-8")
        )
        model_config = PathLensConfig(**raw_config.pop("model"))
        training_config = TrainingConfig(model=model_config, **raw_config)
        train_experiment(args.processed, experiment_output, training_config)


if __name__ == "__main__":
    main()
