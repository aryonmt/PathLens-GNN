from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any

import yaml

from pathlens_gnn.model.pathlens import PathLensConfig
from pathlens_gnn.training.runner import TrainingConfig, train_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--search-space", type=Path, default=Path("configs/tuning/search_space.yaml")
    )
    args = parser.parse_args()
    spec: dict[str, Any] = yaml.safe_load(args.search_space.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    trials = _registered_trials(spec)
    deadline = time.monotonic() + int(spec["budget"]["max_wall_seconds"])
    leaderboard: list[dict[str, Any]] = []

    for trial_index, parameters in enumerate(trials[: int(spec["budget"]["max_trials"])]):
        if time.monotonic() >= deadline:
            break
        trial_dir = args.output / f"trial-{trial_index:02d}"
        metrics_path = trial_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            model = PathLensConfig(
                embedding_dim=parameters["embedding_dim"],
                branch_dim=spec["fixed"]["branch_dim"],
                expert_hidden_dim=spec["fixed"]["expert_hidden_dim"],
                gate_hidden_dim=parameters["gate_hidden_dim"],
                dropout=parameters["dropout"],
                s2_weighting=parameters["s2_weighting"],
            )
            config = TrainingConfig(
                seed=spec["seed"],
                learning_rate=parameters["learning_rate"],
                weight_decay=parameters["weight_decay"],
                max_epochs=spec["fixed"]["max_epochs"],
                patience=spec["fixed"]["patience"],
                model=model,
            )
            metrics = train_experiment(args.processed, trial_dir, config)
        leaderboard.append(
            {
                "trial": trial_index,
                "parameters": parameters,
                "validation_hard_auprc": metrics["validation_hard_auprc"],
                "checkpoint": str(trial_dir / "checkpoint.pt"),
            }
        )
        leaderboard.sort(key=lambda item: item["validation_hard_auprc"], reverse=True)
        (args.output / "leaderboard.json").write_text(
            json.dumps(leaderboard, indent=2), encoding="utf-8"
        )
    print(json.dumps(leaderboard[:5], indent=2))


def _registered_trials(spec: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(spec["search"])
    combinations = [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(spec["search"][name] for name in names))
    ]
    random.Random(spec["seed"]).shuffle(combinations)
    return combinations


if __name__ == "__main__":
    main()
