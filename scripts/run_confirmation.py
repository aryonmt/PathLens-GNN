from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from pathlens_gnn.model.pathlens import PathLensConfig
from pathlens_gnn.training.runner import TrainingConfig, train_experiment

SEEDS = (13, 29, 71)


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirm the two leading configurations.")
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--search-space", type=Path, default=Path("configs/tuning/search_space.yaml")
    )
    args = parser.parse_args()
    leaders: list[dict[str, Any]] = json.loads(args.leaderboard.read_text(encoding="utf-8"))
    spec: dict[str, Any] = yaml.safe_load(args.search_space.read_text(encoding="utf-8"))
    summaries: list[dict[str, Any]] = []
    for candidate_index, leader in enumerate(leaders[:2]):
        parameters = leader["parameters"]
        runs = []
        for seed in SEEDS:
            run_dir = args.output / f"candidate-{candidate_index}" / f"seed-{seed}"
            metrics_path = run_dir / "metrics.json"
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
                metrics = train_experiment(
                    args.processed,
                    run_dir,
                    TrainingConfig(
                        seed=seed,
                        learning_rate=parameters["learning_rate"],
                        weight_decay=parameters["weight_decay"],
                        max_epochs=spec["fixed"]["max_epochs"],
                        patience=spec["fixed"]["patience"],
                        model=model,
                    ),
                )
            runs.append(metrics)
        values = [float(run["validation_hard_auprc"]) for run in runs]
        summaries.append(
            {
                "candidate": candidate_index,
                "parameters": parameters,
                "runs": runs,
                "mean_validation_hard_auprc": statistics.mean(values),
                "sample_sd_validation_hard_auprc": statistics.stdev(values),
                "seed_13_checkpoint": str(
                    args.output / f"candidate-{candidate_index}" / "seed-13/checkpoint.pt"
                ),
            }
        )
    summaries.sort(key=lambda item: item["mean_validation_hard_auprc"], reverse=True)
    payload = {"seeds": list(SEEDS), "candidates": summaries, "selected": summaries[0]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "confirmation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
