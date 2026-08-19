from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from pathlens_gnn.training.runner import train_experiment, training_config_from_trial

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
                metrics = train_experiment(
                    args.processed,
                    run_dir,
                    training_config_from_trial(spec, parameters, seed=seed),
                )
            runs.append(metrics)
        auprc_values = [float(run["validation_hard_auprc"]) for run in runs]
        mrr_values = [
            float(run["validation_filtered_mrr"])
            for run in runs
            if "validation_filtered_mrr" in run
        ]
        summary = {
            "candidate": candidate_index,
            "parameters": parameters,
            "runs": runs,
            "mean_validation_hard_auprc": statistics.mean(auprc_values),
            "sample_sd_validation_hard_auprc": statistics.stdev(auprc_values),
            "seed_13_checkpoint": str(
                args.output / f"candidate-{candidate_index}" / "seed-13/checkpoint.pt"
            ),
        }
        if mrr_values:
            summary["mean_validation_filtered_mrr"] = statistics.mean(mrr_values)
            summary["sample_sd_validation_filtered_mrr"] = statistics.stdev(mrr_values)
        summaries.append(summary)
    summaries.sort(key=_confirmation_key, reverse=True)
    payload = {"seeds": list(SEEDS), "candidates": summaries, "selected": summaries[0]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "confirmation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _confirmation_key(item: dict[str, Any]) -> tuple[float, float]:
    return (
        float(item.get("mean_validation_filtered_mrr", 0.0)),
        float(item["mean_validation_hard_auprc"]),
    )


if __name__ == "__main__":
    main()
