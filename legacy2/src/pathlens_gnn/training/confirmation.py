from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from pathlens_gnn.constants import DEFAULT_SEEDS
from pathlens_gnn.training.runner import (
    train_experiment,
    training_config_from_trial,
    training_config_from_yaml,
)


def run_confirmation(
    processed: Path,
    output: Path,
    *,
    config_path: Path | None = None,
    leaderboard: Path | None = None,
    search_space: Path | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    if (config_path is None) == (leaderboard is None):
        raise ValueError("Provide exactly one of config_path or leaderboard")
    output.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        parameters = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        candidates = [{"parameters": parameters, "config_path": config_path}]
        spec: dict[str, Any] | None = None
    else:
        assert leaderboard is not None
        search_space = search_space or Path("configs/tuning/search_space.yaml")
        spec = yaml.safe_load(search_space.read_text(encoding="utf-8"))
        candidates = json.loads(leaderboard.read_text(encoding="utf-8"))[:2]
    summaries: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        parameters = candidate["parameters"]
        runs = []
        for seed in seeds:
            run_dir = output / f"candidate-{candidate_index}" / f"seed-{seed}"
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            elif config_path is not None:
                metrics = train_experiment(
                    processed,
                    run_dir,
                    training_config_from_yaml(config_path, seed=seed),
                )
            else:
                assert spec is not None
                metrics = train_experiment(
                    processed,
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
            "sample_sd_validation_hard_auprc": (
                statistics.stdev(auprc_values) if len(auprc_values) > 1 else 0.0
            ),
            "seed_13_checkpoint": str(
                output / f"candidate-{candidate_index}" / "seed-13/checkpoint.pt"
            ),
        }
        if mrr_values:
            summary["mean_validation_filtered_mrr"] = statistics.mean(mrr_values)
            summary["sample_sd_validation_filtered_mrr"] = (
                statistics.stdev(mrr_values) if len(mrr_values) > 1 else 0.0
            )
        summaries.append(summary)
    summaries.sort(key=_confirmation_key, reverse=True)
    payload = {"seeds": list(seeds), "candidates": summaries, "selected": summaries[0]}
    (output / "confirmation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _confirmation_key(item: dict[str, Any]) -> tuple[float, float]:
    return (
        float(item.get("mean_validation_filtered_mrr", 0.0)),
        float(item["mean_validation_hard_auprc"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm a registered ranking config or the two tuning leaders."
    )
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--leaderboard", type=Path, default=None)
    parser.add_argument(
        "--search-space", type=Path, default=Path("configs/tuning/search_space.yaml")
    )
    args = parser.parse_args()
    payload = run_confirmation(
        args.processed,
        args.output,
        config_path=args.config,
        leaderboard=args.leaderboard,
        search_space=args.search_space,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
