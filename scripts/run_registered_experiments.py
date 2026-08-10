from __future__ import annotations

import argparse
import time
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
    suite_start = time.perf_counter()
    model_count = len(registry["models"])
    print(
        f"[registered] Starting heuristic evaluation and {model_count} model runs",
        flush=True,
    )
    heuristics_output = args.output / "heuristics.json"
    if not heuristics_output.exists():
        heuristic_start = time.perf_counter()
        print("[registered] Running CPU heuristics", flush=True)
        evaluate_heuristics(args.processed, heuristics_output)
        print(
            f"[registered] CPU heuristics completed in "
            f"{time.perf_counter() - heuristic_start:.1f}s",
            flush=True,
        )
    else:
        print("[registered] Reusing completed CPU heuristics", flush=True)
    for index, experiment in enumerate(registry["models"], start=1):
        experiment_output = args.output / experiment["name"]
        if (experiment_output / "metrics.json").exists() and (
            experiment_output / "checkpoint.pt"
        ).exists():
            print(
                f"[registered] [{index}/{model_count}] Skipping completed "
                f"{experiment['name']}",
                flush=True,
            )
            continue
        print(
            f"[registered] [{index}/{model_count}] Starting {experiment['name']}",
            flush=True,
        )
        raw_config: dict[str, Any] = yaml.safe_load(
            Path(experiment["config"]).read_text(encoding="utf-8")
        )
        model_config = PathLensConfig(**raw_config.pop("model"))
        training_config = TrainingConfig(model=model_config, **raw_config)
        result = train_experiment(args.processed, experiment_output, training_config)
        print(
            f"[registered] [{index}/{model_count}] Completed {experiment['name']} "
            f"with validation AUPRC={result['validation_hard_auprc']:.4f}",
            flush=True,
        )
    print(
        f"[registered] Suite completed in {time.perf_counter() - suite_start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
