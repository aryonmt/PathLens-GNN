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
from pathlens_gnn.training.tuning_state import (
    load_state,
    sha256_file,
    validate_state,
    write_json,
    write_state,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--search-space", type=Path, default=Path("configs/tuning/search_space.yaml")
    )
    parser.add_argument(
        "--session-seconds",
        type=int,
        default=None,
        help="Stop this invocation early so a hosted notebook can persist resumable outputs.",
    )
    args = parser.parse_args()
    if args.session_seconds is not None and args.session_seconds <= 0:
        parser.error("--session-seconds must be positive")
    spec: dict[str, Any] = yaml.safe_load(args.search_space.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    trials = _registered_trials(spec)
    max_trials = int(spec["budget"]["max_trials"])
    max_wall_seconds = int(spec["budget"]["max_wall_seconds"])
    state_path = args.output / "run_state.json"
    state = load_state(state_path)
    search_space_sha256 = sha256_file(args.search_space)
    validate_state(state, search_space_sha256, max_trials, max_wall_seconds)
    previous_elapsed = float(state.get("active_elapsed_seconds", 0.0))
    invocation_started = time.monotonic()
    leaderboard: list[dict[str, Any]] = []

    try:
        for trial_index, parameters in enumerate(trials[:max_trials]):
            invocation_elapsed = time.monotonic() - invocation_started
            elapsed = previous_elapsed + invocation_elapsed
            if elapsed >= max_wall_seconds:
                break
            if args.session_seconds is not None and invocation_elapsed >= args.session_seconds:
                break
            trial_dir = args.output / f"trial-{trial_index:02d}"
            metrics_path = trial_dir / "metrics.json"
            checkpoint_path = trial_dir / "checkpoint.pt"
            if metrics_path.exists() and checkpoint_path.exists():
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
                    "checkpoint": str(checkpoint_path),
                }
            )
            leaderboard.sort(key=lambda item: item["validation_hard_auprc"], reverse=True)
            write_json(args.output / "leaderboard.json", leaderboard)
            write_state(
                state_path,
                search_space_sha256=search_space_sha256,
                max_trials=max_trials,
                max_wall_seconds=max_wall_seconds,
                active_elapsed_seconds=previous_elapsed + time.monotonic() - invocation_started,
                completed_trials=len(leaderboard),
                status="running",
            )
    finally:
        total_elapsed = previous_elapsed + time.monotonic() - invocation_started
        completed_trials = sum(
            (args.output / f"trial-{index:02d}" / "metrics.json").exists()
            and (args.output / f"trial-{index:02d}" / "checkpoint.pt").exists()
            for index in range(max_trials)
        )
        status = (
            "budget_complete"
            if completed_trials >= max_trials or total_elapsed >= max_wall_seconds
            else "paused"
        )
        write_state(
            state_path,
            search_space_sha256=search_space_sha256,
            max_trials=max_trials,
            max_wall_seconds=max_wall_seconds,
            active_elapsed_seconds=total_elapsed,
            completed_trials=completed_trials,
            status=status,
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
