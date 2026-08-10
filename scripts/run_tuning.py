from __future__ import annotations

import argparse
import hashlib
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
    state = _load_state(state_path)
    search_space_sha256 = _sha256(args.search_space)
    _validate_state(state, search_space_sha256, max_trials, max_wall_seconds)
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
            _write_json(args.output / "leaderboard.json", leaderboard)
            _write_state(
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
        _write_state(
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


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_state(
    state: dict[str, Any], search_space_sha256: str, max_trials: int, max_wall_seconds: int
) -> None:
    if not state:
        return
    registered = (
        state.get("search_space_sha256"),
        state.get("max_trials"),
        state.get("max_wall_seconds"),
    )
    current = (search_space_sha256, max_trials, max_wall_seconds)
    if registered != current:
        raise SystemExit(
            "Refusing to resume tuning with a changed registered budget or search space"
        )


def _write_state(
    path: Path,
    *,
    search_space_sha256: str,
    max_trials: int,
    max_wall_seconds: int,
    active_elapsed_seconds: float,
    completed_trials: int,
    status: str,
) -> None:
    _write_json(
        path,
        {
            "schema_version": "1.0",
            "search_space_sha256": search_space_sha256,
            "max_trials": max_trials,
            "max_wall_seconds": max_wall_seconds,
            "active_elapsed_seconds": active_elapsed_seconds,
            "completed_trials": completed_trials,
            "status": status,
        },
    )


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
