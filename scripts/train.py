from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pathlens_gnn.model.pathlens import PathLensConfig
from pathlens_gnn.training.runner import TrainingConfig, train_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model/pathlens_full.yaml"))
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_config = PathLensConfig(**raw.pop("model"))
    config = TrainingConfig(model=model_config, **raw)
    result = train_experiment(args.processed, args.output, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
