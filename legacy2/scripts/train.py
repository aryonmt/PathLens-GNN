from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathlens_gnn.training.runner import train_experiment, training_config_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model/pathlens_full.yaml"))
    args = parser.parse_args()
    result = train_experiment(
        args.processed,
        args.output,
        training_config_from_yaml(args.config),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
