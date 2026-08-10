from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathlens_gnn.evaluation.heuristics import evaluate_heuristics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate registered non-learned DTI baselines.")
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = evaluate_heuristics(args.processed, args.output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
