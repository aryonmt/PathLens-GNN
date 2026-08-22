from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathlens_gnn.training.report import write_final_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-time sealed PathLens evaluation.")
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--freeze-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-sealed-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_sealed_test:
        raise SystemExit("Refusing to open test data without --confirm-sealed-test")
    print("[final] Freeze record matched. Opening sealed test...", flush=True)
    payload = write_final_artifacts(
        args.processed,
        args.checkpoint,
        args.output,
        freeze_record=args.freeze_record,
    )
    print("[final] Sealed evaluation finished.", flush=True)
    compact = {
        name: {
            "hard_auprc": model["classification"]["hard"]["auprc"],
            "uniform_auprc": model["classification"]["uniform"]["auprc"],
            "mrr": model["filtered_ranking"]["mrr"],
            "hits_at_10": model["filtered_ranking"]["hits_at_10"],
            "hits_at_50": model["filtered_ranking"]["hits_at_50"],
        }
        for name, model in payload["models"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
