from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathlens_gnn.artifact.export import export_inference_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--explanation-k", type=int, default=25)
    args = parser.parse_args()
    manifest = export_inference_artifact(
        args.processed,
        args.checkpoint,
        args.output,
        model_version=args.model_version,
        top_k=args.top_k,
        explanation_k=args.explanation_k,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
