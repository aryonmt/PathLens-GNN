from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record the validation-selected checkpoint before opening the sealed test."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": "1.0",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "selection_metric": "validation_filtered_mrr",
        "tie_break": "validation_hard_auprc",
        "decision": args.decision,
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
