from __future__ import annotations

import argparse
from pathlib import Path

from pathlens_gnn.training.report import write_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write validation plot artifacts without opening the sealed test."
    )
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-record", type=Path, default=None)
    parser.add_argument("--registered", type=Path, default=None)
    args = parser.parse_args()
    write_validation_report(
        args.processed,
        args.output,
        checkpoint=args.checkpoint,
        freeze_record=args.freeze_record,
        registered=args.registered,
    )


if __name__ == "__main__":
    main()
