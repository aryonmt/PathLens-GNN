from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathlens.constants import CAMPAIGN_ID
from pathlens.data.prepare import prepare_biosnap_dataset
from pathlens.importing.campaign_v2 import DEFAULT_V2_REPORT_ZIP, import_v2_report
from pathlens.runtime.runner import run_stage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pathlens")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Build the processed BioSNAP split")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=Path("data/processed") / CAMPAIGN_ID)
    prepare.add_argument("--seed", type=int, default=None)

    run = sub.add_parser("run", help="Run one method stage (smoke|train|eval|final)")
    run.add_argument("--method", required=True)
    run.add_argument("--stage", default="smoke")
    run.add_argument("--device", default="auto")
    run.add_argument("--raw", type=Path, default=None)
    run.add_argument("--processed", type=Path, default=None)
    run.add_argument("--no-download", action="store_true")
    run.add_argument("--final-test-token", default="", help="Required for STAGE=final")

    imported = sub.add_parser(
        "import-v2",
        help="Import locked campaign-v2 checkpoints (no retrain)",
    )
    imported.add_argument("--archive", type=Path, default=DEFAULT_V2_REPORT_ZIP)
    imported.add_argument("--output", type=Path, default=Path("runs") / CAMPAIGN_ID)
    imported.add_argument("--skip-weights", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        kwargs = {} if args.seed is None else {"seed": args.seed}
        prepare_biosnap_dataset(args.source, args.output, **kwargs)
        return 0
    if args.command == "import-v2":
        catalog = import_v2_report(
            args.archive,
            args.output,
            copy_weights=not args.skip_weights,
        )
        print(json.dumps(catalog, indent=2))
        return 0
    run_stage(
        args.method,
        args.stage,
        device=args.device,
        raw_path=args.raw,
        processed_dir=args.processed,
        download=not args.no_download,
        final_test_token=args.final_test_token,
    )
    return 0
