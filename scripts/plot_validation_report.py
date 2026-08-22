from __future__ import annotations

import argparse
from pathlib import Path

from pathlens.evaluation.figures import load_validation_report, write_validation_figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write validation figures from a report JSON, ZIP, or runs directory."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = load_validation_report(args.report)
    written = write_validation_figures(report, args.output)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
