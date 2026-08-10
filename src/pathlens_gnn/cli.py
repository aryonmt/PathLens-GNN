from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from pathlens_gnn.data.prepare import prepare_biosnap_dataset

app = typer.Typer(help="PathLens-GNN research and inference utilities.")


@app.command("prepare-data")
def prepare_data(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("data/processed/biosnap-dti-v1"),
    seed: Annotated[int, typer.Option()] = 13,
) -> None:
    """Canonicalize BioSNAP and create leakage-safe split manifests."""
    manifest = prepare_biosnap_dataset(source, output, seed=seed)
    typer.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    app()
