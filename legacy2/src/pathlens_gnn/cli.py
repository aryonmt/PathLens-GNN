from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from pathlens_gnn import __version__
from pathlens_gnn.constants import DEFAULT_SPLIT_SEED
from pathlens_gnn.data.prepare import prepare_biosnap_dataset

app = typer.Typer(help="PathLens-GNN research utilities.", no_args_is_help=True)


@app.command("version")
def version() -> None:
    """Print the installed PathLens-GNN version."""
    typer.echo(__version__)


@app.command("prepare-data")
def prepare_data(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("data/processed/biosnap-dti-v2"),
    seed: Annotated[int, typer.Option()] = DEFAULT_SPLIT_SEED,
) -> None:
    """Canonicalize BioSNAP and create leakage-safe split manifests."""
    manifest = prepare_biosnap_dataset(source, output, seed=seed)
    typer.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    app()
