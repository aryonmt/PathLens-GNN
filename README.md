# PathLens-GNN

PathLens-GNN is a leakage-aware, path-aware graph learning system for transductive
drug-target interaction (DTI) ranking on BioSNAP. This repository is a research
codebase. It is not a product, web application, or clinical tool.

> PathLens scores are research-prioritization signals. They are not validated
> biological interactions, medical advice, or clinical probabilities.

## Status

The repository contains the canonical BioSNAP DTI pipeline, sparse three-channel
model, registered baselines and ablations, and a staged Kaggle workflow
(`smoke` → `registered` → `confirmation` → `freeze` → `final`, with optional
`tuning` before confirmation).

A validation-selected configuration is frozen on campaign v1 (split seed 13).
That sealed test was opened once and must not be used for further selection.
Campaign v2 (split seed 41) is the current sealed ranking-loss campaign.
Confirmation uses `configs/model/pathlens_ranking.yaml` unless a later tuning
search produces a new leaderboard.

The archived SkipGNN repository remains locally under `legacy/` and is
intentionally excluded from Git.

## Quick start

```bash
uv sync --extra dev --extra train --frozen
uv run ruff check .
uv run mypy src
uv run pytest
```

Prepare the canonical dataset from a local BioSNAP TSV:

```bash
uv run pathlens prepare-data --source path/to/ChG-Miner_miner-chem-gene.tsv --seed 41
```

Training and the sealed evaluation run on Kaggle. See
[`kaggle/README.md`](kaggle/README.md) and
[`kaggle/pathlens_training.ipynb`](kaggle/pathlens_training.ipynb). The notebook
defaults to a three-epoch `smoke` stage, so Save & Run All cannot tune or open
the test set unless an operator changes `STAGE`.

## Documentation

- [Project charter](docs/decisions/PROJECT_CHARTER.md)
- [Legacy audit](docs/research/LEGACY_AUDIT.md)
- [Research specification](docs/research/RESEARCH_SPEC.md)
- [Evaluation protocol](docs/research/EVALUATION_PROTOCOL.md)
- [Literature and attribution](docs/research/LITERATURE_REVIEW.md)
- [Kaggle runbook](kaggle/README.md)

## Stack

Python 3.11/3.12, PyTorch (training extra), sparse SciPy operators, and
scikit-learn metrics. GPU training is intended for Kaggle.

## License and provenance

Code is licensed under BSD-3-Clause. See [NOTICE](NOTICE) for upstream SkipGNN
and dataset attribution. Dataset terms remain those of their respective
providers.
