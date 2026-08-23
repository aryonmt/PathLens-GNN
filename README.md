# PathLens-GNN

Comparative, leakage-safe drug–target ranking on BioSNAP. This is a research
codebase. It is not a product or a clinical tool.

> Scores are research-prioritization signals, not biological probabilities.

## Where to look

| Path | What it is |
|---|---|
| [`docs/delivery/README.md`](docs/delivery/README.md) | Reading order for the university report |
| [`docs/research/PAPER_CRITIQUE.md`](docs/research/PAPER_CRITIQUE.md) | Critique of Huang et al. SkipGNN |
| [`docs/STATUS.md`](docs/STATUS.md) | Scoreboard: every method, status, last run |
| [`methods/<id>/`](methods/) | Definition of one heuristic or model |
| [`runs/biosnap-dti-v2/`](runs/biosnap-dti-v2/) | Artifacts of actual runs ([`runs/README.md`](runs/README.md)) |
| [`src/pathlens/`](src/pathlens/) | Shared data, eval, and trainers |
| [`kaggle/`](kaggle/) | One notebook; `SUITE=delivery` opens test once |
| [`legacy1/`](legacy1/) | Original SkipGNN clone (gitignored) |
| [`legacy2/`](legacy2/) | Frozen PathLens campaigns v1–v2 |

Campaign `biosnap-dti-v2` uses split seed 41. Validation selected a **negative
freeze** (no learned model beat 3-hop and SkipGNN on MRR). `STAGE=final`
opens the test once for confirmation.

## Methods

Heuristics: `degree`, `resource_allocation`, `three_hop`.

Train here: `skipgnn` (BCE), `gcn`, `graphsage`, `residual_three_hop`.
Mixes: `blend_pathlens_three_hop`, `rrf_pathlens_three_hop`.
Diagnostic: `ranking_diagnostics` (validation only).
Deferred: `gat`, `nbfnet`.

Import, do not retrain: `one_hop`, `s1_s2`, `s1_s2_s3_fixed`, `pathlens_bce`,
`pathlens_ranking`. Nested hops are one architecture; head-to-head plots use
only the BCE and ranking heads.

## Quick start

Local tests do not need CUDA or BioSNAP:

```bash
uv sync --extra dev --frozen
uv run ruff check src tests methods
uv run pytest
```

```bash
python -m pathlens import-v2
```

Import reads a local v2 report ZIP if you still have one. The locked PathLens
family checkpoints are already in the repo under
`runs/biosnap-dti-v2/<method>/imported/checkpoint.pt`.

Training, EDA figures, and `STAGE=final` run on Kaggle GPU (Tesla T4). Push
`main` first. See [`kaggle/README.md`](kaggle/README.md).
