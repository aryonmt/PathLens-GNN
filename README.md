# PathLens-GNN

Comparative, leakage-safe drug–target ranking on BioSNAP. This is a research
codebase. It is not a product or a clinical tool.

> Scores are research-prioritization signals, not biological probabilities.

## Where to look

| Path | What it is |
|---|---|
| [`docs/STATUS.md`](docs/STATUS.md) | Scoreboard: every method, status, last run |
| [`methods/<id>/`](methods/) | Definition of one heuristic or model |
| [`runs/biosnap-dti-v2/`](runs/biosnap-dti-v2/) | Artifacts of actual runs ([`runs/README.md`](runs/README.md)) |
| [`src/pathlens/`](src/pathlens/) | Shared data, eval, and trainers |
| [`kaggle/`](kaggle/) | One notebook; set `METHOD` |
| [`legacy1/`](legacy1/) | Original SkipGNN clone (gitignored) |
| [`legacy2/`](legacy2/) | Frozen PathLens campaigns v1–v2 |

Campaign `biosnap-dti-v2` uses split seed 41. The test set is sealed.

## Phase 1 methods

Heuristics: `degree`, `resource_allocation`, `three_hop`.

Train here (done): `skipgnn` (BCE), `gcn`, `graphsage`. Filed mixes:
`blend_pathlens_three_hop`, `rrf_pathlens_three_hop`. Next: `residual_three_hop`.
Optional: `gat`, `nbfnet`.

Import, do not retrain (done): `one_hop`, `s1_s2`, `s1_s2_s3_fixed`, `pathlens_bce`, `pathlens_ranking`.

Default loss is BCE 1:1. Ranking-loss cards: imported `pathlens_ranking` and
unfiled `residual_three_hop`.

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
`runs/biosnap-dti-v2/<method>/imported/checkpoint.pt`. It does not open the test.

Heuristic scoring and training run on Kaggle GPU (Tesla T4), one method at a time.
The notebook default is `residual_three_hop` / `STAGE=eval`. Push the branch
before the kernel clones GitHub:

```bash
python -m pathlens run --method residual_three_hop --stage eval --device cuda:0
```
