# PathLens-GNN

Comparative, leakage-safe drug–target ranking on BioSNAP. This is a research
codebase. It is not a product or a clinical tool.

> Scores are research-prioritization signals, not biological probabilities.

## Where to look

| Path | What it is |
|---|---|
| [`docs/STATUS.md`](docs/STATUS.md) | Scoreboard: every method, status, last run |
| [`methods/<id>/`](methods/) | Definition of one heuristic or model |
| [`runs/biosnap-dti-v2/`](runs/biosnap-dti-v2/) | Artifacts of actual runs |
| [`src/pathlens/`](src/pathlens/) | Shared data, eval, and trainers |
| [`kaggle/`](kaggle/) | One notebook; set `METHOD` |
| [`legacy1/`](legacy1/) | Original SkipGNN clone (gitignored) |
| [`legacy2/`](legacy2/) | Frozen PathLens campaigns v1–v2 |

Campaign `biosnap-dti-v2` uses split seed 41. The test set is sealed.

## Phase 1 methods

Heuristics: `degree`, `resource_allocation`, `three_hop`.

Train here: `skipgnn` (BCE), then `gcn` and `graphsage`. Optional: `gat`, `nbfnet`.

Import, do not retrain: `one_hop`, `s1_s2`, `s1_s2_s3_fixed`, `pathlens_bce`, `pathlens_ranking`.

Default loss is BCE 1:1. Only `pathlens_ranking` uses sampled softmax.

## Quick start

```bash
uv sync --extra dev --frozen
uv run ruff check src tests methods
uv run pytest
```

Training runs on Kaggle GPU (Tesla T4). Use two processes on `cuda:0` and `cuda:1`
for two methods; do not DataParallel a single small graph.
