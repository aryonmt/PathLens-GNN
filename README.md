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

Train here (done): `skipgnn` (BCE), `gcn`, `graphsage`. New mixes:
`blend_pathlens_three_hop`, `rrf_pathlens_three_hop`, `residual_three_hop`.
Optional: `gat`, `nbfnet`.

Import, do not retrain (done): `one_hop`, `s1_s2`, `s1_s2_s3_fixed`, `pathlens_bce`, `pathlens_ranking`.

Default loss is BCE 1:1. Only `pathlens_ranking` uses sampled softmax.

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

Import reads the local v2 report ZIP, writes run cards under `runs/biosnap-dti-v2/<method>/imported/`, and does not open the test. Checkpoints are copied locally and gitignored.

Heuristic scoring and training run on Kaggle GPU (Tesla T4), one method at a time.
The notebook default is `blend_pathlens_three_hop` / `STAGE=eval`. Attach the
v2 report ZIP so the kernel can load the frozen PathLens checkpoint. Push the
branch before the kernel clones GitHub:

```bash
python -m pathlens run --method blend_pathlens_three_hop --stage eval --device cuda:0
```
