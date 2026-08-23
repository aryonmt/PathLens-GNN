# Kaggle

One notebook: `pathlens_training.ipynb`. Default `SUITE=delivery`, `STAGE=final`,
Tesla T4. One Save Version clones GitHub, writes EDA, scores every delivery
method on the sealed test, and rebuilds figures.

Internet must be on. Push `main` before you run.

| Stage | Heuristic / blend / imported PathLens | `skipgnn` / `gcn` / `graphsage` / `residual_three_hop` |
|---|---|---|
| `smoke` | GPU score matrix, validation MRR + AUPRC | 2-epoch train + validation MRR/AUPRC |
| `train` | refused | full train, checkpoint, validation metrics |
| `eval` | full validation suite | full train + full validation suite |
| `final` | validation + **one-shot test** | retrain (val select) + **one-shot test** |

`ranking_diagnostics` is validation-only. `gat` and `nbfnet` stay deferred.

Imported PathLens cards are scored from `runs/.../imported/checkpoint.pt`.
Do not retrain them. Blend also writes the RRF sibling. `α` is selected on
validation MRR, never on test.

`FINAL_TEST_TOKEN` must be `OPEN_SEALED_TEST_ONCE` for `STAGE=final`.
Set `SUITE = "single"` only to rerun one method.

## Download kernel output (Windows)

```powershell
uv tool install kaggle
kaggle auth login
New-Item -ItemType Directory -Force -Path "D:\Projects\SkipGNN\outputs\kaggle"
kaggle kernels output aryoamirrezanemati/notebooke2d6f990c4 -p "D:\Projects\SkipGNN\outputs\kaggle\notebooke2d6f990c4"
```

File with [`runs/README.md`](../runs/README.md): raw ZIP under
`outputs/kaggle/<kernel-slug>/`, then `metrics.json` into
`runs/biosnap-dti-v2/<method>/final/` and PNGs into
`runs/biosnap-dti-v2/figures/`. Do not commit `outputs/` dumps,
`pathlens-stage-output*.zip`, or GPU `model.pt`.

## Already filed (do not rerun as `eval`)

| Card | Where | Note |
|---|---|---|
| blend / RRF | `outputs/kaggle/blend-eval/` | α=1; RRF wins hard AUPRC, loses MRR |
| residual | `outputs/kaggle/residual-eval/` | best epoch 10, MRR 0.367 |
| ranking diagnostics | `outputs/kaggle/ranking-diagnostics-eval/` | average-tie does not change freeze |
| GraphSAGE | `outputs/kaggle/graphsage-eval/` | MRR 0.131 |
| GCN | `outputs/kaggle/gcn-eval/` | MRR 0.136 |
| SkipGNN | `outputs/kaggle/skipgnn-eval/` | MRR 0.144 |
