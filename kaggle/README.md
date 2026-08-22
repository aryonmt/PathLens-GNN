# Kaggle

One notebook: `pathlens_training.ipynb`. Set `METHOD` to a folder name under
`methods/`. The committed default is `three_hop` with `STAGE=eval` so Save &
Run All completes the heuristic validation card and does **not** open the test.
`STAGE=final` is refused until freeze.

Internet must be on so the kernel can clone the branch **from GitHub** and
download BioSNAP. Push `research/ranking-loss` before you run.

| Stage | Heuristic (`degree`, `resource_allocation`, `three_hop`) | Trained methods |
|---|---|---|
| `smoke` | GPU score matrix, validation MRR + AUPRC | not implemented yet |
| `train` | refused | not implemented yet |
| `eval` | full validation suite (curves, slices, bootstrap) | not implemented yet |
| `final` | refused; v2 test stays sealed | refused |

Imported PathLens cards live under `runs/` from a local `import-v2`. Do not retrain them on Kaggle. The last notebook cell plots every method that has `eval/`, `imported/`, or `smoke/` metrics.

Use one GPU process per session for now. Do not DataParallel.

`FINAL_TEST_TOKEN` stays empty until freeze.

## Operator: `three_hop` eval (next run)

1. Confirm the local branch is committed and pushed to `origin/research/ranking-loss`.
2. New kernel. GPU T4. Internet on. Do not run a second method in parallel.
3. Upload `kaggle/pathlens_training.ipynb` (or copy the cells). Leave the defaults:
   `METHOD="three_hop"`, `STAGE="eval"`, `DEVICE="cuda:0"`, `FINAL_TEST_TOKEN=""`.
4. Save & Run All. Expected log line: `device=cuda:0` and MRR near the v2
   reference `~0.455` (same split, new-system scorer).
5. Download `/kaggle/working/pathlens-stage-output.zip` and the
   `/kaggle/working/figures/` PNGs.
6. Do not set `STAGE=final`.

After the ZIP is on the laptop, copy `metrics.json` into
`runs/biosnap-dti-v2/three_hop/eval/` and flip `docs/STATUS.md` in the same
change. Then run `degree` the same way, then `resource_allocation`, then train
`skipgnn`.
