# Kaggle

One notebook: `pathlens_training.ipynb`. Set `METHOD` to a folder name under
`methods/`. The committed default is `degree` with `STAGE=eval` so Save &
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

## Download kernel output (Windows)

`/path/to/dest` in Kaggle’s help is a placeholder, not a real folder. PowerShell
also will not see `kaggle` until the CLI is installed and `~\.local\bin` is on
PATH.

```powershell
uv tool install kaggle
kaggle auth login
New-Item -ItemType Directory -Force -Path "D:\Projects\SkipGNN\outputs\kaggle"
kaggle kernels output aryoamirrezanemati/notebooke2d6f990c4 -p "D:\Projects\SkipGNN\outputs\kaggle\notebooke2d6f990c4"
```

Open a **new** terminal after the first install so `kaggle` is on PATH. Login
opens a browser once.

File the download using [`runs/README.md`](../runs/README.md): raw ZIP under
`outputs/kaggle/<kernel-slug>/`, then `metrics.json` into
`runs/biosnap-dti-v2/<method>/<stage>/` and PNGs into
`runs/biosnap-dti-v2/figures/`. Do not commit `outputs/` dumps or
`pathlens-stage-output*.zip`.

## Operator: `degree` eval

1. Confirm the local branch is committed and pushed to `origin/research/ranking-loss`.
2. New kernel. GPU T4. Internet on. One method per session.
3. Use `kaggle/pathlens_training.ipynb`. Defaults:
   `METHOD="degree"`, `STAGE="eval"`, `DEVICE="cuda:0"`, `FINAL_TEST_TOKEN=""`.
4. Save & Run All. Log should show `device=cuda:0`. Compare MRR to `three_hop`
   (~0.455); degree is the popularity baseline, not the ranking leader.
5. File `/kaggle/working/pathlens-stage-output.zip` and `/kaggle/working/figures/`
   per [`runs/README.md`](../runs/README.md).
6. Do not set `STAGE=final`.

After `metrics.json` is in `runs/biosnap-dti-v2/degree/eval/`, flip
`docs/STATUS.md` in the same change. Then `resource_allocation`, then train
`skipgnn`.
