# Kaggle

One notebook: `pathlens_training.ipynb`. Set `METHOD` to a folder name under
`methods/`. The committed default is `graphsage` with `STAGE=eval`. Official
`skipgnn` and `gcn` are already filed — do not rerun them. `STAGE=final` is
refused until freeze.

Internet must be on so the kernel can clone the branch **from GitHub** and
download BioSNAP. Push `research/ranking-loss` before you run.

| Stage | Heuristic (`degree`, `resource_allocation`, `three_hop`) | `skipgnn` / `gcn` / `graphsage` |
|---|---|---|
| `smoke` | GPU score matrix, validation MRR + AUPRC | 2-epoch train + validation MRR/AUPRC |
| `train` | refused | full train, checkpoint, validation metrics |
| `eval` | full validation suite (curves, slices, bootstrap) | full train + full validation suite |
| `final` | refused; v2 test stays sealed | refused |

Imported PathLens cards live under `runs/` from a local `import-v2`. Do not retrain them on Kaggle. The last notebook cell plots every method that has `eval/`, `imported/`, or `smoke/` metrics and rebuilds `pathlens-stage-output.zip` so the PNGs are inside that archive.

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

## Operator: `graphsage` eval

1. Confirm the local branch is committed and pushed to `origin/research/ranking-loss`.
2. New kernel. GPU T4. Internet on. One method per session.
3. Use `kaggle/pathlens_training.ipynb`. Defaults:
   `METHOD="graphsage"`, `STAGE="eval"`, `DEVICE="cuda:0"`,
   `FINAL_TEST_TOKEN=""`.
4. Save & Run All. Log should show `device=cuda:0` and `[graphsage] epoch=...`.
   This is Hamilton mean-SAGE, not a Huang et al. paper number.
5. Download `/kaggle/working/pathlens-stage-output.zip`. File per
   [`runs/README.md`](../runs/README.md).
6. Do not set `STAGE=final`. Do not rerun `skipgnn` or `gcn`.

## Filed: `gcn` eval

`runs/biosnap-dti-v2/gcn/eval/` from `outputs/kaggle/gcn-eval/`.
Validation MRR 0.136, hard AUPRC 0.822.

## Filed: `skipgnn` eval

`runs/biosnap-dti-v2/skipgnn/eval/` from `outputs/kaggle/skipgnn-eval/`.
Validation MRR 0.144, hard AUPRC 0.824.
