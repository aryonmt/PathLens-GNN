# Kaggle

One notebook: `pathlens_training.ipynb`. Set `METHOD` to a folder name under
`methods/`. The committed default is `ranking_diagnostics` with `STAGE=eval`.
That rescores heuristics (and frozen PathLens) on GPU; it is not a freeze
candidate. Do not rerun filed cards. `STAGE=final` is refused until freeze.
`gat` and `nbfnet` stay deferred.

Internet must be on so the kernel can clone the branch **from GitHub** and
download BioSNAP. Push `research/ranking-loss` before you run.

| Stage | Heuristic / blend / RRF / `ranking_diagnostics` | `skipgnn` / `gcn` / `graphsage` / `residual_three_hop` |
|---|---|---|
| `smoke` | GPU score matrix, validation MRR + AUPRC; diagnostics skip PathLens | 2-epoch train + validation MRR/AUPRC |
| `train` | refused | full train, checkpoint, validation metrics |
| `eval` | full validation suite, or the tie/filter audit | full train + full validation suite |
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

## Filed: `blend_pathlens_three_hop` / `rrf_pathlens_three_hop` eval

`runs/biosnap-dti-v2/blend_pathlens_three_hop/eval/` and
`runs/biosnap-dti-v2/rrf_pathlens_three_hop/eval/` from
`outputs/kaggle/blend-eval/`. Blend selected `α=1` (MRR 0.455, hard AUPRC
0.887). RRF MRR 0.393, hard AUPRC 0.907. Do not rerun.

## Filed: `residual_three_hop` eval

`runs/biosnap-dti-v2/residual_three_hop/eval/` from
`outputs/kaggle/residual-eval/`. Best epoch 10 / stopped 18. Validation MRR
0.367, hard AUPRC 0.889. GPU train+eval 10.4s — not a truncated PathLens
v2 job. Loses filtered MRR to `three_hop`. Do not rerun.

No further Phase-1 GPU operator except `ranking_diagnostics`. Do not set
`STAGE=final`.

## Operator: `ranking_diagnostics` eval

1. Confirm `research/ranking-loss` is pushed.
2. New kernel. GPU T4. Internet on. No extra Input dataset.
3. `METHOD="ranking_diagnostics"`, `STAGE="eval"`, `DEVICE="cuda:0"`,
   `FINAL_TEST_TOKEN=""`.
4. Log should show `strict=` / `average=` / `random=` MRR per method, then
   PathLens scoring on GPU. `STAGE=train` is refused. Do not set `STAGE=final`.
5. Download `/kaggle/working/pathlens-stage-output.zip`. File `metrics.json`
   into `runs/biosnap-dti-v2/ranking_diagnostics/eval/`. Optional PNG:
   `figures/mrr_tie_break.png` stays in that eval folder, not the scoreboard.

## Filed: `graphsage` eval

`runs/biosnap-dti-v2/graphsage/eval/` from `outputs/kaggle/graphsage-eval/`.
Validation MRR 0.131, hard AUPRC 0.790. Hamilton mean-SAGE, not a Huang et al.
paper number. Do not rerun.

## Filed: `gcn` eval

`runs/biosnap-dti-v2/gcn/eval/` from `outputs/kaggle/gcn-eval/`.
Validation MRR 0.136, hard AUPRC 0.822.

## Filed: `skipgnn` eval

`runs/biosnap-dti-v2/skipgnn/eval/` from `outputs/kaggle/skipgnn-eval/`.
Validation MRR 0.144, hard AUPRC 0.824.
