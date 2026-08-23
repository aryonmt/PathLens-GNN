# Runs

Filed artifacts for campaign `biosnap-dti-v2`. Raw Kaggle dumps do **not** live
here; those go to [`outputs/`](../outputs/README.md) first.

```
runs/biosnap-dti-v2/
  figures/                 ← PNG scoreboard (copy from Kaggle /kaggle/working/figures)
    test/                  ← same names after freeze; empty until then
  imports/manifest.json    ← campaign-v2 import catalog
  <method>/imported/       ← locked PathLens cards (metrics.json; weights gitignored)
  <method>/eval/           ← Kaggle STAGE=eval metrics.json (this is the method card)
  <method>/train/          ← Kaggle STAGE=train only for methods we train here
  <method>/final/          ← Kaggle STAGE=final (validation + one-shot test)
```

## After a Kaggle session

1. Put the kernel ZIP in `outputs/kaggle/<kernel-slug>/`.
2. Copy `metrics.json` to `runs/biosnap-dti-v2/<method>/<stage>/`.
3. Copy `figures/*.png` from the ZIP to `runs/biosnap-dti-v2/figures/`.
4. Update `docs/STATUS.md` in the **same** change. Do not commit `*.pt` or the ZIP.

After `STAGE=final`, also copy `val_vs_test_mrr.png` and EDA PNGs. Do not
reselect a winner from test numbers.
