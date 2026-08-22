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
```

## After a Kaggle session

1. Put the kernel ZIP / CLI download in `outputs/kaggle/<kernel-slug>/`.
2. Copy `metrics.json` to `runs/biosnap-dti-v2/<method>/<stage>/`.
3. Copy the six PNGs to `runs/biosnap-dti-v2/figures/` (see that folder’s README).
4. Update `docs/STATUS.md` in the **same** change. Do not commit `*.pt` or the ZIP.

`STAGE=final` is refused until freeze. Do not create a `test/` metrics folder yet.
