# outputs/

Gitignored drop zone for bulky Kaggle downloads. Git tracks only this README.

```
outputs/kaggle/<kernel-slug>/   ← `kaggle kernels output … -p` or a browser ZIP
```

Example:

```powershell
kaggle kernels output aryoamirrezanemati/notebooke2d6f990c4 -p "D:\Projects\SkipGNN\outputs\kaggle\notebooke2d6f990c4"
```

Then copy keepers into `runs/` as described in [`runs/README.md`](../runs/README.md).
Do not commit anything under `outputs/` except this file.
