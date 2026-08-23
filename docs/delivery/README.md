# How to write the final report from this repository

This folder is the English source pack for the university report and slides.
The PDF and PowerPoint are written outside the repo. Do not paste Huang et al.
PR-AUC 0.928 as a number to beat.

## Reading order

1. [`research/PAPER_CRITIQUE.md`](../research/PAPER_CRITIQUE.md) — report section 1.
2. [`research/EDA.md`](../research/EDA.md) — what the BioSNAP DTI graph actually is.
3. [`research/MODEL_CATALOG.md`](../research/MODEL_CATALOG.md) — exact architectures, losses, graphs, selection rules.
4. [`research/TECHNICAL_NARRATIVE.md`](../research/TECHNICAL_NARRATIVE.md) — why each experiment ran and what it changed.
5. [`research/FREEZE.md`](../research/FREEZE.md) — negative freeze, then one-shot test.
6. [`research/REPORT.md`](../research/REPORT.md) and [`STATUS.md`](../STATUS.md) — living numbers.
7. [`../runs/biosnap-dti-v2/figures/`](../../runs/biosnap-dti-v2/figures/) — plots. After Kaggle `STAGE=final`, also `figures/test/` and `val_vs_test_mrr.png`.

## What belongs in which chapter

| Your chapter | Use these files |
|---|---|
| Critique of SkipGNN | `PAPER_CRITIQUE.md` plus `src/pathlens/graph/skip.py` |
| Data | `EDA.md`, `DATA.md`, EDA PNGs from Kaggle |
| Method | `MODEL_CATALOG.md`, each `methods/<id>/METHOD.md` |
| Experiments | `TECHNICAL_NARRATIVE.md`, `REPORT.md`, scoreboard tables |
| Honest conclusion | L3/RA win filtered MRR; GNN 1:1 AUC does not buy ranking |

## Figure rules

- Head-to-head plots **exclude** nested PathLens hops (`one_hop`, `s1_s2`, `s1_s2_s3_fixed`).
- Those three cards plus `pathlens_bce` and `pathlens_ranking` are **one architecture**. Show them on `pathlens_family.png`.
- `mrr_and_hard_auprc.png` must include every comparison method: heuristics, SkipGNN, GCN, GraphSAGE, both PathLens heads, blend, RRF, residual.

## After the Kaggle delivery run

1. Put the kernel ZIP in `outputs/kaggle/<slug>/`.
2. Copy each `runs/biosnap-dti-v2/<method>/final/metrics.json`.
3. Copy PNGs into `runs/biosnap-dti-v2/figures/` (and `eda/`).
4. Paste test numbers into `REPORT.md` **without changing the winner**.
