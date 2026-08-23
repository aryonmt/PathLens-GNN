# Evaluation Protocol

## Split

Campaign `biosnap-dti-v2` reuses the coverage-preserving ~60/20/10/10 context /
train / val / test split with seed 41 from `legacy2`. Every scored node has
context degree ≥ 1. Negative freeze is recorded in [`FREEZE.md`](FREEZE.md).
`STAGE=final` opens the test once for confirmation.

## Negatives

Unknown non-edges: typed uniform 1:1 and degree-matched hard 1:1. Known positives
never appear as negatives.

## Metric suite

All methods report the same suite on validation. After `STAGE=final` they
also report it on test, without changing the freeze decision.

- Filtered ranking: MRR, Hits@1/3/10/50, NDCG@10/50, per-query ranks
- Classification on uniform and hard banks: AUROC, AUPRC, PR/ROC points, F1 at
  a validation-selected threshold, Brier, ECE
- Degree slices
- Seed mean ± SD when multiple seeds exist; AUPRC bootstrap without an inner F1 search

## Figures

Eval means this figure set, not a single scalar. Copy Kaggle PNGs into
[`runs/biosnap-dti-v2/figures/`](../../runs/biosnap-dti-v2/figures/). See
[`runs/README.md`](../../runs/README.md) for the drop map. Do not generate them
on the laptop. Matplotlib is already on Kaggle; it is not a local dependency.

- `pr_hard.png` / `pr_uniform.png` — precision–recall on each negative bank
- `roc_hard.png` — ROC on hard negatives
- `hits_at_k.png` — filtered Hits@K
- `mrr_and_hard_auprc.png` — MRR vs hard AUPRC bars (comparison set)
- `degree_slices_hard.png` — hard AUPRC by pair-degree tertile
- `pathlens_family.png` — PathLens ablation ladder only
- `bipartite_hops.png`, `blend_alpha_sweep.png`, `epoch_selection.png`,
  `mrr_by_degree_tertile.png` — delivery extras
- `val_vs_test_mrr.png` — after `STAGE=final` only

Heuristics (`degree`, `resource_allocation`, `three_hop`) build the full
drug×protein score matrix with dense GEMM on the assigned GPU. They use context
edges only. The laptop path is NumPy (`--device cpu`) so CI does not need PyTorch.

Learning curves and gate-weight bars are added per trained method when history
exists. Test figures use the same names under a `test/` folder after freeze.

## Freeze

Select on validation filtered MRR. Tie-break: validation hard AUPRC. Freeze only
if the candidate beats `three_hop` and `skipgnn` on MRR. Otherwise report the
negative result. Open test at most once after freeze.

Filed ranks use **optimistic** ties: `1 + count(strictly higher scores)` after
masking other known positives. Exact ties do not move the target down. The
mask is `all_positive` (the canonical edge list), so test positives are hidden
as competitors on validation. `STAGE=eval` never scores test arrays.
`STAGE=final` scores them once. `ranking_diagnostics`
reports average-tie and random-tie MRR and a `visible` mask
(`context ∪ train ∪ val`) as columns beside the filed numbers. It does not
replace them.

A sigmoid is not a biological probability. Do not compare numbers to Huang et al.
SkipGNN PR-AUC 0.928 as a win or loss.
