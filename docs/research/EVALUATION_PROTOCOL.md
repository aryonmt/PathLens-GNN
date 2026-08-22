# Evaluation Protocol

## Split

Campaign `biosnap-dti-v2` reuses the coverage-preserving ~60/20/10/10 context /
train / val / test split with seed 41 from `legacy2`. Every scored node has
context degree ≥ 1. The test set stays sealed.

## Negatives

Unknown non-edges: typed uniform 1:1 and degree-matched hard 1:1. Known positives
never appear as negatives.

## Metric suite

All methods report the same suite on validation until freeze:

- Filtered ranking: MRR, Hits@1/3/10/50, NDCG@10/50, per-query ranks
- Classification on uniform and hard banks: AUROC, AUPRC, PR/ROC points, F1 at
  a validation-selected threshold, Brier, ECE
- Degree slices
- Seed mean ± SD when multiple seeds exist; AUPRC bootstrap without an inner F1 search

## Freeze

Select on validation filtered MRR. Tie-break: validation hard AUPRC. Freeze only
if the candidate beats `three_hop` and `skipgnn` on MRR. Otherwise report the
negative result. Open test at most once after freeze.

A sigmoid is not a biological probability. Do not compare numbers to Huang et al.
SkipGNN PR-AUC 0.928 as a win or loss.
