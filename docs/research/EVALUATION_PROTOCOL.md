# Evaluation Protocol

## Canonical split

From the deduplicated positive universe, construct an approximately 60/20/10/10 context/train/validation/test split. An edge can leave context only when both endpoints retain context degree at least one. Store edge arrays, IDs, source checksum, algorithm version, and seed in an immutable manifest.

Assertions:

- context, train, validation, and test are pairwise disjoint;
- every supervised/evaluation endpoint appears in context;
- every pair is drug-to-protein;
- pseudo-entities, duplicates, and malformed IDs are absent.

Campaign `biosnap-dti-canonical-v1` used split seed 13. Its sealed test is opened and frozen. Campaign `biosnap-dti-canonical-v2` uses the same canonicalizer and split seed 41. Do not mix processed arrays across campaigns, and do not use the v1 test report for selection.

## Negative sets

- Evaluation still uses fixed 1:1 uniform typed unknowns and degree-matched hard unknowns.
- Ranking-loss training samples 64 typed non-edges per positive each epoch, with a 25% degree-quantile hard mix. These training negatives are not the evaluation sets.
- Exclude every known positive from every negative pool.
- Record deterministic fallbacks when a degree stratum is exhausted.

## Model selection

For the ranking-loss campaign: primary validation filtered per-drug MRR, with validation hard-negative AUPRC as the tie-break. Freeze only if the selected model also beats the normalized three-hop heuristic and the binary SkipGNN reimplementation on both metrics. Test remains sealed until freeze.

If the registered ranking configuration already beats those baselines on validation, skip the 24-trial search and confirm `configs/model/pathlens_ranking.yaml` on seeds 13, 29, and 71. Tuning remains available when that stop-early rule fails.

Tuning, when used, ends after 24 configurations or four wall-clock days. The search covers embedding dimension, dropout, learning rate, weight decay, S2 weighting, and gate width. Ranking-loss hyperparameters stay fixed (`num_negatives=64`, `hard_negative_fraction=0.25`, `softmax_temperature=1.0`). Initial trials use training seed 13; the two leading configurations are rerun on seeds 13, 29, and 71.

## Reports

- AUROC/AUPRC on uniform and degree-matched unknowns.
- Validation-selected F1, Brier score, and expected calibration error.
- Exact filtered MRR and Hits@10/50 against all proteins per eligible drug.
- Degree quartile slices, training time, peak memory, and inference latency.
- Mean and sample standard deviation across seeds and stratified bootstrap intervals.

Sampled-nonedge metrics and ranking metrics must be named separately. A sigmoid is not labelled as biological interaction probability.
