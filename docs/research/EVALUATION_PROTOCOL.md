# Evaluation Protocol

## Canonical split

From the deduplicated positive universe, construct an approximately 60/20/10/10 context/train/validation/test split. An edge can leave context only when both endpoints retain context degree at least one. Store edge arrays, IDs, source checksum, algorithm version, and seed in an immutable manifest.

Assertions:

- context, train, validation, and test are pairwise disjoint;
- every supervised/evaluation endpoint appears in context;
- every pair is drug-to-protein;
- pseudo-entities, duplicates, and malformed IDs are absent.

## Negative sets

- Fixed 1:1 uniform typed unknowns for optimization and sampled evaluation.
- Fixed degree-matched hard unknowns from four context-degree quantiles.
- Exclude every known positive from every pool.
- Record deterministic fallbacks when a degree stratum is exhausted.

## Model selection

Primary: validation hard-negative AUPRC. Tie-break: exact filtered per-drug MRR. Test remains sealed until the complete selection procedure finishes.

Tuning ends after 24 configurations or four wall-clock days. The search covers embedding dimension, dropout, learning rate, weight decay, S2 weighting, and gate width. Initial trials use seed 13; the two leading configurations are rerun on seeds 13, 29, and 71.

## Reports

- AUROC/AUPRC on uniform and degree-matched unknowns.
- Validation-selected F1, Brier score, and expected calibration error.
- Exact filtered MRR and Hits@10/50 against all proteins per eligible drug.
- Degree quartile slices, training time, peak memory, and inference latency.
- Mean and sample standard deviation across seeds and stratified bootstrap intervals.

Sampled-nonedge metrics and ranking metrics must be named separately. A sigmoid is not labelled as biological interaction probability.
