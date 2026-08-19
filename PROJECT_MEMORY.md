# PathLens-GNN Session Handoff

Last updated: 2026-08-19 (Asia/Tehran)

## Read order

1. `README.md`
2. `docs/decisions/PROJECT_CHARTER.md`
3. `docs/research/RESEARCH_SPEC.md`
4. `docs/research/EVALUATION_PROTOCOL.md`
5. `kaggle/README.md`

## Locked decisions

- Primary task: transductive BioSNAP DTI link ranking / research priority, not biological probability.
- This repository is research-only. There is no product app, API, or deployment surface.
- PPI, cold-start inputs, global UMAP, ONNX, ensembles, and multimodal chemistry/ESM are out of scope until the ranking protocol is stronger.
- Split: disjoint coverage-preserving ~60/20/10/10 context / train / val / test. Context graph is not the train-edge set.
- Every scored node has context degree ≥ 1. No cold-start evaluation.
- Negatives are unknown non-edges (typed uniform and degree-matched hard), not experimental false interactions.
- Selection: validation hard-negative AUPRC; filtered MRR is the tie-break.
- Two-hop evidence is same-type projection context, not a drug–protein path. Only 1-hop and 3-hop connect opposite types.
- Model: sparse one-hop, resource-allocation two-hop projection, composed three-hop bridge, and pair-conditioned mixture of typed experts.
- Tuning: at most 24 validation configurations or four wall-clock days.
- After a split's test set is opened, never retune or reselect on that split.
- Training and tuning run on Kaggle; local work writes and tests research code.

## Repository state

- `origin` is `https://github.com/aryonmt/PathLens-GNN.git`.
- PRs #1–#4 are merged. Current Kaggle runs used commit `6613633`.
- `legacy/` contains the preserved original repository and nested Git history and must not be modified or committed.
- Canonical data, model, evaluation, and staged Kaggle workflow live in this repository.
- Local Kaggle archives are gitignored under `outputs/kaggle-stages/`.

## Scientific status

Registered, tuning (24/24), confirmation (seeds 13/29/71), freeze, and the one-time sealed test are complete for the current split.

Frozen configuration (confirmation winner, seed-13 checkpoint):

- `embedding_dim=64`, `dropout=0.3`, `learning_rate=0.0003`, `weight_decay=0.0001`
- `s2_weighting=count`, `gate_hidden_dim=128`
- mean validation hard AUPRC 0.869301 (sample SD 0.003142)
- checkpoint SHA-256 `e1c2a6a6a2388da7ca95d1d621a0296cdf0d79b968277012b82e451bc4a56087`

The larger seed-13 tuning leader (trial 11) was less stable across seeds and was not frozen.

One-shot sealed test (seed 13, do not retune on this split):

- test uniform AUPRC 0.935 (bootstrap 95% CI ~[0.926, 0.942]), AUROC 0.925, F1 0.835
- test hard AUPRC 0.867 (bootstrap 95% CI ~[0.854, 0.881]), AUROC 0.845, F1 0.762
- filtered ranking: MRR 0.189, Hits@10 0.333, Hits@50 0.547 (1513 queries)

Do not compare these numbers to the SkipGNN paper PR-AUC 0.928 as a win or loss. The protocols differ (negatives, split leakage, metrics). Paper-like uniform AUPRC is the easy setting here; hard-negative AUPRC and filtered MRR are the research targets.

## Next research work

Future architecture bets (ranking loss, then frozen S3 residual, then pairwise 3-hop decoder only if needed) belong on a new branch after this research-only tree is pushed. They require a new preregistered coverage-preserving split with a sealed test. The current test report is frozen evidence, not a tuning signal.

## Non-negotiable scientific language

- Unknown non-edges are not verified negatives.
- Two-hop DTI evidence is same-type projection context, not a drug-to-protein path.
- Three-hop paths are structural support, not causal proof.
- Known edges must never appear as novel recommendations.
- A sigmoid is not a biological interaction probability.
