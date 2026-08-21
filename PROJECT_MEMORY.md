# PathLens-GNN Session Handoff

Last updated: 2026-08-21 (Asia/Tehran)

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
- Campaign v2 selection: validation filtered MRR, with validation hard-negative AUPRC as the tie-break, on a sealed split.
- Training objective for campaign v2: sampled softmax over 64 typed negatives per positive (25% degree-matched mix). 1:1 BCE is a named ablation.
- Two-hop evidence is same-type projection context, not a drug–protein path. Only 1-hop and 3-hop connect opposite types.
- Model: sparse one-hop, resource-allocation two-hop projection, composed three-hop bridge, and pair-conditioned mixture of typed experts.
- Tuning remains optional. If the registered ranking configuration already beats the three-hop heuristic and SkipGNN-reimpl on validation, confirm seeds 13/29/71 and stop.
- After a split's test set is opened, never retune or reselect on that split.
- Training runs on Kaggle; local work writes and tests research code.

## Repository state

- `origin` is `https://github.com/aryonmt/PathLens-GNN.git`.
- PRs #1–#4 are merged. Research-only tree is `50c6943` on `main`.
- Active implementation branch: `research/ranking-loss`.
- `legacy/` contains the preserved original repository and nested Git history and must not be modified or committed.
- Canonical data, model, evaluation, and staged Kaggle workflow live in this repository.
- Local Kaggle archives are gitignored under `outputs/kaggle-stages/`.
- Campaign v2 registered archive filename: `outputs/kaggle-stages/pathlens-stage-output-registered.zip`.

## Scientific status

### Campaign v1 (`biosnap-dti-canonical-v1`, split seed 13)

Registered, tuning (24/24), confirmation (seeds 13/29/71), freeze, and the one-time sealed test are complete. That test is locked.

Frozen configuration (confirmation winner, seed-13 checkpoint):

- `embedding_dim=64`, `dropout=0.3`, `learning_rate=0.0003`, `weight_decay=0.0001`
- `s2_weighting=count`, `gate_hidden_dim=128`
- mean validation hard AUPRC 0.869301 (sample SD 0.003142)
- checkpoint SHA-256 `e1c2a6a6a2388da7ca95d1d621a0296cdf0d79b968277012b82e451bc4a56087`

One-shot sealed test (seed 13, do not retune on this split):

- test uniform AUPRC 0.935 (bootstrap 95% CI ~[0.926, 0.942]), AUROC 0.925, F1 0.835
- test hard AUPRC 0.867 (bootstrap 95% CI ~[0.854, 0.881]), AUROC 0.845, F1 0.762
- filtered ranking: MRR 0.189, Hits@10 0.333, Hits@50 0.547 (1513 queries)

Do not compare these numbers to the SkipGNN paper PR-AUC 0.928 as a win or loss. The protocols differ (negatives, split leakage, metrics). Paper-like uniform AUPRC is the easy setting here; hard-negative AUPRC and filtered MRR are the research targets.

### Campaign v2 (`biosnap-dti-canonical-v2`, split seed 41)

Bet 1 (sampled-softmax ranking loss) is in progress. The test set is still sealed. Do not restore a v1 stage ZIP into this campaign.

Registered seed-13 validation (commit `71a8b6a`, Tesla T4):

- normalized 3-hop heuristic hard AUPRC 0.847
- binary SkipGNN hard AUPRC 0.835
- PathLens + BCE hard AUPRC 0.852
- PathLens + ranking selected hard AUPRC 0.868, selected filtered MRR 0.375 (best epoch 289/300)
- ranking peak hard AUPRC during training was 0.888 at epoch 72; selection used MRR, not that peak

Next: three-seed confirmation of `configs/model/pathlens_ranking.yaml` (13/29/71). Skip the 24-trial tuning search unless confirmation fails. Freeze only if ranking still beats the three-hop heuristic and SkipGNN-reimpl on both validation metrics, then open the sealed test once.

## Non-negotiable scientific language

- Unknown non-edges are not verified negatives.
- Two-hop DTI evidence is same-type projection context, not a drug-to-protein path.
- Three-hop paths are structural support, not causal proof.
- Known edges must never appear as novel recommendations.
- A sigmoid is not a biological interaction probability.
