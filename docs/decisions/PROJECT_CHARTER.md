# Project Charter

## Objective

Deliver a scientifically defensible extension of SkipGNN for transductive
drug–target interaction ranking on canonical BioSNAP. The work is a research
benchmark and model study, not a product or clinical application.

## Audiences

- University reviewers evaluating scientific method and reproducibility.
- Researchers exploring ranked candidates for follow-up, with no clinical interpretation.

## Required outcomes

1. A canonical typed dataset and leakage-safe benchmark.
2. Faithful structural and SkipGNN baselines.
3. A sparse path-aware model with controlled ablations.
4. Multi-seed confirmation, freeze, validation reporting artifacts, and a
   one-time sealed test. Hyperparameter tuning is optional when a preregistered
   champion already beats the registered baselines.
5. Honest reporting when an ablation or heuristic beats the adaptive model.

## Out of scope

Product UI, inference APIs, deployment, cold-start molecules or proteins, user
training, uploads, authentication, clinical claims, a mutable production
database, global full-graph rendering, and guaranteed uncertainty estimates.

## Success policy

Campaign v1 selected on validation hard-negative AUPRC, with filtered MRR as
the tie-break. Campaign v2 (ranking loss) selects on validation filtered MRR,
with hard-negative AUPRC as the tie-break. If the adaptive model does not beat a
registered ablation or heuristic after the registered budget, the negative result
is reported and the strongest valid candidate is frozen. After the sealed test is
opened, that split is never reused for selection.
