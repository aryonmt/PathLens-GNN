# Project Charter

## Objective

Deliver a scientifically defensible extension of SkipGNN and a polished inference-only application that ranks unrecorded interactions among drugs and proteins already represented in the BioSNAP DTI graph.

## Audiences

- University reviewers evaluating scientific method and engineering depth.
- Researchers exploring candidates for follow-up, with no clinical interpretation.
- Portfolio reviewers evaluating reproducibility, API design, visualization, and deployment.

## Required outcomes

1. A canonical typed dataset and leakage-safe benchmark.
2. Faithful structural and SkipGNN baselines.
3. A sparse path-aware model with controlled ablations and explanations.
4. Versioned, provenance-carrying inference artifacts.
5. Search, Top-K prioritization, pair evidence, export, and focused/ego 3D views.
6. Local Docker Compose and a best-effort public portfolio deployment.

## Out of scope for v1

Cold-start molecules/proteins, user training, uploads, authentication, clinical claims, a mutable production database, global full-graph rendering, and guaranteed uncertainty estimates.

## Success policy

The best validation-selected model is deployed. If the adaptive model does not beat an ablation after the registered tuning budget, the negative result is reported and the strongest valid ablation becomes the production model.
