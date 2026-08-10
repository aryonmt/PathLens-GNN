# PathLens-GNN Session Handoff

Last updated: 2026-08-09 (Asia/Tehran)

## Read order

1. `README.md`
2. `docs/decisions/PROJECT_CHARTER.md`
3. `docs/research/RESEARCH_SPEC.md`
4. `docs/research/EVALUATION_PROTOCOL.md`
5. `docs/contracts/API_AND_ARTIFACT_CONTRACT.md`
6. `docs/delivery/KANBAN_PLAYBOOK.md`

## Locked decisions

- Primary task: transductive BioSNAP DTI link prioritization.
- PPI, cold-start inputs, global UMAP, ONNX, and ensembles are stretch work.
- The product performs inference only and displays a PathLens research-priority score, not a biological probability.
- Core visualization: focused pair explanation and capped one-to-three-hop ego graph.
- Data split: disjoint context/train/validation/test edge sets with context coverage preservation.
- Model: sparse one-hop, resource-allocation two-hop projection context, composed three-hop bridge evidence, and pair-conditioned mixture of typed experts.
- Tuning: at most 24 validation configurations or four wall-clock days; the test set remains sealed.
- Training and tuning run on Kaggle; local work writes/tests code and the inference product.
- Deployment: local Docker Compose plus Vercel static frontend and a single-worker, NumPy-only Render backend.
- GitHub delivery is dependency-driven Kanban; the two-week target is soft and there is no day-by-day schedule.

## Repository state

- PR #1 was squash-merged as commit `673e0c0`; `main` is the authoritative implementation branch.
- Active local branch: `fix/kaggle-main-ref`, created from the merged `main`.
- `origin` is `https://github.com/aryonmt/PathLens-GNN.git`.
- `legacy/` contains the preserved original repository and nested Git history and must not be modified or committed.
- Canonical data, model, artifact export, API, React 3D UI, Docker, CI, and Kaggle workflow are implemented.
- Registered Kaggle training and the sealed final test have not yet run; no research result is claimed.
- `docs/delivery/ISSUE_BACKLOG.yaml` preserves the exact board payload for future GitHub Issues.

## Immediate implementation order

Run Kaggle smoke/tuning -> record freeze -> run sealed evaluation -> publish artifact -> deploy public demo.

## Non-negotiable scientific language

- Unknown non-edges are not verified negatives.
- Two-hop DTI evidence is same-type projection context, not a drug-to-protein path.
- Three-hop paths are structural support, not causal proof.
- Known edges must never appear as novel recommendations.
