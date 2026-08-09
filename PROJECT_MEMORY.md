# SkipGNN Project Memory and Session Handoff

Last updated: 2026-08-09 (Asia/Tehran)

This file is the canonical handoff for continuing the project in a new Codex session. Read this file first, then read the detailed documents in `docs/`.

## 1. Project Goal

Rewrite and substantially extend the 2020 SkipGNN research repository into two connected deliverables:

1. A scientifically defensible, technically modern graph-learning project with a genuine model contribution beyond bug fixes.
2. A deployed inference product for discovering, ranking, and explaining likely biomedical interactions through a polished interactive 3D graph interface.

The deadline is approximately two weeks. The scope must remain focused enough to finish.

## 2. Approved Direction

### Research/model direction

Working model name:

**Path-Aware Adaptive SkipGNN**

Working research title:

**Path-Aware SkipGNN: Adaptive Multi-Hop Fusion for Explainable Molecular Interaction Prediction**

The original SkipGNN idea is retained, but the new model will use:

- a direct graph channel based on `A`;
- a weighted two-hop channel based on `A^2`, instead of only `sign(A^2)`;
- an explicit weighted three-hop channel based on `A^3`, especially for bipartite drug-target and gene-disease prediction;
- pair-conditioned adaptive fusion, so the importance of one-, two-, and three-hop evidence can vary for each candidate pair;
- relation-appropriate decoders: symmetric for homogeneous networks and typed/directional for bipartite networks;
- path-based explanations that expose supporting intermediate nodes and per-hop contributions.

Correct data processing and evaluation are mandatory foundations, but they are not the main claimed contribution.

### Product direction

Working product name:

**PathLens-GNN** or **SkipGNN Studio**

The application is strictly an inference/discovery application. Training, hyperparameter tuning, and notebook execution must not appear in the user interface.

Primary user workflow:

1. Search for a known drug or protein.
2. Rank likely unrecorded interaction candidates.
3. Select a candidate pair.
4. View probability, uncertainty, direct/two-hop/three-hop contributions, and supporting paths.
5. Explore the result in a polished interactive 3D graph.
6. Export candidate rankings and evidence as CSV or JSON.

The application is a research prioritization tool, not a medical or clinical decision system.

## 3. Legacy Repository Status

All original source code, data, notebooks, build products, original README/setup files, original license, original gitignore, and the complete original Git history have been moved to `legacy/` without intentionally changing their contents.

The repository root is now a fresh Git repository. Its intended branch structure is:

- `main`: clean base branch for the rewritten project;
- `planning`: active branch for the current planning and handoff documents.

The following infrastructure is present at the root:

- a new `.git/` history unrelated to the archived upstream history;
- a new `.gitignore` that ignores the entire `legacy/` directory;
- `PROJECT_MEMORY.md`
- `docs/`
- `legacy/`

The archived original repository metadata is available at `legacy/.git/`. The original BSD-3 license is available at `legacy/LICENSE`. A license for the rewritten root project should be selected explicitly before public release; retaining BSD-3 compatibility and attribution is the safest default if legacy code is reused.

There were pre-existing local working-tree changes before reorganization. They belong to the user and must be preserved. Most dataset and notebook modifications were file-mode or line-ending-only changes. `legacy/SkipGNN/train.py` contains the user's plotting adjustment that converts stored loss tensors to CPU scalars before plotting. Do not discard these changes.

## 4. What Legacy SkipGNN Did

The legacy system:

1. Constructed an adjacency matrix `A` from positive training interactions.
2. Constructed a skip graph as `sign(A @ A)`.
3. Initialized nodes with one-hot identity features or externally generated node2vec embeddings.
4. Ran graph convolution paths over the original graph and the skip graph.
5. Fused the paths using fixed summation in an iterative two-layer architecture.
6. Concatenated the final embeddings of a candidate node pair.
7. Used an effectively linear two-layer decoder to predict a binary interaction score.

Supported datasets:

- DDI: drug-drug interactions
- DTI: drug-target interactions
- PPI: protein-protein interactions
- GDI: gene-disease interactions

Read `docs/LEGACY_AUDIT_AND_MODEL_PLAN.md` for the detailed audit.

## 5. Critical Audit Findings

### Invalid DTI negatives

The DTI preprocessing notebook sampled negatives from combinations of a single mixed drug/protein entity list. As a result, approximately 79% of DTI negatives are schema-invalid.

Counts in the provided fold:

| Split | All negatives | Valid drug-to-protein negatives | Schema-invalid negatives |
|---|---:|---:|---:|
| Train | 10,671 | 2,218 | 8,453 |
| Validation | 1,523 | 344 | 1,179 |
| Test | 2,945 | 617 | 2,328 |

A trivial type/order-only classifier reaches approximately:

- test AUROC: 0.8952
- test Average Precision: 0.8345

The stored legacy notebook result for DTI reports approximately AUROC 0.9515 and AUPRC 0.9516, so a substantial portion of the apparent performance may come from a schema shortcut.

### Invalid GDI negatives

The GDI preprocessing notebook uses a stale loop variable in one negative-sampling loop (`i` instead of `j`). The provided train split contains 2,685 gene-gene negatives, which are invalid for a gene-disease relation.

### Evaluation defects

- Validation and test use `shuffle=True` and `drop_last=True`.
- DTI test contains 6,056 rows, but only 5,888 are evaluated with batch size 256.
- F1 thresholds raw logits at 0.5 instead of thresholding logits at 0 or probabilities at 0.5.
- Reported validation/test loss is only the final batch loss.
- Evaluation does not use `torch.no_grad()` or `torch.inference_mode()`.
- `fastmode` can cause the initial untrained model copy to be tested.
- Model selection behavior differs from the paper description.

### Engineering defects

- Explicit one-hot matrices are extremely memory inefficient. GDI creates a 19,783 by 19,783 identity matrix.
- `loss_train` tensors are retained in a history list with their computation graphs. Converting them only at plotting time does not prevent training-time graph retention.
- Node2vec paths omit a path separator.
- Dependency versions are not pinned; required training dependencies are absent from `setup.py`.
- Package imports are not consistently relative.
- The script lacks a Windows-safe `if __name__ == "__main__"` guard while using multiple DataLoader workers.
- The decoder has two linear layers without a non-linearity and is therefore effectively linear.
- Concatenation is order-sensitive, which is unsuitable for undirected DDI and PPI unless explicitly symmetrized.

## 6. Approved Technical Stack

### Model and backend

- Python
- PyTorch
- PyTorch Geometric where it materially simplifies sparse graph operations and heterogeneous/bipartite graph handling
- FastAPI for the inference API
- Pydantic schemas
- Pytest
- YAML configuration
- ONNX or a lightweight NumPy runtime for deployment if time permits

### Frontend and 3D visualization

- React
- TypeScript
- Vite
- `react-force-graph-3d`
- Three.js/WebGL

Streamlit is not the approved primary frontend because polished 3D visualization is a central product requirement.

## 7. Training and Deployment Decisions

Detected local GPU:

- NVIDIA GeForce RTX 3050 Laptop GPU
- 4 GB VRAM

Current default Python is 3.14.6 and PyTorch is not currently installed in that interpreter.

Approved compute workflow:

- Local laptop: development, tests, small runs, DTI/PPI experiments, and offline application demo.
- Kaggle GPU: final multi-seed training, ablations, and larger experiments. Current Kaggle documentation advertises Tesla P100 access with a weekly quota around 30 hours or more depending on availability.
- Production inference: CPU only. Precompute node embeddings, 3D coordinates, graph metadata, and optionally Top-K candidate lists. The deployed service should execute only a lightweight decoder and graph queries.

Deployment options, in priority order:

1. Always maintain a fully local Docker Compose deployment for the university presentation.
2. Portfolio deployment: React/Vite frontend on Vercel and FastAPI backend on Render.
3. Stable always-on deployment: a small 2-vCPU/4-GB-RAM VPS running Docker Compose and a reverse proxy.

Render's free service may sleep after inactivity and have a cold start. The public portfolio version may use that limitation initially.

## 8. Core Scope Versus Stretch Scope

### Mandatory core

- clean repository architecture;
- reproducible environment;
- type-aware train/validation/test generation;
- hard-negative support;
- reliable evaluation;
- original SkipGNN baseline reimplementation;
- weighted `A^2` channel;
- explicit `A^3` channel for bipartite interaction evidence;
- pair-conditioned adaptive fusion;
- relation-appropriate decoder;
- path extraction for explanations;
- one primary dataset (DTI) and one confirmation dataset (PPI if time allows);
- trained checkpoint and precomputed inference artifacts;
- FastAPI inference/ranking/explanation API;
- React 3D discovery interface;
- local Docker deployment and one public deployment.

### Stretch scope

- all four original datasets;
- calibrated deep ensembles;
- arbitrary unseen drug/protein inputs using molecular/sequence features;
- authentication or a database;
- user uploads;
- in-app model training;
- clinical claims;
- exhaustive hyperparameter sweeps.

## 9. Immediate Next Actions for the New Session

Do not begin by writing the frontend. Start in this order:

1. Inspect this memory and all files in `docs/`.
2. Inspect `legacy/` and verify the reorganization preserved all legacy files and user modifications.
3. Create the new package skeleton under `src/pathlens_gnn/` or `src/skipgnn/`.
4. Pin a Python/PyTorch environment compatible with the laptop and Kaggle.
5. Write data-schema tests before implementing the new sampler.
6. Implement deterministic type-aware splits and negative sampling.
7. Reimplement the original SkipGNN baseline against the corrected pipeline.
8. Implement the path-aware model incrementally, with one ablation per channel.

## 10. Important Constraints

- Two-week deadline.
- Do not destroy or overwrite legacy data or user changes.
- Do not claim biological validation from model predictions.
- Do not claim publication-level novelty until a focused literature review verifies it.
- A lower metric on corrected hard negatives can be scientifically better than a higher legacy metric on invalid easy negatives.
- Freeze the AI model and artifacts before serious frontend work begins.
