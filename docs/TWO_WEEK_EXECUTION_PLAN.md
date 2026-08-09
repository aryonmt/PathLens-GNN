# Two-Week Execution Plan

## Guiding Rule

The project has two equally important outputs: a credible model contribution and a polished inference product. The model must be frozen early enough to leave real time for the 3D application and deployment.

## Day-by-Day Plan

| Day | Required output |
|---|---|
| 1 | New package architecture, pinned environment, schema tests, reproducibility setup |
| 2 | Deterministic type-aware splits, valid negative sampler, hard-negative interface |
| 3 | Faithful original SkipGNN baseline on the corrected pipeline |
| 4 | Weighted two-hop channel and efficient sparse path representation |
| 5 | Three-hop bipartite evidence and pair-conditioned adaptive fusion |
| 6 | Relation-appropriate decoder, path extraction, first DTI training run |
| 7 | Ablations, multiple seeds, checkpoint selection, model freeze |
| 8 | Final Kaggle runs if needed; generate embeddings, coordinates, paths, and deployment artifacts |
| 9 | FastAPI predictor, ranking, explanation, graph-neighborhood endpoints |
| 10 | React/TypeScript application shell, design system, search and discovery table |
| 11 | Interactive 3D topology explorer with focus, filters, and path highlighting |
| 12 | 3D embedding view, contribution visualization, export flow, end-to-end integration |
| 13 | Docker Compose, Vercel/Render deployment, automated tests, error handling |
| 14 | README, architecture diagram, demo video/GIF, final result tables, presentation rehearsal, buffer |

## Model Freeze Gate

At the end of Day 7, select a deployable model even if all stretch experiments are incomplete. After the freeze:

- only critical model bugs should change the artifact contract;
- additional experiments may run independently;
- backend and frontend development proceed against stable artifact schemas.

## Minimum Viable Research Result

- corrected typed benchmark;
- reliable original SkipGNN baseline;
- one weighted multi-hop improvement;
- one adaptive pair-fusion improvement;
- DTI results over several seeds;
- at least the critical ablations;
- model explanation artifacts.

## Minimum Viable Product

- search one known drug or protein;
- receive Top-K candidates from a trained model;
- inspect a candidate pair;
- see probability and per-hop evidence;
- view an interactive 3D supporting subgraph;
- export predictions;
- run locally through Docker;
- access one public deployment.

## Cut Order If Time Is Lost

Remove scope in this order:

1. all four datasets;
2. a second confirmation dataset;
3. full deep-ensemble uncertainty;
4. global full-network visualization;
5. advanced visual postprocessing;
6. additional deployment providers.

Do not cut:

- typed data correctness;
- reliable evaluation;
- the main model contribution;
- Top-K inference;
- focused 3D path visualization;
- local reproducible deployment.

## Compute Plan

### Local RTX 3050 Laptop GPU, 4 GB

Use for:

- development;
- unit tests;
- data pipeline runs;
- DTI/PPI smoke training;
- inference and local deployment.

### Kaggle P100

Use for:

- multi-seed final runs;
- ablation matrix;
- larger or memory-sensitive experiments;
- producing final checkpoints.

### Public server

Use CPU inference over precomputed embeddings and a lightweight decoder. Do not deploy the training pipeline or require a GPU for the public application.

