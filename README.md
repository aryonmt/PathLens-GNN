# PathLens-GNN

PathLens-GNN is a leakage-aware, path-aware graph learning system for transductive drug-target interaction (DTI) prioritization. It combines a reproducible research pipeline with an inference-only web product that explains ranked candidates through focused 3D graph views.

> PathLens scores are research-prioritization signals. They are not validated biological interactions, medical advice, or clinical probabilities.

## Status

The implementation branch contains the canonical-data pipeline, sparse three-channel model,
registered Kaggle tuning workflow, NumPy artifact runtime, versioned FastAPI service, and React
3D Pair/Ego explorers. Research results remain explicitly pending until the registered Kaggle run
is complete and the sealed test command is authorized after model freeze.

The archived SkipGNN repository remains locally under `legacy/` and is intentionally excluded
from Git.

## Quick start (fixture product)

The checked-in fixture is synthetic and exists only for product and contract testing.

```bash
uv sync --extra dev --frozen
uv run uvicorn pathlens_gnn.api.app:app --reload
```

In a second terminal:

```bash
corepack enable
pnpm --dir apps/web install --frozen-lockfile
pnpm --dir apps/web dev
```

Open `http://localhost:5173`. For the portable presentation build, run
`docker compose up --build` and open `http://localhost:8080`.

## Kaggle-first training

Training is intentionally separated from local product development. Push this repository, import
[`kaggle/pathlens_training.ipynb`](kaggle/pathlens_training.ipynb) into a GPU Kaggle notebook,
and follow [`kaggle/README.md`](kaggle/README.md). The notebook clones the public repository,
downloads the canonical BioSNAP source, prepares immutable splits, runs the registered budget, and
exports a versioned inference artifact.

## Documentation

- [Project charter](docs/decisions/PROJECT_CHARTER.md)
- [Legacy audit](docs/research/LEGACY_AUDIT.md)
- [Research specification](docs/research/RESEARCH_SPEC.md)
- [Evaluation protocol](docs/research/EVALUATION_PROTOCOL.md)
- [Product specification](docs/product/PRODUCT_SPEC.md)
- [System design](docs/architecture/SYSTEM_DESIGN.md)
- [API and artifact contract](docs/contracts/API_AND_ARTIFACT_CONTRACT.md)
- [Kanban playbook](docs/delivery/KANBAN_PLAYBOOK.md)
- [Risk register](docs/delivery/RISK_REGISTER.md)

## Stack

- Python 3.11/3.12, PyTorch, sparse propagation, scikit-learn
- FastAPI and a lightweight NumPy inference runtime
- React, TypeScript, Vite, `react-force-graph-3d`
- Docker Compose, Vercel, and Render

## License and provenance

Code is licensed under BSD-3-Clause. See [NOTICE](NOTICE) for upstream SkipGNN and dataset attribution. Dataset terms remain those of their respective providers.
