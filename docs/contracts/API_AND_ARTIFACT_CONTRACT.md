# API and Artifact Contract

## Artifact v1

```text
artifacts/<version>/
  manifest.json
  model_weights.npz
  embeddings/*.npy
  graph/*.npy
  pathlens.db
  evaluation_summary.json
  model_card.md
```

`manifest.json` contains schema/model/dataset versions, creation time, source/split/config/checkpoint hashes, tensor dimensions, and SHA-256 for every payload. Arrays are uncompressed and memory-mappable. `pathlens.db` is immutable and contains entities, known edges, Top-100 recommendations, cached Top-25 explanations, and optional fixed coordinates.

PyTorch-to-NumPy logits and contributions must agree within absolute tolerance `1e-5` on a fixed parity fixture.

## HTTP API

All meaningful responses contain `model_version` and `dataset_version`.

| Method | Route | Limits |
|---|---|---|
| GET | `/api/v1/health` | readiness and version only |
| GET | `/api/v1/model` | model/data card summary |
| GET | `/api/v1/entities` | default 25, max 100 |
| GET | `/api/v1/entities/{id}` | one typed entity |
| GET | `/api/v1/recommendations/{id}` | default 25, max 100 |
| POST | `/api/v1/predict` | one canonical drug-protein pair |
| GET | `/api/v1/explanations/{source}/{target}` | max five paths |
| GET | `/api/v1/graph/ego/{id}` | 1-3 hops; max 250/600 |
| GET | `/api/v1/graph/pair` | max 150/300 |
| GET | `/api/v1/exports/recommendations/{id}.csv` | streamed current ranking |

Errors: unknown ID 404; wrong typed relation 422; unavailable/invalid artifact 503. Pair orientation is canonicalized, but the response preserves the requested source for UX. Explanation responses expose `truncated` and distinguish projection context, bridge paths, and model-channel contributions.
