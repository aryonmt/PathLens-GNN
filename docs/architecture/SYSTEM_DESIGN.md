# System Design

## Components

```text
BioSNAP source -> canonicalizer -> split/negative manifests
       -> PyTorch training/evaluation -> frozen checkpoint
       -> artifact exporter -> NumPy arrays + read-only SQLite
       -> FastAPI -> React/Vite -> focused and ego 3D views
```

Training and public inference are separate. The browser never receives a full skip graph, performs path-matrix multiplication, or trains a model.

## Repository layout

```text
src/pathlens_gnn/   Python research, export, runtime, and API package
tests/              unit, integrity, parity, API, and integration tests
configs/            immutable experiment and deployment configurations
scripts/            data preparation, training, export, and smoke commands
apps/web/           React/TypeScript application
artifacts/fixtures/ tiny committed test artifact only
```

## Runtime

The deployed process validates artifact hashes during FastAPI lifespan, memory-maps arrays, opens SQLite read-only, and uses a single worker. NumPy reproduces the trained typed experts and gate; PyTorch is not installed in the production image.

## Deployment

- Docker Compose is the authoritative university demo.
- Vercel serves the static web bundle.
- Render serves one stateless API worker. The build fetches a versioned GitHub Release artifact and verifies its checksum.
- The frontend presents a backend-wakeup state for free-tier cold starts.

## Performance boundaries

Recommendations are precomputed Top-100 per eligible entity. Pair graphs cap at 150 nodes/300 links; ego graphs cap at 250/600. Force simulation freezes after stabilization. Warm cached reads target 250ms and bounded pair explanations target two seconds.
