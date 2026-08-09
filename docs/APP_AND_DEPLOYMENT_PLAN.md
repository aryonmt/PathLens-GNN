# Inference Product and Deployment Plan

## 1. Product Definition

The product is an inference and interaction-discovery application. It must not train models in the UI.

Primary product question:

> Given known biomedical interaction structure, which currently unrecorded interactions should a researcher investigate first, and what graph evidence supports each suggestion?

Initial use case: drug-target prioritization on the DTI network.

## 2. User Workflows

### Drug-to-target discovery

1. Search/select a known drug.
2. Score candidate proteins not recorded as known interactions in the selected benchmark graph.
3. Display Top-K candidates with probability, uncertainty, evidence strength, and novelty status.
4. Open a candidate explanation.
5. Inspect supporting graph paths in 3D.
6. Export selected candidates or the full ranking.

### Protein-to-drug discovery

The inverse ranking workflow lists candidate drugs for a selected protein.

### Pair inspection

Users select one source and one target and receive:

- predicted probability;
- known-versus-unrecorded status;
- uncertainty;
- direct/two-hop/three-hop contribution weights;
- top supporting paths;
- relevant neighborhood statistics.

### Global discovery

Display the highest-ranked unrecorded interactions with filters for probability, uncertainty, node type, degree, and path support.

## 3. 3D Visualization Requirements

Visualization is a central product feature.

Approved stack:

- React + TypeScript + Vite
- `react-force-graph-3d`
- Three.js/WebGL

### Topological graph view

- distinct shapes/colors for drugs, proteins, genes, and diseases;
- known direct edges in a neutral style;
- skip edges in cyan or another distinct style;
- predicted edges as bright animated links;
- top supporting paths highlighted in gold;
- animated directional particles where direction is meaningful;
- node radius based on degree or importance;
- edge opacity/width based on evidence weight;
- camera animation when a node or prediction is selected;
- hover cards and click selection;
- toggles for one-, two-, and three-hop evidence.

### Latent embedding view

- precompute 3D UMAP coordinates from trained node embeddings;
- color by node type or cluster;
- switch between legacy and new-model embeddings;
- locate and connect a selected candidate pair;
- support focus, zoom, labels, and neighborhood filtering.

### Performance strategy

Do not render every skip edge in the full graph.

Provide three levels of detail:

1. Global sampled or clustered view.
2. Local one-to-three-hop ego neighborhood.
3. Explanation-only view with the most influential paths.

The browser performs WebGL rendering. The backend returns compact graph JSON for the current view.

## 4. Inference Architecture

Training and public inference are separated.

### Offline artifact generation

After training, produce:

- node embeddings;
- node ID maps and metadata;
- decoder weights or exported ONNX model;
- 3D embedding coordinates;
- graph adjacency/index data;
- optional precomputed Top-K candidates;
- model card and training configuration;
- evaluation summary;
- path/explanation indices where practical.

### Online request path

```text
React client -> FastAPI -> embedding lookup -> lightweight decoder -> explanation query -> JSON response
```

The full GNN encoder should not run for every public request when the graph and trained model are fixed.

### Proposed API

- `GET /health`
- `GET /model`
- `GET /datasets`
- `GET /entities`
- `POST /predict`
- `GET /recommendations/{entity_id}`
- `POST /explain`
- `GET /graph/neighborhood/{entity_id}`
- `GET /graph/pair`

## 5. Application Pages

### Discovery

- drug/protein search;
- Top-K ranked candidates;
- confidence and uncertainty filters;
- downloadable table.

### Pair Explorer

- pair score;
- known/unrecorded status;
- contribution chart;
- supporting path list;
- focused 3D graph.

### Graph Universe

- global or clustered 3D network;
- entity filters;
- search and animated camera focus;
- switch between topology and embedding layouts.

### Model and Evidence

- dataset sources and dates;
- benchmark results;
- baseline comparison;
- model limitations;
- clear statement that predictions are research hypotheses, not clinical evidence.

## 6. Deployment

### Local/offline deployment

Maintain a Docker Compose configuration that starts frontend and backend locally. This is the reliable university-presentation deployment and must not require internet after images/artifacts are present.

### Public portfolio deployment

Preferred initial split deployment:

- React/Vite frontend: Vercel Hobby for a personal non-commercial portfolio.
- FastAPI backend: Render web service.
- Model artifacts: packaged with the backend if small, otherwise stored in a release or object store with pinned hashes.

The Render free tier may sleep after inactivity and cold-start on the next request. The UI should show a friendly backend-wakeup state.

### Stable deployment

If an always-on public demo is required, use a small Linux VPS with approximately:

- 2 vCPU;
- 4 GB RAM;
- 20 GB disk;
- Docker Compose;
- Nginx or another reverse proxy;
- HTTPS.

No inference GPU should be required.

## 7. Non-Goals for the Two-Week Version

- no in-app training;
- no clinical diagnosis, prescribing, or safety claims;
- no arbitrary unseen molecule/protein inputs without a validated cold-start feature pipeline;
- no authentication unless unexpectedly easy;
- no user-managed datasets;
- no large persistent database unless a concrete need appears.

