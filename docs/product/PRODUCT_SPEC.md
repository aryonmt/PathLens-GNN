# Product Specification

## Product question

For a known BioSNAP drug or protein, which currently unrecorded typed interactions should a researcher inspect first, and what graph/model evidence influenced that ranking?

## Workflows

1. Search a drug or protein by ID or bundled display name.
2. View up to 100 precomputed recommendations and filter by score/evidence.
3. Select a pair and inspect PathLens score, known status, channel contributions, two-hop projection context, and three-hop bridge paths.
4. Explore a focused explanation graph or capped one-to-three-hop ego graph in 3D.
5. Export the visible recommendation set as CSV or JSON.

## Pages

- Discovery: search, summary, recommendation table, filters, export.
- Pair Explorer: evidence cards, contribution chart, path list, focused 3D graph.
- Model & Evidence: benchmark, sources, model version, limitations, non-clinical disclaimer.

## Visual language

Desktop-first dark scientific interface; color-blind-safe cyan/amber palette; drugs and proteins use different geometry as well as color. Known edges are neutral, candidate edges bright, and selected supporting paths gold. Labels appear on hover/selection, and camera movement follows explicit user selections.

## Product language

Use “PathLens score”, “research priority”, “unknown candidate”, “projection context”, and “structural support”. Never use “validated interaction”, “clinical confidence”, or “causal mechanism”.
