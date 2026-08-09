# Literature and Data Sources

## Core sources

- Huang et al., *SkipGNN: predicting molecular interactions with skip-graph networks*, Scientific Reports, 2020. Upstream code: https://github.com/kexinhuang12345/SkipGNN
- Stanford SNAP, BioSNAP drug-target interaction network: https://snap.stanford.edu/biodata/datasets/10002/10002-ChG-Miner.html
- Li et al., *Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking*, NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/file/0be50b4590f1c5fdf4c8feddd63c4f67-Paper-Datasets_and_Benchmarks.pdf
- Hu et al., *Enhancing link prediction in biomedical knowledge graphs with BioPathNet*, Nature Biomedical Engineering, 2026: https://www.nature.com/articles/s41551-025-01598-z
- Yang et al., *Pure Message Passing Can Estimate Common Neighbor for Link Prediction*, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/85970f7bbc821852c1d17052b88c2451-Abstract-Conference.html

## Metadata

Protein labels may be enriched through the public UniProt REST services. Drug labels use a redistribution-compatible public mapping such as Wikidata; missing names fall back to stable IDs. Cached metadata records source, retrieval time, and source identifier.

## Positioning

PathLens-GNN does not claim publication-level novelty before experiments. Its defensible contribution is the combined leakage-safe benchmark, sparse hop-separated architecture, pre-registered ablations, and faithful product evidence contract.
