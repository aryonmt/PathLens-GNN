# Data analysis — BioSNAP DTI (ChG-Miner)

Campaign `biosnap-dti-v2`. Laptop does not keep the raw TSV. Kaggle downloads
it; EDA PNGs are written by `src/pathlens/evaluation/eda.py` in the delivery
notebook.

## Identity

| Field | Value |
|---|---|
| Source | Stanford BioSNAP ChG-Miner |
| URL | https://snap.stanford.edu/biodata/datasets/10002/files/ChG-Miner_miner-chem-gene.tsv.gz |
| SHA-256 | `b54a548bb0b6d7039b5c317bf8251d17233bdc89bc49f477aeb8abddfafc9e6c` |
| Header | `#Drug` / `Gene` — **not an edge** |
| Edges | 15,138 |
| Drugs | 5,017 |
| Proteins | 2,324 |
| Entities | 7,341 |
| Split seed | 41 |

A clone that treats the header as a pair reports 15,139 edges. Tests live in
`tests/test_data.py`.

## Graph type

The graph is **bipartite**. There are no drug–drug or protein–protein edges in
the canonical file. Consequences:

- Even-length walks stay on the same part (SkipGNN's \(A^2\) skip graph).
- Odd-length walks are the only drug–protein paths (1-hop and 3-hop).
- "Two-hop neighbor" in a DTI figure is a **same-type** node, not a target.

## Split

Coverage-preserving ~60 / 20 / 10 / 10 = context / train / val / test.
An edge may enter holdout only if both endpoints keep at least one context
edge. Every validation and test query therefore has context degree ≥ 1.
This is transductive missing-link ranking, not inductive cold-start of new
entities.

Exact processed counts are in `data/processed/biosnap-dti-v2/manifest.json`
after Kaggle prepare. Validation ranking uses **1,514** positive queries
(filed cards).

Negatives: unknown non-edges. Uniform 1:1 and degree-matched hard 1:1. Known
positives are never sampled as negatives.

## What the degree distribution implies

Context degree is heavy-tailed on both sides (typical of BioSNAP DTI).
Ranking diagnostics split validation drugs into tertiles:

| Tertile | Queries | RA MRR | 3-hop MRR | PathLens ranking MRR |
|---|---|---|---|---|
| Low | 560 | **0.496** | 0.478 | 0.250 |
| Mid | 552 | 0.465 | **0.481** | 0.437 |
| High | 402 | 0.410 | 0.386 | **0.466** |

The sparse tail is where **heuristics win**. PathLens wins hubs. That is the
opposite of a "GNN saves cold-start" narrative.

Kaggle EDA writes `degree_hist_drugs.png` and `degree_hist_proteins.png`.

## Zero-mass 3-hop

28.6% of validation positives have RA / 3-hop score **exactly 0**: no
normalized 3-walk witness in **context**. Those targets share a gigantic tie
group at score 0. Optimistic rank (`strict_gt`) does not punish that pile;
average-tie MRR does.

This is a **data fact**, not a bug in GEMM. It is why tie diagnostics exist
and why we did not silently switch the freeze metric.

Kaggle writes `three_hop_zero_mass.png` from the same context matrix the
heuristics use.

## What this dataset is not

- Not DDI, PPI, or GDI. Those graphs are not bipartite in the same way.
- Not a probability of binding. A sigmoid is a ranking score.
- Not Huang et al.'s unpublished internal split. Do not compare 0.928 to
  our AUPRC.
