# Critique of Huang et al. SkipGNN (2020)

English source for report section 1. Claims below are either in Huang et al.
*SkipGNN: predicting molecular interactions with skip-graph networks*,
Scientific Reports 2020, or are measurements from this repository on canonical
BioSNAP DTI (campaign `biosnap-dti-v2`, split seed 41). Do not treat the
paper's DTI PR-AUC **0.928** as a baseline this project lost to.

## What the paper argues

SkipGNN builds two graphs from the observed interaction network:

1. The **original graph**: symmetrized adjacency plus self-loops, then
   \(D^{-1/2} A D^{-1/2}\).
2. The **skip graph**: \(\mathrm{sign}(A^2)\), then the same normalization.

A two-channel GCN mixes the two, and a linear decoder scores pairs. The stated
motivation is that 2-hop neighbors carry similarity that 1-hop GCN message
passing underuses. The paper reports PR-AUC / ROC-AUC on DTI, DDI, PPI, and
GDI, with figures for missing-edge robustness, t-SNE, and hairball layouts.

This repository reimplements the encoder equations from
`legacy1/SkipGNN/utils.py` in [`src/pathlens/graph/skip.py`](../../src/pathlens/graph/skip.py):

```84:92:src/pathlens/graph/skip.py
def build_skipgnn_adjacencies(
    num_drugs: int,
    num_proteins: int,
    bipartite_edges: NDArray[np.int64],
) -> tuple[SparseAdj, SparseAdj]:
    adj = build_symmetric_adjacency(num_drugs, num_proteins, bipartite_edges)
    skip = _normalize_adj(adj.dot(adj).sign())
    original = _normalize_adj(adj + sp.eye(adj.shape[0], dtype=np.float32))
    return original.tocoo(), skip.tocoo()
```

The encoder is therefore a reproduction. The **evaluation protocol is not**.

## The paper's metric is a different task

Huang et al. score **1:1 sampled classification** (PR-AUC / ROC-AUC on a
balanced pair bank). That answers: *given one true edge and one drawn
non-edge, can the model tell them apart?*

Drug–target ranking answers: *for this drug, where does the true protein sit
among ~2,300 candidates after other known positives are removed?* That is
filtered per-drug MRR / Hits@K in
[`src/pathlens/evaluation/ranking.py`](../../src/pathlens/evaluation/ranking.py).

The two numbers can move in opposite directions. On this split:

| Method | 1:1 selection metric | Validation hard AUPRC | Validation filtered MRR |
|---|---|---|---|
| SkipGNN (this repo) | val AUROC 0.932 at epoch 6 | 0.824 | **0.144** |
| GCN (same loop, no skip channel) | val AUROC 0.929 at epoch 8 | 0.822 | 0.136 |
| Resource allocation | none | 0.845 | **0.462** |
| Normalized 3-hop | none | 0.847 | 0.455 |

SkipGNN's sampled AUROC is high and almost identical to GCN. Filtered MRR stays
near a degree baseline (0.134). A 1:1 AUC leaderboard does not certify a
ranker.

Li et al., *Evaluating Graph Neural Networks for Link Prediction* (NeurIPS
2023), make the same methodological point: random negative sampling inflates
link-prediction scores and hides ranking failures.

## On a bipartite DTI graph, \(A^2\) is not a drug–protein path

BioSNAP ChG-Miner is **bipartite**: every edge is drug–protein. Walks of even
length land on the **same type**. Walks of odd length land on the opposite
type.

So \(\mathrm{sign}(A^2)\) connects drug–drug and protein–protein pairs that
share a neighbor. It is a same-type projection, not a missing DTI edge. The
first return to a drug–protein pair is length **3**.

That is why this campaign's strongest parameter-free rankers are
resource-allocation and normalized 3-hop
([`src/pathlens/graph/scoring.py`](../../src/pathlens/graph/scoring.py)), not
1-hop GCN. Huang et al. did include path heuristics (common neighbors, Jaccard,
and L3-style counts) among baselines and **won on sampled PR-AUC**, not on
filtered ranking. Reproducing the encoder here recovers a **+0.008 MRR** skip
channel over GCN — real, small, and nowhere near the heuristic gap.

Mixing DDI / PPI (homogeneous) with DTI (bipartite) in one PR-AUC plot hides
this odd/even structure. Phase 1 of this project is DTI only.

## Sampling and leakage are not the same protocol

Paper-style 1:1 banks draw negatives from unknown non-edges. This repo does
that too, for classification curves, **and** reports filtered ranking against
the full protein list. Known positives never appear as negatives
([`src/pathlens/data/negative.py`](../../src/pathlens/data/negative.py)).

Two extra repairs relative to naive clones:

1. **Header is not an edge.** `#Drug` / `Gene` in ChG-Miner is a TSV header.
   Counting it as an interaction yields 15,139 edges. Canonical counts after
   the fix: **15,138 edges, 5,017 drugs, 2,324 proteins**. See
   [`docs/research/DATA.md`](DATA.md) and
   [`src/pathlens/data/canonical.py`](../../src/pathlens/data/canonical.py).
2. **Coverage-preserving transductive split.** Every scored node has context
   degree ≥ 1. Heuristics see **context only**. GNN trainers see
   `context ∪ train_positive`. Validation and test edges are never in the
   message-passing graph.

The paper's missing-edge robustness curves (PR-AUC vs % removed edges) ask a
different question from "rank the held-out protein for this drug." t-SNE of
embeddings and force-directed hairballs are not ranking evidence. Those figure
types are not the argument of this campaign.

## What a fair SkipGNN comparison looks like here

- Same canonical graph, same seed-41 split, same decoder family, same typed
  1:1 BCE loop as Huang's `train.py` (Adam, keep best validation AUROC).
- Official card: `skipgnn` eval T4 `56970fe`, not the retired v2 overlay
  `binary_skipgnn`.
- Report **both** hard AUPRC and filtered MRR. SkipGNN is competitive on the
  paper's style of metric and weak on ranking.
- Do not declare a win or loss against 0.928. Different split, different
  negatives, different header handling, different task.

The scientific conclusion of this project is not "SkipGNN is wrong." It is:
**on bipartite DTI ranking, the skip graph is a weak same-type channel, and
1:1 AUC is the wrong headline.**
