# Living report — BioSNAP DTI (`biosnap-dti-v2`)

English only. No test numbers until freeze.

## Task

Transductive missing-link ranking on a bipartite drug–protein graph. Negatives
are unknown non-edges.

## Methods

See [`docs/STATUS.md`](../STATUS.md).

The five PathLens cards are **one architecture, ablated**, not five rival
systems. Keep the ladder even though `pathlens_ranking` is the family champion:

| Card | Isolated change |
|---|---|
| `one_hop` | S1 only (opposite-type 1-hop) |
| `s1_s2` | add same-type projection S2 (not a drug–protein path) |
| `s1_s2_s3_fixed` | add S3 with a fixed mix |
| `pathlens_bce` | full model, adaptive gate, BCE 1:1 |
| `pathlens_ranking` | same as `pathlens_bce`, sampled-softmax ranking loss |

Reporting only the best PathLens number would hide two facts this campaign is
about: ranking loss, not extra hops, is what moved PathLens (BCE variants stay
near MRR 0.10–0.16); and the family champion still loses filtered MRR to the
parameter-free `resource_allocation` heuristic. Freeze currently names
`three_hop` and `skipgnn`; on this split `resource_allocation` is the stronger
heuristic.

## Validation figures

Filed under [`runs/biosnap-dti-v2/figures/`](../../runs/biosnap-dti-v2/figures/).
Produced on Kaggle, not on the laptop.

| File | What it shows |
|---|---|
| `pr_hard.png` | Precision–recall on degree-matched unknowns |
| `pr_uniform.png` | Precision–recall on uniform typed unknowns |
| `roc_hard.png` | ROC on degree-matched unknowns |
| `hits_at_k.png` | Filtered Hits@K |
| `mrr_and_hard_auprc.png` | Filtered MRR vs hard AUPRC |
| `degree_slices_hard.png` | Hard AUPRC by endpoint-degree tertile |

## Validation tables

### Heuristics (this repo)

| Method | hard AUPRC | MRR | Hits@10 | NDCG@10 | NDCG@50 | Source |
|---|---|---|---|---|---|---|
| `resource_allocation` | 0.845 | 0.462 | 0.672 | 0.504 | 0.547 | eval T4 `d2b90ca` |
| `three_hop` | 0.847 | 0.455 | 0.673 | 0.499 | 0.544 | eval T4 `46fc64f` |
| `degree` | 0.774 | 0.134 | 0.211 | 0.144 | 0.182 | eval T4 `5daa27c` |

### PathLens family (imported campaign v2)

Validation only. Test sealed. NDCG is computed from the stored filtered ranks
(no rescoring, no test access).

| Method | hard AUPRC | MRR | Hits@10 | NDCG@10 | NDCG@50 | Source |
|---|---|---|---|---|---|---|
| `one_hop` | 0.788 | 0.103 | 0.192 | 0.114 | 0.161 | imported registered |
| `s1_s2` | 0.825 | 0.143 | 0.227 | 0.153 | 0.205 | imported registered |
| `s1_s2_s3_fixed` | 0.826 | 0.136 | 0.224 | 0.147 | 0.199 | imported registered |
| `pathlens_bce` | 0.852 | 0.160 | 0.317 | 0.187 | 0.235 | imported registered |
| `pathlens_ranking` | 0.868 | 0.375 | 0.492 | 0.397 | 0.424 | freeze `646700d4` |

`pathlens_ranking` confirmation (seeds 13/29/71): MRR 0.373 ± 0.003, hard AUPRC 0.883 ± 0.013.

### Official GNN baselines (this repo)

Same split, seed 13, Tesla T4, BCE 1:1, same decoder. Test sealed.
`skipgnn` adds the `sign(A^2)` channel; `gcn` is original-graph only;
`graphsage` is Hamilton mean-SAGE on unnormalized A (not a Huang et al. paper
number).

| Method | hard AUPRC | MRR | Hits@10 | NDCG@10 | NDCG@50 | Source |
|---|---|---|---|---|---|---|
| `skipgnn` | 0.824 | 0.144 | 0.232 | 0.157 | 0.197 | eval T4 `56970fe` |
| `gcn` | 0.822 | 0.136 | 0.221 | 0.147 | 0.187 | eval T4 `9fa9913` |
| `graphsage` | 0.790 | 0.131 | 0.185 | 0.135 | 0.176 | eval T4 `4138751` |

RRF has the best hard AUPRC in the current scoreboard (0.907).
`resource_allocation` has the best filtered MRR (0.462), slightly above
`three_hop` / z-scored 3-hop (0.455). Official `skipgnn`, `gcn`, and `graphsage` are not close
on ranking (MRR 0.144 / 0.136 / 0.131). They sit with the BCE PathLens
ablations, not with the heuristics. The skip channel moved MRR by +0.008 over
GCN. Mean-SAGE is below both. Selection was validation 1:1 AUROC, which
matches uniform AUROC and does not buy filtered MRR.

The v2 ZIP overlay `binary_skipgnn` (MRR 0.110, Hits@10 0.217, hard AUPRC
0.835) is retired as a freeze baseline.

### PathLens × 3-hop mixes (this repo)

Same sealed validation split, Tesla T4, freeze checkpoint `646700d4`.
Late fusion selected `α=1` on validation MRR: mixing in PathLens never beat
pure 3-hop (best mix was `α=0.4` at MRR 0.402). Per-drug z-scoring leaves
ranks unchanged and lifts hard AUPRC. RRF (`k=60`) is worse on MRR and best
so far on hard AUPRC.

| Method | hard AUPRC | MRR | Hits@10 | NDCG@10 | NDCG@50 | Source |
|---|---|---|---|---|---|---|
| `blend_pathlens_three_hop` | 0.887 | 0.455 | 0.673 | 0.499 | 0.544 | eval T4 `73d5a2a` |
| `rrf_pathlens_three_hop` | 0.907 | 0.393 | 0.567 | 0.428 | 0.458 | eval T4 `73d5a2a` |

`residual_three_hop` is not filed yet.
