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
parameter-free `three_hop` heuristic. Freeze compares against `three_hop` and
`skipgnn`, not against other PathLens siblings.

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
| `three_hop` | 0.847 | 0.455 | 0.673 | 0.499 | 0.544 | eval T4 `46fc64f` |
| `degree` | — | — | — | — | — | not_started |
| `resource_allocation` | — | — | — | — | — | not_started |

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

`pathlens_ranking` has the best hard AUPRC in the current scoreboard.
`three_hop` has the best filtered MRR. That split is the result, not a reason
to drop the ablations.

The v2 ZIP also scored `binary_skipgnn` (MRR 0.110, Hits@10 0.217, hard AUPRC
0.835). That overlay is **not** the official `skipgnn` card; train SkipGNN in
this repo before using it as a freeze baseline.

Official SkipGNN, GCN, and GraphSAGE still have empty run cards.
