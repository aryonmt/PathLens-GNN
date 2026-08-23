# Living report — BioSNAP DTI (`biosnap-dti-v2`)

English only. Validation numbers below are frozen. Test numbers are filled
only after the one-shot `STAGE=final` run; they must not change the winner.

For the paper critique, model formulas, and why each experiment ran, start at
[`docs/delivery/README.md`](../delivery/README.md).

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
parameter-free `resource_allocation` heuristic. Freeze names `three_hop` and
`skipgnn`; on this split `resource_allocation` is the stronger heuristic.
Head-to-head figures drop the nested hops and keep `pathlens_bce` plus
`pathlens_ranking`.

## Validation figures

Filed under [`runs/biosnap-dti-v2/figures/`](../../runs/biosnap-dti-v2/figures/).
Produced on Kaggle, not on the laptop.

| File | What it shows |
|---|---|
| `pr_hard.png` | Precision–recall on degree-matched unknowns |
| `pr_uniform.png` | Precision–recall on uniform typed unknowns |
| `roc_hard.png` | ROC on degree-matched unknowns |
| `hits_at_k.png` | Filtered Hits@K |
| `mrr_and_hard_auprc.png` | Filtered MRR vs hard AUPRC (comparison set; no nested hops) |
| `pathlens_family.png` | PathLens ablation ladder only |
| `degree_slices_hard.png` | Hard AUPRC by endpoint-degree tertile |
| `bipartite_hops.png` | Odd vs even walks on a bipartite DTI graph |
| `blend_alpha_sweep.png` | Late-fusion α vs validation MRR |
| `epoch_selection.png` | Checkpoint selection curves |
| `mrr_by_degree_tertile.png` | Filtered MRR by drug-degree tertile |
| `val_vs_test_mrr.png` | Paired MRR after `STAGE=final` |

## Validation tables

### Heuristics (this repo)

| Method | hard AUPRC | MRR | Hits@10 | NDCG@10 | NDCG@50 | Source |
|---|---|---|---|---|---|---|
| `resource_allocation` | 0.845 | 0.462 | 0.672 | 0.504 | 0.547 | eval T4 `d2b90ca` |
| `three_hop` | 0.847 | 0.455 | 0.673 | 0.499 | 0.544 | eval T4 `46fc64f` |
| `degree` | 0.774 | 0.134 | 0.211 | 0.144 | 0.182 | eval T4 `5daa27c` |

### PathLens family (imported campaign v2)

NDCG is computed from the stored filtered ranks
(no rescoring, no test access). These imported cards were validation-only
when filed; `STAGE=final` scores the same checkpoints on test.

| Method | hard AUPRC | MRR | Hits@10 | NDCG@10 | NDCG@50 | Source |
|---|---|---|---|---|---|---|
| `one_hop` | 0.788 | 0.103 | 0.192 | 0.114 | 0.161 | imported registered |
| `s1_s2` | 0.825 | 0.143 | 0.227 | 0.153 | 0.205 | imported registered |
| `s1_s2_s3_fixed` | 0.826 | 0.136 | 0.224 | 0.147 | 0.199 | imported registered |
| `pathlens_bce` | 0.852 | 0.160 | 0.317 | 0.187 | 0.235 | imported registered |
| `pathlens_ranking` | 0.868 | 0.375 | 0.492 | 0.397 | 0.424 | freeze `646700d4` |

`pathlens_ranking` confirmation (seeds 13/29/71): MRR 0.373 ± 0.003, hard AUPRC 0.883 ± 0.013.

### Official GNN baselines (this repo)

Same split, seed 13, Tesla T4, BCE 1:1, same decoder.
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

Same validation split, Tesla T4, freeze checkpoint `646700d4`.
Late fusion selected `α=1` on validation MRR: mixing in PathLens never beat
pure 3-hop (best mix was `α=0.4` at MRR 0.402). Per-drug z-scoring leaves
ranks unchanged and lifts hard AUPRC. RRF (`k=60`) is worse on MRR and best
so far on hard AUPRC.

| Method | hard AUPRC | MRR | Hits@10 | NDCG@10 | NDCG@50 | Source |
|---|---|---|---|---|---|---|
| `blend_pathlens_three_hop` | 0.887 | 0.455 | 0.673 | 0.499 | 0.544 | eval T4 `73d5a2a` |
| `rrf_pathlens_three_hop` | 0.907 | 0.393 | 0.567 | 0.428 | 0.458 | eval T4 `73d5a2a` |
| `residual_three_hop` | 0.889 | 0.367 | 0.545 | 0.404 | 0.430 | eval T4 `a5506bb` |

`residual_three_hop` is a tiny pair MLP added to frozen 3-hop. GPU time was
10.4s for 18 epochs (patience 8 after best epoch 10), the same order as
official `skipgnn` (10.3s / 15 epochs) in this repo. Epoch 1 MRR was already
0.366, below frozen 3-hop 0.455: random residual noise scrambled the
heuristic ranks and training barely moved the loss (4.096 → 4.064). This is
a negative result, not a truncated run. Campaign v2 PathLens used up to 300
epochs; that is the 4–5 hour job, not these trainers.

`ranking_diagnostics` (eval T4 `90202e6`) is filed. It is not a freeze
candidate. Filed cards keep `strict_gt` + `all_positive`. Same 1,514
validation queries, Tesla T4, 5.2s.

Validation MRR by tie rule (`all_positive` mask):

| Method | strict (filed) | average | random |
|---|---|---|---|
| `resource_allocation` | 0.462 | 0.352 | 0.356 |
| `three_hop` | 0.455 | 0.362 | 0.364 |
| `pathlens_ranking` | 0.375 | 0.375 | 0.375 |
| `degree` | 0.134 | 0.133 | 0.133 |

RA / 3-hop have score exactly 0 on 28.6% of validation positives; mean
tied-others is huge because of those zeros, median tied-others is 0.
PathLens has no ties. Optimistic ties therefore inflate the heuristics by
about 0.10 MRR, not all the way down to a PathLens collapse.

The `visible` mask (`context ∪ train ∪ val`, test edges not hidden) lowers
every method (strict RA 0.403, 3-hop 0.401, PathLens 0.304). Test arrays
were not read.

Drug-degree tertiles on filed `strict_gt` / `all_positive`: heuristics lead
on **low** degree (RA 0.496 / 3-hop 0.478 vs PathLens 0.250); PathLens leads
on **high** degree (0.466 vs RA 0.410 / 3-hop 0.386). The sparse tail is
not where PathLens wins.

Freeze still names `three_hop` and `skipgnn` on filed `strict_gt` MRR. This
audit did not change that rule. The negative freeze is recorded in
[`FREEZE.md`](FREEZE.md). Test opens once via `STAGE=final`.
