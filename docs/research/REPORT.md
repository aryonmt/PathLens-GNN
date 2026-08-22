# Living report — BioSNAP DTI (`biosnap-dti-v2`)

English only. No test numbers until freeze.

## Task

Transductive missing-link ranking on a bipartite drug–protein graph. Negatives
are unknown non-edges.

## Methods

See [`docs/STATUS.md`](../STATUS.md).

## Validation figures

The evaluation phase is this figure set. Files are produced on Kaggle under
`runs/biosnap-dti-v2/figures/`, not on the laptop.

| File | What it shows |
|---|---|
| `pr_hard.png` | Precision–recall on degree-matched unknowns |
| `pr_uniform.png` | Precision–recall on uniform typed unknowns |
| `roc_hard.png` | ROC on degree-matched unknowns |
| `hits_at_k.png` | Filtered Hits@K |
| `mrr_and_hard_auprc.png` | Filtered MRR vs hard AUPRC |
| `degree_slices_hard.png` | Hard AUPRC by endpoint-degree tertile |

Imported PathLens / SkipGNN curves in these figures come from campaign v2
checkpoints. Plot them on Kaggle from `runs/biosnap-dti-v2` with
`scripts/plot_validation_report.py --report runs/biosnap-dti-v2`. Official
`skipgnn` training in this repo will replace that overlay when its run card is
`done`. Do not generate the PNGs on the laptop.

## Validation tables

Imported PathLens cards from campaign v2 (`pathlens-stage-output-v2-report.zip`).
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

The v2 ZIP also scored `normalized_three_hop` (MRR 0.455, Hits@10 0.673, hard AUPRC 0.847) and `binary_skipgnn` (MRR 0.110, Hits@10 0.217, hard AUPRC 0.835). Those are **not** finished cards in this repo: `three_hop` will be re-run here, and `skipgnn` will be trained here.

Heuristics, official SkipGNN, GCN, and GraphSAGE still have empty run cards.
