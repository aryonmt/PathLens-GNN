# Figures

Copy these files from Kaggle `/kaggle/working/figures/` into **this folder**.
Do not generate them on the laptop.

Head-to-head plots drop nested PathLens hops (`one_hop`, `s1_s2`,
`s1_s2_s3_fixed`). Those cards belong on `pathlens_family.png`.

| File | Plot |
|---|---|
| `pr_hard.png` | Precision–recall, hard negatives |
| `pr_uniform.png` | Precision–recall, uniform negatives |
| `roc_hard.png` | ROC, hard negatives |
| `hits_at_k.png` | Filtered Hits@K |
| `mrr_and_hard_auprc.png` | MRR vs hard AUPRC (comparison set) |
| `degree_slices_hard.png` | Hard AUPRC by pair-degree tertile |
| `pathlens_family.png` | PathLens ablation ladder |
| `bipartite_hops.png` | Odd vs even walks |
| `blend_alpha_sweep.png` | Late-fusion α sweep |
| `epoch_selection.png` | Checkpoint selection |
| `mrr_by_degree_tertile.png` | Filtered MRR by drug-degree tertile |
| `scoreboard.md` | Tables generated from filed metrics |
| `eda/` | Degree histograms, split sizes, 3-hop zero-mass |
| `val_vs_test_mrr.png` | After `STAGE=final` |

After freeze, matching test-set plots can also live in `test/`.
