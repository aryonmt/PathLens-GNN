# residual_three_hop

- **Kind:** trained residual on a frozen heuristic
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** sampled softmax (64 negatives, 25% hard mix)
- **Train in this repo:** yes
- **Status:** done
- **Notes:** `score = three_hop + f_θ(d, p)` where `f_θ` is a small pair MLP
  on embeddings. The 3-hop matrix is frozen. Checkpoint selection is
  validation filtered MRR, not 1:1 AUROC. Tesla T4 (`cuda:0`), seed 13, max
  40 epochs, patience 8, stopped at epoch 18, best validation MRR at epoch
  10. GPU `score_seconds` 10.4 (same order as official SkipGNN here).
  Validation: hard AUPRC 0.889, MRR 0.367, Hits@10 0.545, NDCG@10 0.404,
  NDCG@50 0.430. Loses filtered MRR to frozen `three_hop` (0.455). Random
  residual at init already sat at MRR 0.366. Not a Huang et al. paper
  number and not a PathLens retrain.

Runs belong in `runs/biosnap-dti-v2/residual_three_hop/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
