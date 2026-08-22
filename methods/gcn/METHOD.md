# gcn

- **Kind:** trained model
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** BCE 1:1
- **Train in this repo:** yes
- **Status:** done
- **Notes:** Two-layer GCN on the original SkipGNN adjacency (`A + I`, symmetric
  normalization). Same decoder, loss, and hyperparameters as official `skipgnn`.
  No skip graph. Trained on Tesla T4 (`cuda:0`), seed 13, 15 epochs, best
  validation AUROC at epoch 8. Validation: hard AUPRC 0.822, MRR 0.136,
  Hits@10 0.221, NDCG@10 0.147, NDCG@50 0.187.

Runs belong in `runs/biosnap-dti-v2/gcn/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
