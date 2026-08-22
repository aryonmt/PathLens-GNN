# graphsage

- **Kind:** trained model
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** BCE 1:1
- **Train in this repo:** yes
- **Status:** done
- **Notes:** Hamilton GraphSAGE-mean on the unnormalized original adjacency
  (no self-loops; self is concatenated via `W_self`). Same two-layer widths
  (64/16), decoder, loss, and optimizer as official `gcn` / `skipgnn`. Huang
  et al. did **not** report GraphSAGE; their GNN table is GCN, GIN, JK-Net,
  MixHop. This card is the campaign's mean-SAGE baseline, not a paper number.
  Trained on Tesla T4 (`cuda:0`), seed 13, 15 epochs, best validation AUROC
  at epoch 9 (0.917). Validation: hard AUPRC 0.790, MRR 0.131, Hits@10 0.185,
  NDCG@10 0.135, NDCG@50 0.176.

Runs belong in `runs/biosnap-dti-v2/graphsage/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
