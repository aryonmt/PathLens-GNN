# skipgnn

- **Kind:** trained model
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** BCE 1:1
- **Train in this repo:** yes
- **Status:** done
- **Notes:** Official Huang et al. SkipGNN encoder (`legacy1`): original GCN +
  `sign(A^2)` skip graph, two-layer linear decoder, one-hot features. Data and
  eval bugs from the clone are repaired. Do not use Huang et al. PR-AUC 0.928
  or the imported `binary_skipgnn` overlay. Trained on Tesla T4 (`cuda:0`),
  seed 13, 15 epochs, best validation AUROC at epoch 6. Validation: hard AUPRC
  0.824, MRR 0.144, Hits@10 0.232, NDCG@10 0.157, NDCG@50 0.197.

Runs belong in `runs/biosnap-dti-v2/skipgnn/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
