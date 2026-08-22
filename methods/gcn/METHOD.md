# gcn

- **Kind:** trained model
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** BCE 1:1
- **Train in this repo:** yes
- **Status:** not_started
- **Notes:** Two-layer GCN on the original SkipGNN adjacency (`A + I`, symmetric
  normalization). Same decoder, loss, and hyperparameters as official `skipgnn`.
  No skip graph. This is the paper GCN baseline, not GraphSAGE.

Runs belong in `runs/biosnap-dti-v2/gcn/<stage>/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
