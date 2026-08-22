# graphsage

- **Kind:** trained model
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** BCE 1:1
- **Train in this repo:** yes
- **Status:** not_started
- **Notes:** Hamilton GraphSAGE-mean on the unnormalized original adjacency
  (no self-loops; self is concatenated via `W_self`). Same two-layer widths
  (64/16), decoder, loss, and optimizer as official `gcn` / `skipgnn`. Huang
  et al. did **not** report GraphSAGE; their GNN table is GCN, GIN, JK-Net,
  MixHop. This card is the campaign's mean-SAGE baseline, not a paper number.

Runs belong in `runs/biosnap-dti-v2/graphsage/<stage>/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
