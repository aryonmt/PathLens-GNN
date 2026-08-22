# skipgnn

- **Kind:** trained model
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** BCE 1:1
- **Train in this repo:** yes
- **Status:** not_started
- **Notes:** Official Huang et al. SkipGNN encoder (`legacy1`): original GCN +
  `sign(A^2)` skip graph, two-layer linear decoder, one-hot features. Data and
  eval bugs from the clone are repaired (canonical BioSNAP, typed 1:1
  negatives, no `drop_last` on leftover pairs, pair indices on device,
  `BCEWithLogitsLoss`, official validation scoring after AUROC selection).
  Do not use Huang et al. PR-AUC 0.928 or the imported `binary_skipgnn` overlay.

Runs belong in `runs/biosnap-dti-v2/skipgnn/<stage>/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
