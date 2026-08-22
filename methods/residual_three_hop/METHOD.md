# residual_three_hop

- **Kind:** trained residual on a frozen heuristic
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** sampled softmax (64 negatives, 25% hard mix)
- **Train in this repo:** yes
- **Status:** not_started
- **Notes:** `score = three_hop + f_θ(d, p)` where `f_θ` is a small pair MLP
  on embeddings. The 3-hop matrix is frozen. Checkpoint selection is
  validation filtered MRR, not 1:1 AUROC. This is a new experiment, not a
  Huang et al. paper number and not a PathLens retrain. Next Kaggle card
  after filed blend/RRF. Do not open the sealed test.

Runs belong in `runs/biosnap-dti-v2/residual_three_hop/<stage>/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
