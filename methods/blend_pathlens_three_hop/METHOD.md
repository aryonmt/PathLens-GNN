# blend_pathlens_three_hop

- **Kind:** late fusion of frozen scores
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** none
- **Train in this repo:** no
- **Status:** not_started
- **Notes:** Per-drug z-score mix
  `α · three_hop + (1-α) · pathlens_ranking`. `α` is chosen on
  validation filtered MRR (same split; slightly optimistic). `α=1` is pure
  3-hop. PathLens scores come from the freeze checkpoint `646700d4`, not a
  retrain. Running this card also writes `rrf_pathlens_three_hop`.

Runs belong in `runs/biosnap-dti-v2/blend_pathlens_three_hop/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
