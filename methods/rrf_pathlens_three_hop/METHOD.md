# rrf_pathlens_three_hop

- **Kind:** rank fusion of frozen scores
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** none
- **Train in this repo:** no
- **Status:** not_started
- **Notes:** Reciprocal rank fusion of `three_hop` and frozen
  `pathlens_ranking` with `k=60`. Mean-rank fusion is stored as a
  diagnostic, not the card score. Written by the same Kaggle eval as
  `blend_pathlens_three_hop`.

Runs belong in `runs/biosnap-dti-v2/rrf_pathlens_three_hop/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
