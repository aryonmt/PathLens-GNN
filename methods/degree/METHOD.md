# degree

- **Kind:** heuristic
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** none
- **Train in this repo:** no
- **Status:** not_started
- **Notes:** `log1p(deg(d) * deg(p))` on the context graph. No parameters.
  Scored as a dense matrix on GPU (`cuda:0` / `cuda:1`).

Runs belong in `runs/biosnap-dti-v2/degree/<stage>/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
