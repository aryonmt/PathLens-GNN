# resource_allocation

- **Kind:** heuristic
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** none
- **Train in this repo:** no
- **Status:** not_started
- **Notes:** Same-type RA/AA projection, then one hop to the opposite part.
  `W[d, d'] = sum_p B[d,p] B[d',p] / deg(p)`; `score(d,p) = sum_{d' ≠ d} W[d,d'] B[d',p]`.
  This is **not** the normalized three-hop bridge. Scored on GPU.

Runs belong in `runs/biosnap-dti-v2/resource_allocation/<stage>/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
