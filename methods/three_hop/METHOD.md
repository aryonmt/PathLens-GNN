# three_hop

- **Kind:** heuristic
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** none
- **Train in this repo:** no
- **Status:** not_started
- **Notes:** Normalized three-hop bridge (campaign v2 `normalized_three_hop`).
  Same walk as resource allocation, with an extra `1/deg(d')` on the intermediate
  drug. Current validation MRR leader on v2. Scored on GPU.

Runs belong in `runs/biosnap-dti-v2/three_hop/<stage>/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
