# ranking_diagnostics

- **Kind:** diagnostic (not a freeze candidate)
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** validation only (not scored on test)
- **Loss:** none
- **Train in this repo:** no
- **Status:** done
- **Notes:** Tesla T4 (`cuda:0`), 5.2s GPU. Rescored `degree`,
  `resource_allocation`, `three_hop`, and frozen `pathlens_ranking`.
  Filed freeze metric stays `strict_gt` + `all_positive`. Under average
  ties, PathLens MRR 0.375 is above 3-hop 0.362 and RA 0.352. RA/3-hop
  have score 0 on 28.6% of validation positives; PathLens has no ties.
  The `visible` filter (no test in the mask) lowers every method.
  Heuristics lead on low-degree drugs; PathLens leads on high-degree.
  Does not rewrite filed cards.

Runs belong in `runs/biosnap-dti-v2/ranking_diagnostics/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
