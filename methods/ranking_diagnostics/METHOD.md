# ranking_diagnostics

- **Kind:** diagnostic (not a freeze candidate)
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** none
- **Train in this repo:** no
- **Status:** not_started
- **Notes:** Wave-1 audit of the filed ranking metric. Rescores `degree`,
  `resource_allocation`, `three_hop` on GPU, and on `STAGE=eval` also the
  frozen `pathlens_ranking` checkpoint. Reports MRR under `strict_gt` (filed
  default), `average` ties, and seeded `random` ties; zero-score / tie-group
  stats; MRR by drug-degree tertile; and a `visible` filter
  (`context ∪ train ∪ val`) versus `all_positive` (canonical edges, so test
  positives are in the mask only — test arrays are never read). Does not
  rewrite filed cards.

Runs belong in `runs/biosnap-dti-v2/ranking_diagnostics/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
