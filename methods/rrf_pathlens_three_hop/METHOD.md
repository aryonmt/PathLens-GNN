# rrf_pathlens_three_hop

- **Kind:** rank fusion of frozen scores
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** none
- **Train in this repo:** no
- **Status:** done
- **Notes:** Reciprocal rank fusion of `three_hop` and frozen
  `pathlens_ranking` with `k=60`. Mean-rank MRR 0.384 (diagnostic). Tesla T4
  (`cuda:0`), same session as `blend_pathlens_three_hop`. Validation: hard
  AUPRC 0.907, MRR 0.393, Hits@10 0.567, NDCG@10 0.428, NDCG@50 0.458.
  Best hard AUPRC on the board; loses filtered MRR to 3-hop / RA.

Runs belong in `runs/biosnap-dti-v2/rrf_pathlens_three_hop/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
