# three_hop

- **Kind:** heuristic
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** none
- **Train in this repo:** no
- **Status:** done
- **Notes:** Normalized three-hop bridge (campaign v2 `normalized_three_hop`).
  Same walk as resource allocation, with an extra `1/deg(d')` on the intermediate
  drug. Scored on Tesla T4 (`cuda:0`).
  Validation: hard AUPRC 0.847, MRR 0.455, Hits@10 0.673, NDCG@10 0.499,
  NDCG@50 0.544. On this split `resource_allocation` is slightly higher on MRR.

Runs belong in `runs/biosnap-dti-v2/three_hop/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
