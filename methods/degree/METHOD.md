# degree

- **Kind:** heuristic
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** none
- **Train in this repo:** no
- **Status:** done
- **Notes:** `log1p(deg(d) * deg(p))` on the context graph. No parameters.
  Popularity baseline; scored on Tesla T4 (`cuda:0`).
  Validation: hard AUPRC 0.774, MRR 0.134, Hits@10 0.211, NDCG@10 0.144,
  NDCG@50 0.182. Well below `three_hop` on MRR, so the 3-hop lead is not degree
  product.

Runs belong in `runs/biosnap-dti-v2/degree/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
