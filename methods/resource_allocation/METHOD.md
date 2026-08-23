# resource_allocation

- **Kind:** heuristic
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** one-shot STAGE=final after negative freeze
- **Loss:** none
- **Train in this repo:** no
- **Status:** done
- **Notes:** Same-type RA/AA projection, then one hop to the opposite part.
  `W[d, d'] = sum_p B[d,p] B[d',p] / deg(p)`; `score(d,p) = sum_{d' ≠ d} W[d,d'] B[d',p]`.
  This is **not** the extra `1/deg(d')` of `three_hop`. Scored on Tesla T4 (`cuda:0`).
  Validation: hard AUPRC 0.845, MRR 0.462, Hits@10 0.672, NDCG@10 0.504,
  NDCG@50 0.547. Current validation MRR leader; slightly above `three_hop`.

Runs belong in `runs/biosnap-dti-v2/resource_allocation/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
