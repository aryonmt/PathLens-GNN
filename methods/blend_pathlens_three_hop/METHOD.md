# blend_pathlens_three_hop

- **Kind:** late fusion of frozen scores
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** one-shot STAGE=final after negative freeze
- **Loss:** none
- **Train in this repo:** no
- **Status:** done
- **Notes:** Per-drug z-score mix
  `α · three_hop + (1-α) · pathlens_ranking`. Selected `α=1` on validation
  MRR (pure z-scored 3-hop). Mixing PathLens never beat 3-hop; best mix was
  `α=0.4` at MRR 0.402. Tesla T4 (`cuda:0`), freeze checkpoint `646700d4`.
  Validation: hard AUPRC 0.887, MRR 0.455, Hits@10 0.673, NDCG@10 0.499,
  NDCG@50 0.544. Ranking matches `three_hop`; AUPRC rose from per-drug
  z-scoring.

Runs belong in `runs/biosnap-dti-v2/blend_pathlens_three_hop/eval/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
