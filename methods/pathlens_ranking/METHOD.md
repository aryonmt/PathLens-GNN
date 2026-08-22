# pathlens_ranking

- **Kind:** trained model
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** sealed
- **Loss:** sampled softmax / InfoNCE (64 negatives, 25% hard mix)
- **Train in this repo:** no
- **Status:** done
- **Notes:** Imported freeze checkpoint (confirmation seed 13). Do not retrain.
  Checkpoint SHA-256 `646700d456ba4cb0de795e7067e1830e8f18598b97fe5cc89d195d631d356586`.
  Validation seed 13 (v2 report): hard AUPRC 0.868, MRR 0.375, Hits@10 0.492,
  NDCG@10 0.397, NDCG@50 0.424.
  Confirmation seeds 13/29/71: MRR 0.373 ± 0.003, hard AUPRC 0.883 ± 0.013.
  Freeze vs three-hop on this split is a negative result on MRR; test stays sealed.

Runs belong in `runs/biosnap-dti-v2/pathlens_ranking/imported/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
