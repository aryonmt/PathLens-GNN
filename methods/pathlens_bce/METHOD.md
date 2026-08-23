# pathlens_bce

- **Kind:** trained model
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** one-shot STAGE=final after negative freeze
- **Loss:** BCE 1:1
- **Train in this repo:** no
- **Status:** done
- **Notes:** Imported from campaign v2 registered `full_adaptive_fusion`. Do not retrain.
  Checkpoint SHA-256 `aef8a3eae41eafd656b8ad41891460afa54f3372ca4176e25f091002fb98ae36`.
  Validation (v2 report): hard AUPRC 0.852, MRR 0.160, Hits@10 0.317,
  NDCG@10 0.187, NDCG@50 0.235.

Runs belong in `runs/biosnap-dti-v2/pathlens_bce/imported/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
