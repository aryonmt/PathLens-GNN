# s1_s2

- **Kind:** trained model
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** one-shot STAGE=final after negative freeze
- **Loss:** BCE 1:1
- **Train in this repo:** no
- **Status:** done
- **Notes:** Imported from campaign v2 registered `s1_s2_weighted`. Do not retrain.
  Checkpoint SHA-256 `6d2eb356f000296511ea9ed597a4200c9feecdd0d8acf2590ad25470d8f4bb79`.
  Validation (v2 report): hard AUPRC 0.825, MRR 0.143, Hits@10 0.227,
  NDCG@10 0.153, NDCG@50 0.205.

Runs belong in `runs/biosnap-dti-v2/s1_s2/imported/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
