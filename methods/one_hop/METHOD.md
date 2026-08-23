# one_hop

- **Kind:** trained model
- **Campaign:** `biosnap-dti-v2` (BioSNAP DTI, split seed 41)
- **Test set:** one-shot STAGE=final after negative freeze
- **Loss:** BCE 1:1
- **Train in this repo:** no
- **Status:** done
- **Notes:** Imported from campaign v2 registered `one_hop`. Do not retrain.
  Checkpoint SHA-256 `bc7a4d503ddeab000085dd783288336de3dab7f1b01eef8bfd34ad90b9178e82`.
  Validation (v2 report): hard AUPRC 0.788, MRR 0.103, Hits@10 0.192,
  NDCG@10 0.114, NDCG@50 0.161.

Runs belong in `runs/biosnap-dti-v2/one_hop/imported/`.
Update `docs/STATUS.md` in the same change that flips status to `done`.
