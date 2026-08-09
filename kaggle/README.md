# Kaggle Training Runbook

Kaggle is the authoritative environment for PyTorch smoke training, registered tuning, multi-seed confirmation, and artifact export. Local development intentionally does not require PyTorch.

## Kernel settings

- Accelerator: GPU (P100/T4 or current available GPU)
- Internet: enabled while cloning the public repository and downloading BioSNAP
- Persistence: save `/kaggle/working/output` as a notebook version/output before the session ends

## Flow

1. Clone `https://github.com/aryonmt/PathLens-GNN.git`.
2. Install the repository editable with `--no-deps`; Kaggle's existing PyTorch/NumPy/SciPy stack is preferred.
3. Download or attach the BioSNAP TSV and run `pathlens prepare-data`.
4. Run a short smoke configuration before the registered search.
5. Run `scripts/run_tuning.py`; it resumes completed trial directories and stops at 24 trials or four wall-clock days.
6. Rerun the two leading configurations on seeds 13, 29, and 71.
7. Record the model-freeze decision, then run `scripts/evaluate_checkpoint.py` once with the
   explicit sealed-test confirmation flag.
8. Export the chosen checkpoint with `scripts/export_artifact.py` and download the artifact ZIP.

Do not inspect the test arrays during tuning. The final-evaluation command refuses to run unless a
freeze record is supplied and the operator passes the explicit sealed-test confirmation flag.
