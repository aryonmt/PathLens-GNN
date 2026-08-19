# Kaggle Training Runbook

Kaggle is the authoritative GPU environment for smoke training, registered
experiments, tuning, multi-seed confirmation, freeze, and the one-time sealed
evaluation. Local development does not require a CUDA installation.

Campaign `biosnap-dti-canonical-v2` uses split seed 41. Do not restore a v1
processed dataset or a v1 stage ZIP into this campaign.

## Kernel settings

- Accelerator: GPU (P100, T4, or the current available GPU)
- Internet: enabled while cloning the public repository and downloading BioSNAP
- Persistence: preserve `/kaggle/working/pathlens-stage-output.zip` before a session ends

Install the repository with `pip install -e . --no-deps`. This deliberately keeps
Kaggle's CUDA-compatible PyTorch stack. `scripts/capture_environment.py` records
the actual versions used.

## Safe staged execution

The first notebook cell exposes one `STAGE` value:

| Stage | Purpose | Required restored output |
|---|---|---|
| `smoke` | Three-epoch GPU and pipeline check | none |
| `registered` | Heuristics and preregistered model/baseline suite | none |
| `tuning` | Up to 24 seed-13 validation configurations | prior tuning ZIP when resuming |
| `confirmation` | Top two configurations on seeds 13, 29, and 71 | completed tuning ZIP |
| `freeze` | Persist the validation-only model choice and checkpoint hash | confirmation ZIP |
| `final` | Open the sealed test exactly once | frozen output ZIP and explicit token |

The committed default is `STAGE = "smoke"`. Therefore **Save & Run All is safe by
default**: it cannot tune or access test arrays. Change only one stage per Kaggle
version.

`pathlens_training.ipynb` is the only committed training notebook. Do not add
session-specific operator notebooks to the repository.

## Resume between Kaggle sessions

Every successful stage creates:

```text
/kaggle/working/pathlens-stage-output.zip
```

Download it or save the notebook version so it becomes an output. In the next
session, add that output through Kaggle's **Add Input** panel and set
`RESUME_ARCHIVE` to the resulting path, for example:

```python
RESUME_ARCHIVE = "/kaggle/input/pathlens-tuning-output/pathlens-stage-output.zip"
```

The notebook validates ZIP paths before extraction. Registered experiments,
tuning trials, and multi-seed confirmation skip complete checkpoint/metrics pairs
and continue missing work. `run_tuning.py` persists `run_state.json`, binds the
resume to the registered search-space hash, and accumulates active runtime across
sessions. A changed search space or budget is rejected. Each Kaggle tuning
invocation stops after nine hours by default, leaving time for the final archive
cell to produce the resumable ZIP before the hosted session limit.

## Sealed-test procedure

The test set stays unopened through `smoke`, `registered`, `tuning`,
`confirmation`, and `freeze`. After freeze is on disk, run `final` with:

```python
STAGE = "final"
FINAL_TEST_TOKEN = "OPEN_SEALED_TEST_ONCE"
```

The final stage refuses to run without a matching freeze record and refuses to
overwrite an existing `final-evaluation.json`. Never use the test output to
change the model or hyperparameters. After this split's test is opened, later
architecture work needs a new preregistered split.

## Observed smoke evidence

The first manual Kaggle smoke run passed on a Tesla T4 with Python 3.12.13 and
PyTorch 2.10.0+cu128. Canonical counts were 15,138 edges, 5,017 drugs, 2,324
proteins, and 7,341 entities. The three-epoch run completed in 16.315 seconds on
CUDA. Its validation AUPRC is operational evidence only and is not a research
result.
