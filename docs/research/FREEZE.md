# Freeze record — `biosnap-dti-v2`

## Decision

**Negative freeze.** No learned method beat `three_hop` **and** `skipgnn` on
validation filtered MRR (`strict_gt`, `all_positive` mask).

Preregistered rule ([`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md)):

> Select on validation filtered MRR. Tie-break: validation hard AUPRC.
> Freeze only if the candidate beats `three_hop` and `skipgnn` on MRR.
> Otherwise report the negative result. Open test at most once after freeze.

Filed validation MRR: `resource_allocation` 0.462, `three_hop` 0.455,
`pathlens_ranking` 0.375, `skipgnn` 0.144.

`pathlens_ranking` beats SkipGNN and loses to 3-hop. `resource_allocation` is
the MRR leader but was **not** added to the freeze rule after the fact.

Average-tie diagnostics would make PathLens look like a narrow winner. The
rule was **not** changed after seeing that table.

## What opens now

`STAGE=final` with `FINAL_TEST_TOKEN=OPEN_SEALED_TEST_ONCE` scores test
**once**. Winner and metric stay frozen on validation. Test is confirmation.

The Kaggle notebook `kaggle/pathlens_training.ipynb` (`SUITE=delivery`) is
the operator.

## What must not happen after test

- No α re-sweep on test.
- No epoch re-pick on test.
- No switching `strict_gt` → `average`.
- No dropping RA from the scoreboard.
- No comparing to Huang et al. 0.928.
