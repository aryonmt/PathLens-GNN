# Project Charter

## Objective

Compare leakage-safe methods for transductive drug–target ranking on canonical
BioSNAP DTI, and write an honest multi-metric report. This is not a product.

## Audiences

University reviewers and researchers who need ranked candidates without clinical
interpretation.

## Required outcomes

1. One folder per method (`methods/`) and one folder per run (`runs/`).
2. A living scoreboard in `docs/STATUS.md`.
3. Heuristics and trained models on the same split and the same eval suite.
4. SkipGNN trained in this repository. Paper PR-AUC is not a baseline number.
5. Negative freeze, then one-shot test. Do not retune on test.

## Out of scope

Product UI, APIs, clinical claims, DDI/PPI in phase 1, using the v2 test for
selection, DataParallel on this graph.
