# Kaggle

One notebook: `pathlens_training.ipynb`. Set `METHOD` to a folder name under
`methods/`. Default `STAGE` is `smoke` so Save & Run All cannot open the test.

Use two processes for two methods (`cuda:0` and `cuda:1`). Do not DataParallel.

The v2 test stays sealed. `FINAL_TEST_TOKEN` stays empty until freeze.
