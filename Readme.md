# ICLFE / ICLFS Code

This directory contains the training and evaluation code for ICLFS, an
inverted contrastive learning method for unsupervised feature selection.

## Install

From the repository root, install the required packages with:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Place all benchmark `.mat` files in the repository `dataset/` directory. The
loaders expect these exact filenames:

- `ALLAML.mat`
- `arcene.mat`
- `BASEHOCK.mat`
- `COIL20.mat`
- `lung.mat`
- `nci9.mat`
- `PCMAC.mat`
- `Prostate_GE.mat`
- `RELATHE.mat`
- `warpPIE10P.mat`

## Reproduce Paper Results

From the `src/` directory, run:

```bash
python icl_eval.py
```

This runs ICLFS on all 10 benchmark datasets using the default paper settings and seed(42)
and reproduces the reported paper results.

## Outputs

Result summaries are written to `src/results/`:

- `ICLFS_<dataset>_summary.csv`: one summary per dataset
- `ICLFS_all_summary.csv`: combined summary over all datasets

## Optional Flags

If you want to change the default paper reported setup, the main useful flags are:

- `--dataset <name>`: run a single dataset
- `--epochs <int>`: training epochs, default `100`
- `--temperature <float>`: InfoNCE temperature
- `--decorrelation-weight`: option to change the decorrelation weight
- `--n-heads` : option to change the number of attetntion heads, default `1`

## Notes

- The evaluator standardizes each dataset before training and evaluation.
- The model processes the full inverted feature set in one batch.
- The default paper protocol evaluates subset sizes
  `50, 100, 150, 200, 250, 300`.
