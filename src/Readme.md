# ICLFS Code

This directory contains the training and evaluation code for ICLFS, an
inverted contrastive learning method for unsupervised feature selection.

## Datasets

Place all benchmark `.mat` files in the repository `data/` directory. The
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

## Requirements

The code expects a Python environment with:

- `torch`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `pyyaml`

## Reproducing paper results

From the `src/` directory, run:

```bash
python icl_eval.py
```

This trains and evaluates ICLFS on the full paper dataset list using the
default paper settings.

## Outputs

Result summaries are written to `src/results/`:

- `ICLFS_<dataset>_summary.csv`: one summary per dataset
- `ICLFS_all_summary.csv`: combined summary over all datasets

## Useful flags

- `--dataset <name>`: run a single dataset
- `--seed <int>`: random seed, default `42`
- `--epochs <int>`: training epochs, default `100`
- `--kmeans-runs <int>`: clustering runs per subset size, default `20`
- `--temperature <float>`: InfoNCE temperature, default `0.05`
- `--config-path <path>`: optional path to `config.yaml`
- `--out <path>`: optional path for the combined summary CSV

## Notes

- The evaluator standardizes each dataset before training and evaluation.
- The model processes the full inverted feature set in one batch.
- The default paper protocol evaluates subset sizes
  `50, 100, 150, 200, 250, 300`.
