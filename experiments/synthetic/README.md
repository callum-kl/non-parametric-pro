# Synthetic experiments

Produces `figures/example_grid_and_summary_columns.png`: illustrative Bayes-GP vs
PrO-GP fits on the left, held-out NLPD against training size on the right.

```sh
./scripts/run_sweep.sh
```

Or step by step, from `scripts/`:

```sh
# 4 regimes x n in {100,200,400} x 2 methods x 20 splits -> results/<source>/n_<n>/<algorithm>/
python synthetic.py -m \
  ds@_global_=block_outliers,heteroskedastic,multimodal,well_specified \
  algorithm=standard_gp,pro_gp n=50,100,200

python aggregate_results.py            # -> results/summary.csv
python combine_grid_summary_columns.py # -> figures/example_grid_and_summary_columns.png
```

## Data-generating processes

The **right-hand NLPD panels** use the benchmark generators in
`src/non_parametric_pro/data/synthetic/{well_specified,block_outliers,heteroskedastic,multimodal}.py`:
an RBF kernel with lengthscale `l ~ U[0.5,1]` and amplitude `a ~ U[0.5,2]`, inputs
`X ~ U[-2,2]^n`, latent `f ~ GP(0, K)`, observations `y = f(X) + e` with
`e ~ N(0, (0.15a)^2)`, split 70/30 into train/test. Each regime then corrupts the
training data in one way (a contiguous outlier block, a local noise bump, or a
two-branch mixture).

The **left-hand illustrative panels** use
`src/non_parametric_pro/data/synthetic/illustrative.py` instead, which fixes one
latent, `f(x) = sin(2 pi x) + 0.3 cos(4 pi x)` on `[0,1]`, and hand-picks each
corruption's magnitude so the panels are comparable and easy to read. These
datasets are for the figure only; no reported number depends on them.
