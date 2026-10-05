# nonparametricpro

Sampling Predictively Oriented Posteriors for Gaussian Processes

## Built With

- [GPJax](https://gpjax.quantclimate.com/)
- [BlackJax](https://blackjax-devs.github.io/blackjax/)


### Prerequisites

Requires Python 3.11&ndash;3.14.

### Reproducing the results

All commands are run from the repository root in an environment where the package is
installed (`uv pip install -e .`). Scripts use absolute paths, so they can be run from
any directory. Each experiment writes its runs to `experiments/<experiment>/results/` and
its figures to `experiments/<experiment>/figures/`.

#### Synthetic

```sh
./experiments/synthetic/scripts/run_sweep.sh
```

This runs the four regimes (block outliers, heteroskedastic, multimodal, well
specified) × n ∈ {50, 100, 200} × {standard GP, PrO-GP}, then produces:

- `results/summary.csv` (`aggregate_results.py`)
- `figures/example_grid_and_summary_columns.png` (`combine_grid_summary_columns.py`)
- `figures/multimodal_overlay.png` (`multimodal_overlay.py`)

Bayesian overlapping mixture of GPs (OMGP) baselines, fitted by Gibbs in the same
Cholesky basis and with the same kernel as PrO-GP (needs the sweep above):

```sh
./experiments/synthetic/scripts/run_mixture_sweep.sh
```

This fits OMGP with K ∈ {1, 2, 3, 5, 10}, a sparse-Dirichlet OMGP (K = 10) and OMGP
with K chosen on a validation split, on the four regimes and on a multi-branch regime
with K* ∈ {1, 2, 3, 4} branches. It also sweeps PrO-GP's particle count against OMGP's
component count, then produces:

- `results/omgp_k_sensitivity.csv`, `results/mixture_multibranch.csv` (`plot_mixture.py`)
- `figures/omgp_k_sensitivity_n100.png`, `figures/multibranch_heatmap.png`,
  `figures/particles_vs_components.png` (`plot_mixture.py`)

Add `mixture.shared_sigma=true` to the `synthetic.py` calls (and `--prefix omgp_shared` to
`plot_mixture.py`) for OMGP with one noise scale shared across components.
- `figures/multibranch_overlay.png` (`multibranch_overlay.py`)


#### UCI

Download the data once.

```sh
python -c "
from non_parametric_pro.data.uci import uci
uci.download_uci_regression_datasets()
uci.download_wine_quality_white()
uci.download_abalone()
uci.download_air_quality()
"
```

Test NLPD for all 16 datasets over 10 splits:

```sh
python experiments/uci/scripts/run_uci.py
```

- Small datasets (machine, autompg, housing, stock, concrete, concreteslump, energy, servo) fit
  `exact_gp`, then `pro_gp_gibbs`.
- The larger datasets fit `vgp_noncollapsed` and `ppgpr`, then
  `inducing_pro_gp_gibbs` and `inducing_pro_gp_gibbs_ppgpr` (seeded from each of those).
- Per-dataset settings live in `experiments/uci/conf/ds/<dataset>.yaml`.
- Use `--datasets machine,wine --splits 1,2` to run a subset, then
  `python experiments/uci/scripts/aggregate_results.py` to rebuild the summary.

Normality figure (needs the `exact_gp` fits above):

```sh
python experiments/uci/scripts/plot_normality.py
```

OMGP baselines on the exact datasets (needs the `exact_gp` fits above):

```sh
python experiments/uci/scripts/run_uci.py --mixture-only --datasets machine,autompg,housing,stock,concrete,concreteslump,energy,servo
python experiments/uci/scripts/plot_mixture_uci.py
```

`fit_mixture.py shared_sigma=true` (with `plot_mixture_uci.py --prefix omgp_shared`) shares one
noise scale across components. On the inducing datasets `fit_mixture.py` uses the saved VGP's
inducing points, e.g. `fit_mixture.py -m ds@_global_=wine split=1,2,3,4,5,6,7,8,9,10 shared_sigma=true`.

PeMS OMGP baselines (needs the exact graph GP fits):

```sh
python experiments/pems/scripts/run_pems.py --mixture-only
```

Convergence and runtime figures (parkinsons, split 1). These run sequentially so the
timings aren't contended, and write to `experiments/uci/convergence_results/`, which is
kept separate from `results/summary.csv`:

```sh
python experiments/uci/scripts/run_convergence.py
python experiments/uci/scripts/run_convergence.py --ms 1000 --seeds 0
python experiments/uci/scripts/plot_convergence.py
```

The m = 1000 run only feeds the time-per-iteration panel, so it can use a smaller
`gp_num_iters=...` override.

#### PeMS

The road-network data is downloaded automatically on first use.

```sh
python experiments/pems/scripts/run_pems.py
```

This fits the exact graph GP, then `pro_gp_gibbs`, for num_train ∈ {200, 225, 250, 275} × 20 splits.

Road-map figures (num_train = 250, split 1 by default; these need the fits above):

```sh
python experiments/pems/scripts/plot_map.py --method gp
python experiments/pems/scripts/plot_map.py --method pro
python experiments/pems/scripts/combine_road_maps.py
```