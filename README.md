# nonparametricpro

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE.md)

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/callum-kl/non-parametric-pro/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/callum-kl/non-parametric-pro/actions/workflows/tests.yml
[linting-badge]:            https://github.com/callum-kl/non-parametric-pro/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/callum-kl/non-parametric-pro/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/callum-kl/non-parametric-pro/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/callum-kl/non-parametric-pro/actions/workflows/docs.yml
[license-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

Sampling Predictively Oriented Posteriors for Gaussian Processes

## About

### Project Team

Callum Lau ([callum_lau@hotmail.com](mailto:callum_lau@hotmail.com))

<!-- TODO: how do we have an array of collaborators ? -->

## Built With

<!-- TODO: can cookiecutter make a list of frameworks? -->

- [Framework 1](https://something.com)
- [Framework 2](https://something.com)
- [Framework 3](https://something.com)

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`non-parametric-pro` requires Python 3.11&ndash;3.14.

### Installation

<!-- How to build or install the application. -->

We recommend installing in a project specific virtual environment created using
a environment management tool such as
[uv](https://docs.astral.sh/uv/). To install the latest
development version of `non-parametric-pro` using `uv pip` in the currently active
environment run

```sh
uv pip install git+https://github.com/callum-kl/non-parametric-pro.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/callum-kl/non-parametric-pro.git
```

and then install in editable mode by running

```sh
uv pip install -e .
```

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

Test NLPD for all 16 datasets over 5 splits:

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

This fits the exact graph GP, then `pro_gp_gibbs`, for num_train ∈ {200, 225, 250, 275} × 10 splits.
`run_pems.py` sets automatically.

Road-map figures (num_train = 250, split 1 by default; these need the fits above):

```sh
python experiments/pems/scripts/plot_map.py --method gp
python experiments/pems/scripts/plot_map.py --method pro
python experiments/pems/scripts/combine_road_maps.py
```

### Running Tests

<!-- How to run tests on your local system. -->

Tests can be run across all compatible Python versions in isolated environments
using [`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

### Building Documentation

The MkDocs HTML documentation can be built locally by running

```sh
tox -e docs
```

from the root of the repository. The built documentation will be written to
`site`.

Alternatively to build and preview the documentation locally, in a Python
environment with the `docs` dependency group installed, run

```sh
mkdocs serve
```

## Roadmap

- [x] Initial Research
- [ ] Minimum viable product <-- You are Here
- [ ] Alpha Release
- [ ] Feature-Complete Release
