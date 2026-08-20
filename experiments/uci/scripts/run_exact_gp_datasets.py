"""Run fit_exact_gp.py, for one objective, across a fixed list of datasets, in sequence.

For each dataset, reproduces this hydra multirun invocation:

    fit_exact_gp.py [objective=<objective>] [name=loo]   (exact_gp / exact_gp_loo)

`name=loo` is added automatically for `--objective loocv`, matching results dirs are
keyed off `cfg.name` not `cfg.objective` -- without it, a loocv run would land in the
same `exact_gp` dir as an `mll` run and overwrite it (see run_variants.py's own
`objective=loocv name=loo` pairing).

Datasets default to the small ones this repo fits exact GPs on directly (no inducing
points needed): autompg, concrete, forest, housing, machine, servo, solar, stock.

Usage:
    python experiments/uci/run_exact_gp_datasets.py
    python experiments/uci/run_exact_gp_datasets.py --objective loocv
    python experiments/uci/run_exact_gp_datasets.py --splits 1,2,3 --n-jobs 4
    python experiments/uci/run_exact_gp_datasets.py --datasets autompg,concrete
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASETS = [
    "autompg", "concrete", "forest", "housing", "machine", "servo", "solar", "stock",
]


def _run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets", default=",".join(DEFAULT_DATASETS),
        help=f"comma-separated ds@_global_ overrides (default: {','.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument(
        "--objective", default="mll", choices=["mll", "loocv"],
        help="cfg.objective for fit_exact_gp.py (default: mll)",
    )
    parser.add_argument("--splits", default="1,2,3,4,5", help="comma-separated split list (default: 1,2,3,4,5)")
    parser.add_argument("--n-jobs", default="-1", help="hydra.launcher.n_jobs for the joblib launcher (default: -1)")
    args = parser.parse_args()

    launcher = ["-m", "hydra/launcher=joblib", f"hydra.launcher.n_jobs={args.n_jobs}"]
    splits = f"split={args.splits}"
    exact_gp = str(SCRIPT_DIR / "fit_exact_gp.py")

    overrides = [f"objective={args.objective}"]
    if args.objective == "loocv":
        overrides.append("name=loo")

    for dataset in args.datasets.split(","):
        ds = f"ds@_global_={dataset}"
        _run([sys.executable, exact_gp, *launcher, ds, splits, *overrides])


if __name__ == "__main__":
    main()
