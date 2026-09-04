"""
Run fit_pro.py across a fixed list of datasets, in sequence.

Datasets default to: autompg, concrete, forest, housing, machine, servo, solar, stock.
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASETS = [
    "autompg",
    "concrete",
    "forest",
    "housing",
    "machine",
    "servo",
    "solar",
    "stock",
]


def _run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help=f"comma-separated ds@_global_ overrides (default: {','.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument(
        "--splits",
        default="1,2,3,4,5",
        help="comma-separated split list (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--n-jobs",
        default="-1",
        help="hydra.launcher.n_jobs for the joblib launcher (default: -1)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="override cfg.val_fraction (default: fit_pro.yaml's own default)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="override cfg.alpha (default: fit_pro.yaml's own default)",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=None,
        help="override cfg.num_particles (default: fit_pro.yaml's own default)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="override cfg.name -- results dir suffix, e.g. name=untuned -> pro_gp_untuned "
        "(default: fit_pro.yaml's own default, i.e. no suffix)",
    )
    args = parser.parse_args()

    launcher = ["-m", "hydra/launcher=joblib", f"hydra.launcher.n_jobs={args.n_jobs}"]
    splits = f"split={args.splits}"
    fit_pro = str(SCRIPT_DIR / "fit_pro.py")

    overrides = []
    if args.val_fraction is not None:
        overrides.append(f"val_fraction={args.val_fraction}")
    if args.alpha is not None:
        overrides.append(f"alpha={args.alpha}")
    if args.num_particles is not None:
        overrides.append(f"num_particles={args.num_particles}")
    if args.name is not None:
        overrides.append(f"name={args.name}")

    for dataset in args.datasets.split(","):
        ds = f"ds@_global_={dataset}"
        _run([sys.executable, fit_pro, *launcher, ds, splits, *overrides])


if __name__ == "__main__":
    main()
