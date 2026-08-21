"""Run fit_pro_alpha_cv.py across a fixed list of datasets, in sequence.

Datasets default to: whitewine, wine, airquality, abalone, airfoil.

--val-fraction/--c-grid/--num-particles/--name are optional overrides for
fit_pro_alpha_cv.py's cfg.val_fraction/cfg.c_grid/cfg.num_particles/cfg.name -- if not
given, no override is passed and fit_pro_alpha_cv.yaml's own defaults apply.

Usage:
    python run_pro_alpha_cv_datasets.py
    python run_pro_alpha_cv_datasets.py --val-fraction 0.3 --c-grid 10,25,50,100 --num-particles 100
    python run_pro_alpha_cv_datasets.py --name untuned
    python run_pro_alpha_cv_datasets.py --splits 1,2,3 --n-jobs 4
    python run_pro_alpha_cv_datasets.py --datasets whitewine,wine
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASETS = ["whitewine", "wine", "airquality", "abalone", "airfoil"]


def _run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets", default=",".join(DEFAULT_DATASETS),
        help=f"comma-separated ds@_global_ overrides (default: {','.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument("--splits", default="1,2,3,4,5", help="comma-separated split list (default: 1,2,3,4,5)")
    parser.add_argument("--n-jobs", default="-1", help="hydra.launcher.n_jobs for the joblib launcher (default: -1)")
    parser.add_argument(
        "--val-fraction", type=float, default=None,
        help="override cfg.val_fraction (default: fit_pro_alpha_cv.yaml's own default)",
    )
    parser.add_argument(
        "--c-grid", default=None,
        help="comma-separated override for cfg.c_grid, e.g. '10,25,50,100' "
             "(default: fit_pro_alpha_cv.yaml's own default)",
    )
    parser.add_argument(
        "--num-particles", type=int, default=None,
        help="override cfg.num_particles (default: fit_pro_alpha_cv.yaml's own default)",
    )
    parser.add_argument(
        "--name", default=None,
        help="override cfg.name -- results dir suffix, e.g. name=untuned -> pro_gp_alpha_cv_untuned "
             "(default: fit_pro_alpha_cv.yaml's own default, i.e. no suffix)",
    )
    args = parser.parse_args()

    launcher = ["-m", "hydra/launcher=joblib", f"hydra.launcher.n_jobs={args.n_jobs}"]
    splits = f"split={args.splits}"
    fit_pro_alpha_cv = str(SCRIPT_DIR / "fit_pro_alpha_cv.py")

    overrides = []
    if args.val_fraction is not None:
        overrides.append(f"val_fraction={args.val_fraction}")
    if args.c_grid is not None:
        overrides.append(f"c_grid=[{args.c_grid}]")
    if args.num_particles is not None:
        overrides.append(f"num_particles={args.num_particles}")
    if args.name is not None:
        overrides.append(f"name={args.name}")

    for dataset in args.datasets.split(","):
        ds = f"ds@_global_={dataset}"
        _run([sys.executable, fit_pro_alpha_cv, *launcher, ds, splits, *overrides])


if __name__ == "__main__":
    main()
