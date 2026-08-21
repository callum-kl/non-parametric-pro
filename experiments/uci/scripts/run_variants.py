"""Run the 6 exact-GP/PRO-CV variant combinations for one dataset, in sequence."""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="ds@_global_ override, e.g. autompg")
    parser.add_argument("--splits", default="1,2,3,4,5", help="comma-separated split list (default: 1,2,3,4,5)")
    parser.add_argument("--n-jobs", default="-1", help="hydra.launcher.n_jobs for the joblib launcher (default: -1)")
    parser.add_argument("--kernel-adapt-steps", default="150", help="kernel_adapt_steps for the *_adapted variants (default: 150)")
    args = parser.parse_args()

    ds = f"ds@_global_={args.dataset}"
    launcher = ["-m", "hydra/launcher=joblib", f"hydra.launcher.n_jobs={args.n_jobs}"]
    splits = f"split={args.splits}"

    exact_gp = str(SCRIPT_DIR / "fit_exact_gp.py")
    pro_gp_cv = str(SCRIPT_DIR / "fit_pro_cv.py")

    runs = [
        [sys.executable, exact_gp, *launcher, ds, splits],
        [sys.executable, exact_gp, *launcher, ds, splits, "objective=loocv", "name=loo"],
        [sys.executable, pro_gp_cv, *launcher, ds, splits],
        [sys.executable, pro_gp_cv, *launcher, ds, splits, f"kernel_adapt_steps={args.kernel_adapt_steps}", "name=adapted"],
        [sys.executable, pro_gp_cv, *launcher, ds, splits, "gp_name=loo", "name=loo"],
        [sys.executable, pro_gp_cv, *launcher, ds, splits, f"kernel_adapt_steps={args.kernel_adapt_steps}", "gp_name=loo", "name=loo_adapted"],
    ]

    for run_args in runs:
        _run(run_args)


if __name__ == "__main__":
    main()
