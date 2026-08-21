"""
Run the SVGP / PPGPR / PRO-CV (inducing) variant combination for one dataset, in sequence.

Reproduces these 3 hydra multirun invocations (results dirs in parens):

    fit_vgp.py                                    (vgp)
    fit_ppgpr.py                                  (ppgpr)
    fit_pro_cv.py inducing=true                   (inducing_pro_gp_cv, seeded from the vgp fit above)

Usage:
    python experiments/uci/run_inducing_variants.py autompg
    python experiments/uci/run_inducing_variants.py autompg --splits 1,2,3 --n-jobs 4
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="ds@_global_ override, e.g. autompg")
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
    args = parser.parse_args()

    ds = f"ds@_global_={args.dataset}"
    launcher = ["-m", "hydra/launcher=joblib", f"hydra.launcher.n_jobs={args.n_jobs}"]
    splits = f"split={args.splits}"

    fit_vgp = str(SCRIPT_DIR / "fit_vgp.py")
    fit_ppgpr = str(SCRIPT_DIR / "fit_ppgpr.py")
    fit_pro_cv = str(SCRIPT_DIR / "fit_pro_cv.py")

    runs = [
        [sys.executable, fit_vgp, *launcher, ds, splits],
        [sys.executable, fit_ppgpr, *launcher, ds, splits],
        [sys.executable, fit_pro_cv, *launcher, ds, splits, "inducing=true"],
    ]

    for run_args in runs:
        _run(run_args)


if __name__ == "__main__":
    main()
