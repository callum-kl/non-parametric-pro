"""Run exact-GP fitting followed by PRO sampling across several splits, in sequence."""

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
    parser.add_argument(
        "--splits",
        default="1,2,3,4,5",
        help="comma-separated split list (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--n-jobs",
        default="1",
        help=(
            "hydra.launcher.n_jobs for the joblib launcher (default: 1 -- sequential). "
            "Each parallel worker holds its own copy of GraphKernel's "
            "(num_nodes, num_nodes) eigendecomposition, so raise this cautiously."
        ),
    )
    parser.add_argument(
        "--pro-config",
        default="fit_pro_gibbs",
        help="Hydra config name for the PRO stage (default: fit_pro_gibbs)",
    )
    args = parser.parse_args()

    launcher = ["-m", "hydra/launcher=joblib", f"hydra.launcher.n_jobs={args.n_jobs}"]
    splits = f"split={args.splits}"

    exact_gp = str(SCRIPT_DIR / "fit_exact_gp.py")
    pro_gp = str(SCRIPT_DIR / "fit_pro.py")

    runs = [
        [sys.executable, exact_gp, *launcher, splits],
        [
            sys.executable,
            pro_gp,
            f"--config-name={args.pro_config}",
            *launcher,
            splits,
        ],
    ]

    for run_args in runs:
        _run(run_args)


if __name__ == "__main__":
    main()
