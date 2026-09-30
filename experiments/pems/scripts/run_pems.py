"""Fit the exact graph GP then PRO-GP for every training size and split, then aggregate."""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trains", default="200,225,250,275")
    parser.add_argument("--splits", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-jobs",
        default="1",
        help="joblib workers; each holds a (num_nodes, num_nodes) eigendecomposition",
    )
    args = parser.parse_args()

    launcher = ["-m", "hydra/launcher=joblib", f"hydra.launcher.n_jobs={args.n_jobs}"]
    for num_train in map(int, args.num_trains.split(",")):
        overrides = [
            f"split={args.splits}",
            f"num_train={num_train}",
            f"seed={args.seed}",
        ]
        for script in ("fit_exact_gp.py", "fit_pro.py"):
            _run([sys.executable, str(SCRIPT_DIR / script), *launcher, *overrides])

    _run([sys.executable, str(SCRIPT_DIR / "aggregate_results.py")])


if __name__ == "__main__":
    main()
