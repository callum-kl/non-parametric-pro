"""
Run the full UCI benchmark, then aggregate into results/summary.csv.

Exact datasets:     fit_exact_gp.py -> fit_pro.py                          (exact_gp, pro_gp_gibbs)
Inducing datasets:  fit_vgp.py -> fit_ppgpr.py -> fit_pro.py x2            (vgp_noncollapsed, ppgpr,
                    inducing_pro_gp_gibbs, inducing_pro_gp_gibbs_ppgpr)

Per-dataset settings live in conf/ds/<dataset>.yaml.

Usage:
    python experiments/uci/scripts/run_uci.py
    python experiments/uci/scripts/run_uci.py --datasets servo,wine --splits 1,2 --n-jobs 4
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

EXACT_DATASETS = ["servo", "machine", "autompg", "housing", "stock", "concrete", "solar"]
INDUCING_DATASETS = [
    "airfoil",
    "wine",
    "skillcraft",
    "abalone",
    "whitewine",
    "parkinsons",
    "airquality",
    "elevators",
    "protein",
]


def _run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets", default=",".join(EXACT_DATASETS + INDUCING_DATASETS)
    )
    parser.add_argument("--splits", default="1,2,3,4,5")
    parser.add_argument("--n-jobs", default="-1")
    args = parser.parse_args()

    launcher = ["-m", "hydra/launcher=joblib", f"hydra.launcher.n_jobs={args.n_jobs}"]

    def script(name: str, dataset: str, *overrides: str) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT_DIR / name),
            *launcher,
            f"ds@_global_={dataset}",
            f"split={args.splits}",
            *overrides,
        ]

    for dataset in args.datasets.split(","):
        if dataset in EXACT_DATASETS:
            _run(script("fit_exact_gp.py", dataset))
            _run(script("fit_pro.py", dataset))
        elif dataset in INDUCING_DATASETS:
            _run(script("fit_vgp.py", dataset))
            _run(script("fit_ppgpr.py", dataset))
            _run(script("fit_pro.py", dataset, "vgp_variant=noncollapsed", "name=gibbs"))
            _run(script("fit_pro.py", dataset, "vgp_variant=ppgpr", "name=gibbs_ppgpr"))
        else:
            raise ValueError(f"Unknown dataset {dataset!r}")

    _run([sys.executable, str(SCRIPT_DIR / "aggregate_results.py")])


if __name__ == "__main__":
    main()
