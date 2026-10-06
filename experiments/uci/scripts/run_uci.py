"""
Run the full UCI benchmark, then aggregate into results/summary.csv.

Exact datasets:     fit_exact_gp.py -> fit_pro.py                          (exact_gp, pro_gp_gibbs)
Inducing datasets:  fit_vgp.py -> fit_ppgpr.py -> fit_pro.py x2            (vgp_noncollapsed, ppgpr,
                    inducing_pro_gp_gibbs, inducing_pro_gp_gibbs_ppgpr)
--mixture:          fit_mixture.py on the datasets in MIXTURE_RUNS          (omgp_shared_k*)

Per-dataset settings live in conf/ds/<dataset>.yaml.

Usage:
    python experiments/uci/scripts/run_uci.py
    python experiments/uci/scripts/run_uci.py --datasets machine,wine --splits 1,2 --n-jobs 4
    python experiments/uci/scripts/run_uci.py --mixture
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

EXACT_DATASETS = [
    "machine",
    "autompg",
    "housing",
    "stock",
    "concrete",
    "concreteslump",
    "energy",
    "autos",
]
INDUCING_DATASETS = [
    "wine",
    "skillcraft",
    "abalone",
    "parkinsons",
    "airquality",
    "elevators",
    "protein",
    "kin40k",
]

# Peak memory per protein split is ~2.3GB, so 5 in parallel would exceed 8GB of RAM.
MAX_JOBS = {"protein": 2, "kin40k": 1}

# OMGP baselines for the mixture figure: (splits, Gibbs steps, batches of K). The batches
# are how the stored results were run, and the random keys depend on the batch, so this
# reproduces them exactly.
_SMALL_K_BATCHES = ([1, 2, 3, 5], [10], [20])
_LARGE_K_BATCHES = ([1, 2, 3, 5, 10], [20])
MIXTURE_RUNS = {
    **dict.fromkeys(
        ("concreteslump", "autos", "machine", "autompg", "housing", "stock", "energy",
         "concrete"),
        ("1,2,3,4,5,6,7,8,9,10", 1000, _SMALL_K_BATCHES),
    ),
    **dict.fromkeys(
        ("wine", "skillcraft", "abalone", "parkinsons", "airquality"),
        ("1,2,3,4,5", 1000, _SMALL_K_BATCHES),
    ),
    "kin40k": ("1,2,3,4,5", 500, _LARGE_K_BATCHES),
    "elevators": ("1,2,3", 500, _LARGE_K_BATCHES),
    "protein": ("1,2,3", 500, _LARGE_K_BATCHES),
}


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
    parser.add_argument("--splits", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--n-jobs", default="5")
    parser.add_argument(
        "--mixture",
        action="store_true",
        help="Fit only the OMGP baselines in MIXTURE_RUNS (needs the GP fits above).",
    )
    args = parser.parse_args()

    def script(name: str, dataset: str, *overrides: str) -> list[str]:
        n_jobs = int(args.n_jobs)
        if dataset in MAX_JOBS:
            cap = MAX_JOBS[dataset]
            n_jobs = cap if n_jobs < 1 else min(n_jobs, cap)
        return [
            sys.executable,
            str(SCRIPT_DIR / name),
            "-m",
            "hydra/launcher=joblib",
            f"hydra.launcher.n_jobs={n_jobs}",
            f"ds@_global_={dataset}",
            f"split={args.splits}",
            *overrides,
        ]

    if args.mixture:
        for dataset in args.datasets.split(","):
            if dataset not in MIXTURE_RUNS:
                continue
            splits, num_steps, k_batches = MIXTURE_RUNS[dataset]
            for ks in k_batches:
                _run([
                    sys.executable, str(SCRIPT_DIR / "fit_mixture.py"), "-m",
                    "hydra/launcher=joblib", f"hydra.launcher.n_jobs={args.n_jobs}",
                    f"ds@_global_={dataset}", f"split={splits}", f"num_steps={num_steps}",
                    f"ks=[{','.join(map(str, ks))}]",
                ])
        return

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
