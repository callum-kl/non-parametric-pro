"""Aggregate per-split test NLPD across datasets and methods into results/summary.csv."""

import csv
import json
from pathlib import Path

import numpy as np
from run_uci import EXACT_DATASETS, INDUCING_DATASETS

from non_parametric_pro.data.uci.uci import UCI_REGRESSION_DATASET_SIZES

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

DATASETS = sorted(
    EXACT_DATASETS + INDUCING_DATASETS, key=lambda ds: UCI_REGRESSION_DATASET_SIZES[ds][0]
)

# method dir -> (metrics file, NLPD key)
METHODS = {
    "exact_gp": ("gp_metrics.json", "gp_nlpd"),
    "vgp_noncollapsed": ("gp_metrics.json", "vgp_nlpd"),
    "ppgpr": ("gp_metrics.json", "ppgpr_nlpd"),
    "pro_gp_gibbs": ("pro_metrics.json", "pro_nlpd"),
    "inducing_pro_gp_gibbs": ("pro_metrics.json", "pro_nlpd"),
    "inducing_pro_gp_gibbs_ppgpr": ("pro_metrics.json", "pro_nlpd"),
    **{
        f"{prefix}_{variant}": ("mixture_metrics.json", "mixture_nlpd")
        for prefix in ("omgp", "omgp_shared")
        for variant in ("k1", "k2", "k3", "k5", "sparse", "valk")
    },
}


def collect() -> dict[str, dict[str, list[float]]]:
    records = {ds: {m: [] for m in METHODS} for ds in DATASETS}
    for ds in DATASETS:
        for split_dir in sorted((RESULTS_ROOT / ds).glob("split_*")):
            for method, (fname, key) in METHODS.items():
                path = split_dir / method / fname
                if path.exists():
                    records[ds][method].append(json.loads(path.read_text())[key])
    return records


def _mean_se(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    se = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(np.mean(values)), float(se)


def main() -> None:
    records = collect()
    fieldnames = ["n", "dataset"] + [
        f"{m}_nlpd_{stat}" for m in METHODS for stat in ("mean", "se")
    ]
    path = RESULTS_ROOT / "summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        print(f"{'NLPD':<12}" + "".join(f"{m:>28}" for m in METHODS))
        for ds in DATASETS:
            row = {"n": UCI_REGRESSION_DATASET_SIZES[ds][0], "dataset": ds}
            line = f"{ds:<12}"
            for method in METHODS:
                entry = _mean_se(records[ds][method])
                mean, se = entry or ("", "")
                row[f"{method}_nlpd_mean"], row[f"{method}_nlpd_se"] = mean, se
                line += f"{f'{mean:.3f}±{se:.3f}' if entry else '—':>28}"
            writer.writerow(row)
            print(line)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
