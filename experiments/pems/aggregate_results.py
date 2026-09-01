"""Aggregate exact-GP vs PRO metrics across splits, print a summary table."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(__file__).parent / "results"

_PRO_KEYS = {
    "nlpd": "pro_nlpd",
    "diag_nll": "pro_diag_nll",
    "full_nll": "pro_full_nll",
    "diag_nll_gauss": "pro_diag_nll_gauss",
    "full_nll_gauss": "pro_full_nll_gauss",
    "energy_score": "pro_energy_score",
    "rmse": "pro_rmse",
    "sigma": "pro_sigma",
}

# method dir name -> (metrics filename, {common metric name: key in that file})
METHOD_METRICS = {
    "exact_gp": (
        "gp_metrics.json",
        {
            "nlpd": "gp_nlpd",
            "diag_nll": "gp_diag_nll",
            "full_nll": "gp_full_nll",
            "energy_score": "gp_energy_score",
            "rmse": "gp_rmse",
            "sigma": "gp_sigma",
        },
    ),
    "pro_gp": ("pro_metrics.json", _PRO_KEYS),
    "pro_gp_gibbs": ("pro_metrics.json", _PRO_KEYS),
}


def collect():
    records: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_split: dict[str, dict[int, dict]] = defaultdict(dict)

    if not RESULTS_ROOT.is_dir():
        return records, per_split

    for split_dir in sorted(RESULTS_ROOT.iterdir()):
        if not split_dir.is_dir() or not split_dir.name.startswith("split_"):
            continue
        split = int(split_dir.name.removeprefix("split_"))
        for method, (filename, metric_keys) in METHOD_METRICS.items():
            path = split_dir / method / filename
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            values = {}
            for metric, key in metric_keys.items():
                if key in data:
                    records[method][metric].append(data[key])
                    values[metric] = data[key]
            per_split[method][split] = values

    return records, per_split


def summarise(records):
    summary = {}
    for method, metrics in records.items():
        summary[method] = {
            metric: {
                "mean": float(np.mean(values)),
                "se": float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0,
                "n_splits": len(values),
            }
            for metric, values in metrics.items()
        }
    return summary


def print_table(summary):
    print(f"\n{'=' * 60}\n  PEMS graph GP: exact GP vs PRO\n{'=' * 60}")
    for method, metrics in sorted(summary.items()):
        print(f"\n  method: {method}")
        for metric, stats in metrics.items():
            print(
                f"    {metric:>6}: mean={stats['mean']:.4f}  se={stats['se']:.4f}  "
                f"(n={stats['n_splits']} splits)"
            )


def save_csv(per_split, path: Path):
    rows = []
    for method, splits in per_split.items():
        for split, values in splits.items():
            rows.append({"method": method, "split": split, **values})

    if not rows:
        print("No results found to save.")
        return

    fieldnames = [
        "method",
        "split",
        "nlpd",
        "diag_nll",
        "full_nll",
        "diag_nll_gauss",
        "full_nll_gauss",
        "energy_score",
        "rmse",
        "sigma",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-split results to {path}")


if __name__ == "__main__":
    records, per_split_records = collect()
    summary = summarise(records)
    print_table(summary)
    save_csv(per_split_records, RESULTS_ROOT / "summary.csv")
