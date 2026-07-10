"""Aggregate per-split metrics across datasets and methods, print a summary table."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(__file__).parent / "results"

# (filename, {json_key -> normalised_key})
# Each method's own fitted noise level is normalised to a single "sigma" key -- for
# pro_gp/inducing_pro_gp that's `pro_sigma` (PRO's own, possibly-adapted sigma), not the
# `gp_sigma` also logged in pro_metrics.json for reference (that's the seeding exact_gp/vgp
# run's sigma, already reported under its own method row).
METHOD_METRICS = {
    "exact_gp": (
        "gp_metrics.json",
        {"gp_nlpd": "nlpd", "gp_crps": "crps", "gp_sigma": "sigma"},
    ),
    "vgp": (
        "gp_metrics.json",
        {"vgp_nlpd": "nlpd", "vgp_crps": "crps", "gp_sigma": "sigma"},
    ),
    "vgp_noncollapsed": (
        "gp_metrics.json",
        {"vgp_nlpd": "nlpd", "vgp_crps": "crps", "gp_sigma": "sigma"},
    ),
    "pro_gp": (
        "pro_metrics.json",
        {"pro_nlpd": "nlpd", "pro_crps": "crps", "pro_sigma": "sigma"},
    ),
    "inducing_pro_gp": (
        "pro_metrics.json",
        {"pro_nlpd": "nlpd", "pro_crps": "crps", "pro_sigma": "sigma"},
    ),
}

SUMMARY_METRICS = ("nlpd", "crps", "sigma")


def collect():
    # {dataset: {method: {metric: [values across splits]}}}
    records: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for ds_dir in sorted(RESULTS_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        for split_dir in sorted(ds_dir.iterdir()):
            if not split_dir.is_dir() or not split_dir.name.startswith("split_"):
                continue
            for method, (fname, key_map) in METHOD_METRICS.items():
                path = split_dir / method / fname
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                for json_key, norm_key in key_map.items():
                    if json_key in data:
                        records[dataset][method][norm_key].append(data[json_key])

    return records


def summarise(records):
    # {dataset: {method: {metric: (mean, std)}}}
    summary = {}
    for dataset, methods in records.items():
        summary[dataset] = {}
        for method, metrics in methods.items():
            summary[dataset][method] = {
                k: (float(np.mean(v)), float(np.std(v)))
                for k, v in metrics.items()
            }
    return summary


def print_table(summary):
    datasets = sorted(summary)
    methods = sorted({m for ds in summary.values() for m in ds})

    for metric in SUMMARY_METRICS:
        print(f"\n{'─'*72}")
        print(f"  {metric.upper()}")
        print(f"{'─'*72}")
        col_width = max(16, max((len(m) for m in methods), default=0) + 1)
        header = f"{'dataset':<20}" + "".join(f"{m:>{col_width}}" for m in methods)
        print(header)
        print("─" * len(header))
        for ds in datasets:
            row = f"{ds:<20}"
            for method in methods:
                entry = summary[ds].get(method, {}).get(metric)
                if entry is None:
                    row += f"{'—':>{col_width}}"
                else:
                    mean, std = entry
                    row += f"{f'{mean:.4f}±{std:.4f}':>{col_width}}"
            print(row)


def save_csv(summary, path: Path):
    import csv

    datasets = sorted(summary)
    methods = sorted({m for ds in summary.values() for m in ds})

    fieldnames = ["dataset"] + [
        f"{method}_{metric}_{stat}"
        for method in methods
        for metric in SUMMARY_METRICS
        for stat in ("mean", "std")
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ds in datasets:
            row = {"dataset": ds}
            for method in methods:
                for metric in SUMMARY_METRICS:
                    entry = summary[ds].get(method, {}).get(metric)
                    mean_key = f"{method}_{metric}_mean"
                    std_key = f"{method}_{metric}_std"
                    if entry is not None:
                        row[mean_key], row[std_key] = entry
                    else:
                        row[mean_key] = row[std_key] = ""
            writer.writerow(row)

    print(f"Saved to {path}")


if __name__ == "__main__":
    records = collect()
    summary = summarise(records)
    print_table(summary)
    save_csv(summary, RESULTS_ROOT / "summary.csv")
