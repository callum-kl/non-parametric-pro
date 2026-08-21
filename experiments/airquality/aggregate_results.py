"""Aggregate per-site SVGP metrics across sites, print a summary table."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(__file__).parent / "results"
METHODS = ("svgp", "svgp_no_outliers")
METRICS = ("rmse", "nlpd", "crps")


def collect():
    records: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    per_site: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    if not RESULTS_ROOT.is_dir():
        return records, per_site

    for mode_dir in sorted(RESULTS_ROOT.iterdir()):
        if not mode_dir.is_dir():
            continue
        mode = mode_dir.name
        for site_dir in sorted(mode_dir.iterdir()):
            if not site_dir.is_dir():
                continue
            site_id = site_dir.name
            for method in METHODS:
                path = site_dir / method / "svgp_metrics.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                for metric in METRICS:
                    if metric in data:
                        records[mode][method][metric].append(data[metric])
                per_site[mode][method][site_id] = data

    return records, per_site


def summarise(records):
    summary = {}
    for mode, methods in records.items():
        summary[mode] = {}
        for method, metrics in methods.items():
            summary[mode][method] = {
                metric: {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_sites": len(values),
                }
                for metric, values in metrics.items()
            }
    return summary


def print_table(summary):
    for mode, methods in sorted(summary.items()):
        print(f"\n{'=' * 72}\n  MODE: {mode}\n{'=' * 72}")
        for method, metrics in sorted(methods.items()):
            print(f"\n  method: {method}")
            for metric, stats in metrics.items():
                print(
                    f"    {metric:>6}: mean={stats['mean']:.4f}  "
                    f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
                    f"(n={stats['n_sites']} sites)"
                )


def save_csv(per_site, path: Path):
    rows = []
    for mode, methods in per_site.items():
        for method, sites in methods.items():
            for site_id, data in sites.items():
                row = {"mode": mode, "method": method, "site_id": site_id}
                row.update({k: data.get(k, "") for k in METRICS})
                rows.append(row)

    if not rows:
        print("No results found to save.")
        return

    fieldnames = ["mode", "method", "site_id", *METRICS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-site results to {path}")


if __name__ == "__main__":
    records, per_site_records = collect()
    summary = summarise(records)
    print_table(summary)
    save_csv(per_site_records, RESULTS_ROOT / "summary_per_site.csv")
