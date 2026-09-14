"""
Aggregate per-run metrics across synthetic datasets, training sizes and algorithms;
print a summary table per dataset and write `results/summary.csv`.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"



def collect():
    records: dict[str, dict[object, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    param_names: dict[str, str] = {}

    if not RESULTS_ROOT.exists():
        return records, param_names

    for source_dir in sorted(RESULTS_ROOT.iterdir()):
        if not source_dir.is_dir():
            continue
        for param_dir in sorted(source_dir.iterdir()):
            if not param_dir.is_dir():
                continue
            for algo_dir in sorted(param_dir.iterdir()):
                metrics_path = algo_dir / "metrics.json"
                if not metrics_path.exists():
                    continue
                metrics = json.loads(metrics_path.read_text())
                source = metrics["source"]
                param_names[source] = metrics["param_name"]
                records[source][metrics["param_value"]][metrics["algorithm"]] = {
                    "nlpd_mean": metrics["nlpd_mean"],
                    "nlpd_std": metrics["nlpd_std"],
                    "nlpds": metrics.get("nlpds"),
                    "num_instances": metrics.get("num_instances"),
                }

    return records, param_names


def print_tables(records, param_names):
    for source in sorted(records):
        param_name = param_names[source]
        by_param = records[source]
        algorithms = sorted({a for algos in by_param.values() for a in algos})

        col_width = max(16, max((len(a) for a in algorithms), default=0) + 1)
        header = f"{param_name:<16}" + "".join(f"{a:>{col_width}}" for a in algorithms)

        print(f"\n{'─' * len(header)}")
        print(f"  {source}  (mean±1 SE)")
        print(f"{'─' * len(header)}")
        print(header)
        print("─" * len(header))
        for param_value in sorted(by_param):
            row = f"{param_value!s:<16}"
            for algorithm in algorithms:
                entry = by_param[param_value].get(algorithm)
                if entry is None:
                    row += f"{'—':>{col_width}}"
                else:
                    mean = entry["nlpd_mean"]
                    sem = _nlpd_sem(entry)
                    cell = (
                        f"{mean:.4f}±{sem:.4f}"
                        if sem is not None
                        else f"{mean:.4f}±?"
                    )
                    row += f"{cell:>{col_width}}"
            print(row)


def save_csv(records, param_names, path: Path):
    import csv

    fieldnames = [
        "source",
        "param_name",
        "param_value",
        "algorithm",
        "nlpd_mean",
        "nlpd_std",
        "nlpd_sem",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for source in sorted(records):
            for param_value in sorted(records[source]):
                for algorithm, entry in sorted(records[source][param_value].items()):
                    writer.writerow(
                        {
                            "source": source,
                            "param_name": param_names[source],
                            "param_value": param_value,
                            "algorithm": algorithm,
                            "nlpd_mean": entry["nlpd_mean"],
                            "nlpd_std": entry["nlpd_std"],
                            "nlpd_sem": _nlpd_sem(entry),
                        }
                    )

    print(f"Saved to {path}")


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _nlpd_sem(entry: dict) -> float | None:
    """Standard error of the mean NLPD across instances."""
    num_instances = entry.get("num_instances")
    if not num_instances:
        return None
    std = _sample_std(entry["nlpds"]) if entry.get("nlpds") else entry["nlpd_std"]
    return std / np.sqrt(num_instances)


if __name__ == "__main__":
    records, param_names = collect()
    if not records:
        print(f"No results found under {RESULTS_ROOT}")
    else:
        print_tables(records, param_names)
        save_csv(records, param_names, RESULTS_ROOT / "summary.csv")
