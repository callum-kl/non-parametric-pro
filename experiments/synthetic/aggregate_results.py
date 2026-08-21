"""Aggregate per-run metrics across synthetic datasets, their distinguishing parameter
values, and algorithms; print a summary table per dataset, plus a paired per-instance
NLPD-difference table against a baseline algorithm.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(__file__).parent / "results"

BASELINE_ALGORITHM = "standard_gp"
Z_95 = 1.96
REGION_LABELS = ("region", "background")


def collect():
    records: dict[str, dict[object, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
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
                    # `param_name` is duplicated here (alongside the source-level
                    # `param_names` dict) because a source can have been swept over
                    # more than one distinguishing parameter across its history (e.g.
                    # an old `amplitude_frac` sweep and a newer `n` sweep both sitting
                    # under `results/heteroskedastic/`) -- `param_names[source]` can
                    # only hold one name (whichever `param_dir` sorts last), so
                    # `save_csv` needs each row's *own* name to label it correctly
                    # rather than mislabelling every row with that single name.
                    "param_name": metrics["param_name"],
                    "nlpd_mean": metrics["nlpd_mean"],
                    "nlpd_std": metrics["nlpd_std"],
                    "nlpds": metrics.get("nlpds"),
                    "region_nlpds": metrics.get("region_nlpds"),
                    "seed": metrics.get("seed"),
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
        print(f"  {source}  (mean±95% CI)")
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
                    ci95 = _nlpd_ci95(entry)
                    cell = f"{mean:.4f}±{ci95:.4f}" if ci95 is not None else f"{mean:.4f}±?"
                    row += f"{cell:>{col_width}}"
            print(row)


def save_csv(records, param_names, path: Path):
    import csv

    fieldnames = [
        "source", "param_name", "param_value", "algorithm",
        "nlpd_mean", "nlpd_std", "nlpd_ci95",
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
                            # Each row's own param_name (see `collect`), not the
                            # source-level `param_names[source]` -- a source can have
                            # rows from more than one swept parameter.
                            "param_name": entry["param_name"],
                            "param_value": param_value,
                            "algorithm": algorithm,
                            "nlpd_mean": entry["nlpd_mean"],
                            "nlpd_std": entry["nlpd_std"],
                            "nlpd_ci95": _nlpd_ci95(entry),
                        }
                    )

    print(f"Saved to {path}")


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _nlpd_ci95(entry: dict) -> float | None:
    num_instances = entry.get("num_instances")
    if not num_instances:
        return None
    std = _sample_std(entry["nlpds"]) if entry.get("nlpds") else entry["nlpd_std"]
    return Z_95 * std / np.sqrt(num_instances)


def _paired_stats(entry_values: list[float], base_values: list[float]) -> dict | None:
    diffs = [
        a - b
        for a, b in zip(entry_values, base_values, strict=True)
        if not (math.isnan(a) or math.isnan(b))
    ]
    n = len(diffs)
    if n == 0:
        return None
    sem = _sample_std(diffs) / np.sqrt(n)
    return {"diff_mean": float(np.mean(diffs)), "diff_sem": sem, "diff_ci95": Z_95 * sem, "n": n}


def compute_paired_diffs(records, baseline=BASELINE_ALGORITHM):
    diffs: dict[str, dict[object, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for source, by_param in records.items():
        for param_value, by_algo in by_param.items():
            base = by_algo.get(baseline)
            if base is None or base.get("nlpds") is None:
                continue
            for algorithm, entry in by_algo.items():
                if algorithm == baseline or entry.get("nlpds") is None:
                    continue
                if (
                    entry["seed"] != base["seed"]
                    or entry["num_instances"] != base["num_instances"]
                    or len(entry["nlpds"]) != len(base["nlpds"])
                ):
                    continue
                stats = _paired_stats(entry["nlpds"], base["nlpds"])
                if stats is not None:
                    diffs[source][param_value][algorithm] = stats

    return diffs


def compute_paired_region_diffs(records, region: str, baseline=BASELINE_ALGORITHM):
    diffs: dict[str, dict[object, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for source, by_param in records.items():
        for param_value, by_algo in by_param.items():
            base = by_algo.get(baseline)
            base_values = (base or {}).get("region_nlpds") or {}
            base_values = base_values.get(region)
            if base_values is None:
                continue
            for algorithm, entry in by_algo.items():
                if algorithm == baseline:
                    continue
                entry_values = (entry.get("region_nlpds") or {}).get(region)
                if entry_values is None:
                    continue
                if (
                    entry["seed"] != base["seed"]
                    or entry["num_instances"] != base["num_instances"]
                    or len(entry_values) != len(base_values)
                ):
                    continue
                stats = _paired_stats(entry_values, base_values)
                if stats is not None:
                    diffs[source][param_value][algorithm] = stats

    return diffs


def print_diff_tables(diffs, param_names, baseline=BASELINE_ALGORITHM, label="Δ NLPD"):
    for source in sorted(diffs):
        param_name = param_names[source]
        by_param = diffs[source]
        algorithms = sorted({a for algos in by_param.values() for a in algos})
        if not algorithms:
            continue

        col_width = max(24, max((len(a) for a in algorithms), default=0) + 1)
        header = f"{param_name:<16}" + "".join(f"{a:>{col_width}}" for a in algorithms)

        print(f"\n{'─' * len(header)}")
        print(f"  {source}  ({label} vs {baseline}, paired per instance, mean±95% CI)")
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
                    mean, ci95, n = entry["diff_mean"], entry["diff_ci95"], entry["n"]
                    row += f"{f'{mean:+.4f}±{ci95:.4f} (n={n})':>{col_width}}"
            print(row)


def save_diff_csv(diffs_by_region: dict[str, dict], param_names, path: Path, baseline=BASELINE_ALGORITHM):
    import csv

    fieldnames = [
        "source", "param_name", "param_value", "region", "baseline", "algorithm",
        "diff_mean", "diff_sem", "diff_ci95", "n",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for region_label, diffs in diffs_by_region.items():
            for source in sorted(diffs):
                param_name = param_names[source]
                for param_value in sorted(diffs[source]):
                    for algorithm, entry in sorted(diffs[source][param_value].items()):
                        writer.writerow(
                            {
                                "source": source,
                                "param_name": param_name,
                                "param_value": param_value,
                                "region": region_label,
                                "baseline": baseline,
                                "algorithm": algorithm,
                                "diff_mean": entry["diff_mean"],
                                "diff_sem": entry["diff_sem"],
                                "diff_ci95": entry["diff_ci95"],
                                "n": entry["n"],
                            }
                        )

    print(f"Saved to {path}")


def region_summary(records):
    summary: dict[str, dict] = {
        label: defaultdict(lambda: defaultdict(dict)) for label in REGION_LABELS
    }

    for source, by_param in records.items():
        for param_value, by_algo in by_param.items():
            for algorithm, entry in by_algo.items():
                region_nlpds = entry.get("region_nlpds")
                if region_nlpds is None:
                    continue
                for label in REGION_LABELS:
                    values = [v for v in region_nlpds.get(label, []) if not math.isnan(v)]
                    n = len(values)
                    if n == 0:
                        continue
                    sem = _sample_std(values) / np.sqrt(n)
                    summary[label][source][param_value][algorithm] = {
                        "mean": float(np.mean(values)),
                        "ci95": Z_95 * sem,
                        "n": n,
                    }

    return summary


def print_region_summary_tables(summary, param_names):
    titles = {"region": "misspecified region", "background": "background (well-specified)"}
    for label in REGION_LABELS:
        by_source = summary[label]
        for source in sorted(by_source):
            param_name = param_names[source]
            by_param = by_source[source]
            algorithms = sorted({a for algos in by_param.values() for a in algos})
            if not algorithms:
                continue

            col_width = max(24, max((len(a) for a in algorithms), default=0) + 1)
            header = f"{param_name:<16}" + "".join(f"{a:>{col_width}}" for a in algorithms)

            print(f"\n{'─' * len(header)}")
            print(f"  {source}  ({titles[label]}, mean±95% CI)")
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
                        mean, ci95, n = entry["mean"], entry["ci95"], entry["n"]
                        row += f"{f'{mean:.4f}±{ci95:.4f} (n={n})':>{col_width}}"
                print(row)


def save_region_summary_csv(summary, param_names, path: Path):
    import csv

    fieldnames = [
        "source", "param_name", "param_value", "region", "algorithm",
        "nlpd_mean", "nlpd_ci95", "n",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in REGION_LABELS:
            by_source = summary[label]
            for source in sorted(by_source):
                param_name = param_names[source]
                for param_value in sorted(by_source[source]):
                    for algorithm, entry in sorted(by_source[source][param_value].items()):
                        writer.writerow(
                            {
                                "source": source,
                                "param_name": param_name,
                                "param_value": param_value,
                                "region": label,
                                "algorithm": algorithm,
                                "nlpd_mean": entry["mean"],
                                "nlpd_ci95": entry["ci95"],
                                "n": entry["n"],
                            }
                        )

    print(f"Saved to {path}")


if __name__ == "__main__":
    records, param_names = collect()
    if not records:
        print(f"No results found under {RESULTS_ROOT}")
    else:
        print_tables(records, param_names)
        save_csv(records, param_names, RESULTS_ROOT / "summary.csv")

        summary = region_summary(records)
        if any(summary[label] for label in REGION_LABELS):
            print_region_summary_tables(summary, param_names)
            save_region_summary_csv(summary, param_names, RESULTS_ROOT / "region_summary.csv")

        diffs_by_region = {"overall": compute_paired_diffs(records)}
        for label in REGION_LABELS:
            diffs_by_region[label] = compute_paired_region_diffs(records, label)

        labels = {"overall": "Δ NLPD", "region": "Δ NLPD (region)", "background": "Δ NLPD (background)"}
        if any(diffs_by_region.values()):
            for key, diffs in diffs_by_region.items():
                if diffs:
                    print_diff_tables(diffs, param_names, label=labels[key])
            save_diff_csv(diffs_by_region, param_names, RESULTS_ROOT / "paired_diff.csv")
