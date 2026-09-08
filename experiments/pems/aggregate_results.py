"""Aggregate exact-GP vs PRO metrics across dataset sizes and splits, print a summary table."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(__file__).parent / "results"

# Method *directories* are matched by prefix rather than an exact registered name,
# so any name=<suffix> variant is picked up automatically: (dir name prefix,
# metrics filename, {metric: json key}, PIT key).
_METHOD_PREFIXES = [
    (
        "exact_gp",
        "gp_metrics.json",
        {"nlpd": "gp_nlpd", "variogram_score": "gp_variogram_score"},
        "gp_pit",
    ),
    (
        "pro_gp",
        "pro_metrics.json",
        {"nlpd": "pro_nlpd", "variogram_score": "pro_variogram_score"},
        "pro_pit",
    ),
]

HEADLINE_METRICS = ["nlpd", "variogram_score"]


def collect():
    records: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_split: dict[tuple[int, str], dict[int, dict]] = defaultdict(dict)
    pit_records: dict[tuple[int, str], list[float]] = defaultdict(list)

    if not RESULTS_ROOT.is_dir():
        return records, per_split, pit_records

    for num_train_dir in sorted(RESULTS_ROOT.iterdir()):
        if not num_train_dir.is_dir() or not num_train_dir.name.startswith("num_train_"):
            continue
        num_train = int(num_train_dir.name.removeprefix("num_train_"))
        for split_dir in sorted(num_train_dir.iterdir()):
            if not split_dir.is_dir() or not split_dir.name.startswith("split_"):
                continue
            split = int(split_dir.name.removeprefix("split_"))
            for method_dir in sorted(split_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                match = next(
                    (m for m in _METHOD_PREFIXES if method.startswith(m[0])), None
                )
                if match is None:
                    continue
                _, filename, metric_keys, pit_key = match
                path = method_dir / filename
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                key = (num_train, method)
                values = {}
                for metric, json_key in metric_keys.items():
                    if json_key in data:
                        records[key][metric].append(data[json_key])
                        values[metric] = data[json_key]
                per_split[key][split] = values

                if pit_key in data:
                    pit_records[key].extend(data[pit_key])

    return records, per_split, pit_records


def summarise(records):
    summary = {}
    for key, metrics in records.items():
        summary[key] = {
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


def print_table(summary, *, metrics: list[str] = HEADLINE_METRICS):
    print(f"\n{'=' * 60}\n  PEMS graph GP: exact GP vs PRO\n{'=' * 60}")
    num_trains = sorted({num_train for num_train, _ in summary})
    val_col = 16
    for num_train in num_trains:
        methods = sorted(method for nt, method in summary if nt == num_train)
        if not methods:
            continue
        print(f"\n-- num_train={num_train} --")
        method_col = max((len(m) for m in methods), default=6) + 2
        print(f"{'method':<{method_col}}" + "".join(f"{m:>{val_col}}" for m in metrics))
        for method in methods:
            row = f"{method:<{method_col}}"
            for metric in metrics:
                s = summary[(num_train, method)].get(metric)
                cell = f"{s['mean']:.3f}±{s['se']:.3f}" if s else "--"
                row += f"{cell:>{val_col}}"
            print(row)
    print("\n(mean±SE across splits; see summary.csv for per-split values)")


def save_csv(per_split, path: Path):
    rows = []
    for (num_train, method), splits in per_split.items():
        for split, values in splits.items():
            rows.append({"num_train": num_train, "method": method, "split": split, **values})

    if not rows:
        print("No results found to save.")
        return

    fieldnames = ["num_train", "method", "split", *HEADLINE_METRICS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-split results to {path}")


def print_pit_summary(pit_records, *, num_bins: int = 10):
    """One line per (num_train, method): chi-square departure from a uniform PIT
    (pooled across splits). Larger = worse marginal calibration; as a rough rule of
    thumb, values well above num_bins-1 indicate a clear departure from uniform (no
    p-value, to avoid a scipy dependency).
    """
    if not pit_records:
        return

    print(f"\n{'=' * 60}\n  PIT calibration (pooled across splits)\n{'=' * 60}")
    method_col = max((len(method) for _, method in pit_records), default=6) + 2
    for (num_train, method), values in sorted(pit_records.items()):
        arr = np.asarray(values)
        n = len(arr)
        if n == 0:
            continue
        counts, _ = np.histogram(arr, bins=num_bins, range=(0.0, 1.0))
        expected = n / num_bins
        chi2_stat = float(np.sum((counts - expected) ** 2 / expected))
        print(
            f"  num_train={num_train:<5} {method:<{method_col}} "
            f"chi2={chi2_stat:7.2f}  (df={num_bins - 1}, n={n})"
        )


if __name__ == "__main__":
    records, per_split_records, pit_records = collect()
    summary = summarise(records)
    print_table(summary)
    save_csv(per_split_records, RESULTS_ROOT / "summary.csv")
    print_pit_summary(pit_records)
