"""Print per-split metrics for a single dataset (companion to aggregate_results.py).

Usage:
    python print_split_results.py <dataset>
"""

import argparse
import json

from aggregate_results import RESULTS_ROOT, SUMMARY_METRICS, _resolve_method


def collect_dataset(dataset: str):
    # {split: {method: {metric: value}}}
    records: dict[int, dict[str, dict[str, float]]] = {}

    ds_dir = RESULTS_ROOT / dataset
    if not ds_dir.is_dir():
        return records

    for split_dir in ds_dir.iterdir():
        if not split_dir.is_dir() or not split_dir.name.startswith("split_"):
            continue
        split = int(split_dir.name.removeprefix("split_"))

        for method_dir in sorted(split_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            resolved = _resolve_method(method_dir.name)
            if resolved is None:
                continue
            _, fname, key_map = resolved
            path = method_dir / fname
            if not path.exists():
                continue

            data = json.loads(path.read_text())
            metrics = {
                norm_key: data[json_key]
                for json_key, norm_key in key_map.items()
                if json_key in data
            }
            records.setdefault(split, {})[method_dir.name] = metrics

    return records


def print_table(dataset: str, records: dict[int, dict[str, dict[str, float]]]):
    splits = sorted(records)
    methods = sorted({m for r in records.values() for m in r})

    for metric in SUMMARY_METRICS:
        print(f"\n{'─' * 72}")
        print(f"  {dataset} — {metric.upper()}")
        print(f"{'─' * 72}")
        col_width = max(16, max((len(m) for m in methods), default=0) + 1)
        header = f"{'split':<10}" + "".join(f"{m:>{col_width}}" for m in methods)
        print(header)
        print("─" * len(header))
        for split in splits:
            row = f"{split:<10}"
            for method in methods:
                value = records[split].get(method, {}).get(metric)
                row += f"{'—' if value is None else f'{value:.4f}':>{col_width}}"
            print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset name, e.g. 'servo' (matches results/<dataset>/)")
    args = parser.parse_args()

    records = collect_dataset(args.dataset)
    if not records:
        raise SystemExit(f"No results found for dataset '{args.dataset}' under {RESULTS_ROOT}")

    print_table(args.dataset, records)
