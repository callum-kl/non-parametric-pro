"""
Print val NLPD per alpha (best marked with *) per split for fit_pro_alpha_cv.py's results.

Usage:
    python print_best_alpha.py           # all datasets with alpha-CV results
    python print_best_alpha.py <dataset> # just one dataset
"""

import argparse
import json
from typing import NamedTuple

import numpy as np
from aggregate_results import RESULTS_ROOT


class AlphaCVRun(NamedTuple):
    c_grid: list[str]
    alpha_grid: list[float]
    val_nlpd: list[float]
    best_alpha: float | None


def _c_labels(data: dict) -> list[str]:
    if "c_grid_spec" in data:
        return [str(c) for c in data["c_grid_spec"]]
    src = data.get("c_grid_requested", data["c_grid"])
    return [f"{c:.4g}" for c in src]


def _label_sort_key(label: str) -> tuple[int, float, str]:
    try:
        return (0, float(label), "")
    except ValueError:
        return (1, 0.0, label)


def collect_alpha_nlpd(dataset: str | None = None):
    # {dataset: {method: {split: AlphaCVRun}}}
    records: dict[str, dict[str, dict[int, AlphaCVRun]]] = {}

    for ds_dir in sorted(RESULTS_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        if dataset is not None and ds_dir.name != dataset:
            continue

        for split_dir in sorted(ds_dir.iterdir()):
            if not split_dir.is_dir() or not split_dir.name.startswith("split_"):
                continue
            split = int(split_dir.name.removeprefix("split_"))

            for method_dir in sorted(split_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                path = method_dir / "pro_metrics.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                # Identify alpha-CV runs by the keys fit_pro_alpha_cv.py writes rather
                # than by directory name, which renames have repeatedly invalidated.
                if "alpha_grid" not in data or "c_val_nlpd" not in data:
                    continue
                records.setdefault(ds_dir.name, {}).setdefault(method_dir.name, {})[
                    split
                ] = AlphaCVRun(
                    # Column key is the symbolic c spec, not alpha or a resolved c:
                    # alpha = c/sqrt(n) and n varies by split, and `sqrt_n` resolves to a
                    # different number per split, so neither would line up.
                    c_grid=_c_labels(data),
                    alpha_grid=data["alpha_grid"],
                    val_nlpd=data["c_val_nlpd"],
                    best_alpha=data.get("best_alpha"),
                )

    return records


def print_table(records: dict[str, dict[str, dict[int, AlphaCVRun]]]):
    for dataset in sorted(records):
        for method in sorted(records[dataset]):
            runs = records[dataset][method]
            splits = sorted(runs)

            cs = sorted(
                {c for split in splits for c in runs[split].c_grid},
                key=_label_sort_key,
            )

            # alpha differs slightly per split (n varies); show the mean as a guide.
            alpha_by_c: dict[str, list[float]] = {c: [] for c in cs}
            for split in splits:
                run = runs[split]
                for c, a in zip(run.c_grid, run.alpha_grid, strict=True):
                    alpha_by_c[c].append(a)

            col_width = 12
            print(f"\n{'─' * 72}")
            print(f"  {dataset} — {method} — val NLPD by c (* = best)")
            print(f"{'─' * 72}")
            header = f"{'c':<10}" + "".join(f"{c:>{col_width}}" for c in cs)
            alpha_row = f"{'alpha≈':<10}" + "".join(
                f"{np.mean(alpha_by_c[c]):>{col_width}.4g}" for c in cs
            )
            print(header)
            print(alpha_row)
            print("─" * len(header))
            for split in splits:
                run = runs[split]
                nlpd_by_c = dict(zip(run.c_grid, run.val_nlpd, strict=True))
                best_c = None
                if run.best_alpha is not None:
                    for c, a in zip(run.c_grid, run.alpha_grid, strict=True):
                        if a == run.best_alpha:
                            best_c = c
                row = f"{f'split {split}':<10}"
                for c in cs:
                    value = nlpd_by_c.get(c)
                    if value is None:
                        cell = "—"
                    else:
                        cell = f"{value:.4f}{'*' if c == best_c else ''}"
                    row += f"{cell:>{col_width}}"
                print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Optional dataset name to filter to (default: all datasets with alpha-CV results)",
    )
    args = parser.parse_args()

    records = collect_alpha_nlpd(args.dataset)
    if not records:
        scope = f"dataset {args.dataset!r}" if args.dataset else "any dataset"
        raise SystemExit(f"No alpha-CV results found for {scope} under {RESULTS_ROOT}")

    print_table(records)
