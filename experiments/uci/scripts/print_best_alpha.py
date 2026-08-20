"""Print val NLPD per alpha (best marked with *) per split for fit_pro_alpha_cv.py's results.

Usage:
    python print_best_alpha.py           # all datasets with alpha-CV results
    python print_best_alpha.py <dataset> # just one dataset
"""

import argparse
import json
from typing import NamedTuple

from aggregate_results import RESULTS_ROOT, _resolve_method

_ALPHA_CV_BASES = ("pro_gp_alpha_cv", "inducing_pro_gp_alpha_cv")


class AlphaCVRun(NamedTuple):
    alpha_grid: list[float]
    alpha_val_nlpd: list[float]
    best_alpha: float | None


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
                resolved = _resolve_method(method_dir.name)
                if resolved is None or resolved[0] not in _ALPHA_CV_BASES:
                    continue
                path = method_dir / "pro_metrics.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                if "alpha_grid" not in data or "alpha_val_nlpd" not in data:
                    continue
                records.setdefault(ds_dir.name, {}).setdefault(method_dir.name, {})[
                    split
                ] = AlphaCVRun(
                    alpha_grid=data["alpha_grid"],
                    alpha_val_nlpd=data["alpha_val_nlpd"],
                    best_alpha=data.get("best_alpha"),
                )

    return records


def print_table(records: dict[str, dict[str, dict[int, AlphaCVRun]]]):
    for dataset in sorted(records):
        for method in sorted(records[dataset]):
            runs = records[dataset][method]
            splits = sorted(runs)

            # Union of alpha values seen across splits, in first-seen order -- normally
            # every split shares the same alpha_grid, but this stays correct even if not.
            alphas: list[float] = []
            for split in splits:
                for a in runs[split].alpha_grid:
                    if a not in alphas:
                        alphas.append(a)

            col_width = max(12, 10)
            print(f"\n{'─' * 72}")
            print(f"  {dataset} — {method} — val NLPD by alpha (* = best)")
            print(f"{'─' * 72}")
            header = f"{'split':<10}" + "".join(f"{a:>{col_width}.4g}" for a in alphas)
            print(header)
            print("─" * len(header))
            for split in splits:
                run = runs[split]
                nlpd_by_alpha = dict(zip(run.alpha_grid, run.alpha_val_nlpd, strict=True))
                row = f"{split:<10}"
                for a in alphas:
                    value = nlpd_by_alpha.get(a)
                    if value is None:
                        cell = "—"
                    else:
                        marker = "*" if run.best_alpha is not None and a == run.best_alpha else ""
                        cell = f"{value:.4f}{marker}"
                    row += f"{cell:>{col_width}}"
                print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", nargs="?", default=None,
        help="Optional dataset name to filter to (default: all datasets with alpha-CV results)",
    )
    args = parser.parse_args()

    records = collect_alpha_nlpd(args.dataset)
    if not records:
        scope = f"dataset {args.dataset!r}" if args.dataset else "any dataset"
        raise SystemExit(f"No alpha-CV results found for {scope} under {RESULTS_ROOT}")

    print_table(records)
