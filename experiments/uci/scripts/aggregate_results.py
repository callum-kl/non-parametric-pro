"""Aggregate per-split metrics across datasets and methods, print a summary table."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

METHOD_METRICS = {
    "exact_gp": (
        "gp_metrics.json",
        {"gp_nlpd": "nlpd", "gp_sigma": "sigma"},
    ),
    "vgp": (
        "gp_metrics.json",
        {"vgp_nlpd": "nlpd", "gp_sigma": "sigma"},
    ),
    "vgp_noncollapsed": (
        "gp_metrics.json",
        {"vgp_nlpd": "nlpd", "gp_sigma": "sigma"},
    ),
    "ppgpr": (
        "gp_metrics.json",
        {"ppgpr_nlpd": "nlpd", "gp_sigma": "sigma"},
    ),
    "pro_gp": (
        "pro_metrics.json",
        {"pro_nlpd": "nlpd", "pro_sigma": "sigma"},
    ),
    "inducing_pro_gp": (
        "pro_metrics.json",
        {"pro_nlpd": "nlpd", "pro_sigma": "sigma"},
    ),
    "pro_gp_cv": (
        "pro_metrics.json",
        {"pro_nlpd": "nlpd", "pro_sigma": "sigma"},
    ),
    "inducing_pro_gp_cv": (
        "pro_metrics.json",
        {"pro_nlpd": "nlpd", "pro_sigma": "sigma"},
    ),
    "pro_gp_alpha_cv": (
        "pro_metrics.json",
        {"pro_nlpd": "nlpd", "pro_sigma": "sigma"},
    ),
    "inducing_pro_gp_alpha_cv": (
        "pro_metrics.json",
        {"pro_nlpd": "nlpd", "pro_sigma": "sigma"},
    ),
}

SUMMARY_METRICS = ("nlpd", "sigma")
_BASES_BY_LENGTH = sorted(METHOD_METRICS, key=len, reverse=True)


def _resolve_method(dirname: str) -> tuple[str, str, dict[str, str]] | None:
    """Match a results subdir name against a known base method, allowing a `_<suffix>`.

    Returns ``(base, fname, key_map)`` for an exact base match or a `<base>_<suffix>`
    match, or ``None`` if the dir doesn't correspond to a known method.
    """
    if dirname in METHOD_METRICS:
        fname, key_map = METHOD_METRICS[dirname]
        return dirname, fname, key_map
    for base in _BASES_BY_LENGTH:
        if dirname.startswith(base + "_"):
            fname, key_map = METHOD_METRICS[base]
            return base, fname, key_map
    return None


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
                # Report under the actual dir name (e.g. "pro_gp_untuned"), not the base,
                # so suffixed runs get their own column instead of merging into the default.
                method = method_dir.name
                for json_key, norm_key in key_map.items():
                    if json_key in data:
                        records[dataset][method][norm_key].append(data[json_key])

    return records


def _standard_error(v: list[float]) -> float:
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def summarise(records):
    summary = {}
    for dataset, methods in records.items():
        summary[dataset] = {}
        for method, metrics in methods.items():
            summary[dataset][method] = {
                k: (float(np.mean(v)), _standard_error(v))
                for k, v in metrics.items()
            }
    return summary


def print_table(summary):
    datasets = sorted(summary)
    methods = sorted({m for ds in summary.values() for m in ds})
    row_label_width = max(20, max((len(m) for m in methods), default=0) + 1)

    for metric in SUMMARY_METRICS:
        print(f"\n{'─'*72}")
        print(f"  {metric.upper()}")
        print(f"{'─'*72}")
        col_width = max(16, max((len(ds) for ds in datasets), default=0) + 1)
        header = f"{'model':<{row_label_width}}" + "".join(f"{ds:>{col_width}}" for ds in datasets)
        print(header)
        print("─" * len(header))
        for method in methods:
            row = f"{method:<{row_label_width}}"
            for ds in datasets:
                entry = summary[ds].get(method, {}).get(metric)
                if entry is None:
                    row += f"{'—':>{col_width}}"
                else:
                    mean, se = entry
                    row += f"{f'{mean:.4f}±{se:.4f}':>{col_width}}"
            print(row)


def save_csv(summary, path: Path):
    import csv

    datasets = sorted(summary)
    methods = sorted({m for ds in summary.values() for m in ds})

    fieldnames = ["dataset"] + [
        f"{method}_{metric}_{stat}"
        for method in methods
        for metric in SUMMARY_METRICS
        for stat in ("mean", "se")
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
                    se_key = f"{method}_{metric}_se"
                    if entry is not None:
                        row[mean_key], row[se_key] = entry
                    else:
                        row[mean_key] = row[se_key] = ""
            writer.writerow(row)

    print(f"Saved to {path}")


if __name__ == "__main__":
    records = collect()
    summary = summarise(records)
    print_table(summary)
    save_csv(summary, RESULTS_ROOT / "summary.csv")
