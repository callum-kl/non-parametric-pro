"""Aggregate exact-GP vs PRO metrics across splits, print a summary table."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(__file__).parent / "results"


def _with_region_suffixes(keys: dict[str, str], prefix: str) -> dict[str, str]:
    """Add _in_region/_out_region variants of every {common_name: json_key} pair,
    plus an n_test_<region> point-count -- only present for regime-shift
    misspecification runs (see fit_exact_gp.py/fit_pro.py's _region_metrics).
    collect() only records keys actually present in a given metrics.json, so these
    are silently absent/skipped for normal (non-regime-shift) runs.
    """
    expanded = dict(keys)
    for region in ("in_region", "out_region"):
        expanded.update({f"{n}_{region}": f"{k}_{region}" for n, k in keys.items()})
        expanded[f"n_test_{region}"] = f"{prefix}_n_test_{region}"
    return expanded


_BASE_GP_KEYS = {
    "nlpd": "gp_nlpd",
    "diag_nll": "gp_diag_nll",
    "full_nll": "gp_full_nll",
    "energy_score": "gp_energy_score",
    "crps": "gp_crps",
    "variogram_score": "gp_variogram_score",
    "rmse": "gp_rmse",
}
_BASE_PRO_KEYS = {
    "nlpd": "pro_nlpd",
    "diag_nll": "pro_diag_nll",
    "full_nll": "pro_full_nll",
    "diag_nll_gauss": "pro_diag_nll_gauss",
    "full_nll_gauss": "pro_full_nll_gauss",
    "energy_score": "pro_energy_score",
    "crps": "pro_crps",
    "variogram_score": "pro_variogram_score",
    "rmse": "pro_rmse",
}

_GP_KEYS = {**_with_region_suffixes(_BASE_GP_KEYS, "gp"), "sigma": "gp_sigma"}
_PRO_KEYS = _with_region_suffixes(_BASE_PRO_KEYS, "pro")
_PRO_KEYS["sigma"] = "pro_sigma"  # a single fitted scalar -- no region variant

# Method *directories* are matched by prefix rather than an exact registered name,
# so any name=<suffix> variant (baseline, regime_shift, per-bridge-cut sweeps, ...)
# is picked up automatically: (dir name prefix, metrics filename, metric keys, PIT key).
_METHOD_PREFIXES = [
    ("exact_gp", "gp_metrics.json", _GP_KEYS, "gp_pit"),
    ("pro_gp", "pro_metrics.json", _PRO_KEYS, "pro_pit"),
]


def collect():
    records: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_split: dict[str, dict[int, dict]] = defaultdict(dict)
    pit_records: dict[str, list[float]] = defaultdict(list)

    if not RESULTS_ROOT.is_dir():
        return records, per_split, pit_records

    for split_dir in sorted(RESULTS_ROOT.iterdir()):
        if not split_dir.is_dir() or not split_dir.name.startswith("split_"):
            continue
        split = int(split_dir.name.removeprefix("split_"))
        for method_dir in sorted(split_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            match = next((m for m in _METHOD_PREFIXES if method.startswith(m[0])), None)
            if match is None:
                continue
            _, filename, metric_keys, pit_key = match
            path = method_dir / filename
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            values = {}
            for metric, key in metric_keys.items():
                if key in data:
                    records[method][metric].append(data[key])
                    values[metric] = data[key]
            per_split[method][split] = values

            if pit_key in data:
                pit_records[method].extend(data[pit_key])

    return records, per_split, pit_records


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


# The full metric set (crps, sigma, n_test, ...) is always in summary.csv for
# anyone who wants it; the printed table is this subset. Note pro_diag_nll
# requires one particle to jointly explain the whole test vector (a stricter,
# differently-defined quantity than the standard "diagonal NLL" of independent
# per-point marginals, unlike gp_diag_nll, which has no such ambiguity for a
# single non-mixture Gaussian) -- full_nll is the apples-to-apples joint metric.
HEADLINE_METRICS = [
    "nlpd",
    "diag_nll",
    "full_nll",
    "diag_nll_gauss",
    "full_nll_gauss",
    "energy_score",
    "variogram_score",
    "rmse",
]
_SCOPES = [
    ("", "Overall"),
    ("_in_region", "In-region"),
    ("_out_region", "Out-of-region"),
]


def print_table(summary, *, metrics: list[str] = HEADLINE_METRICS):
    print(f"\n{'=' * 60}\n  PEMS graph GP: exact GP vs PRO\n{'=' * 60}")
    method_col = max((len(m) for m in summary), default=6) + 2
    val_col = 16
    for suffix, title in _SCOPES:
        keys = [f"{m}{suffix}" for m in metrics]
        methods = sorted(m for m in summary if any(k in summary[m] for k in keys))
        if not methods:
            continue
        print(f"\n-- {title} --")
        print(f"{'method':<{method_col}}" + "".join(f"{m:>{val_col}}" for m in metrics))
        for method in methods:
            row = f"{method:<{method_col}}"
            for key in keys:
                if key in summary[method]:
                    s = summary[method][key]
                    cell = f"{s['mean']:.3f}±{s['se']:.3f}"
                else:
                    cell = "--"
                row += f"{cell:>{val_col}}"
            print(row)
    print(
        "\n(mean±SE across splits; see summary.csv for the full metric set "
        "-- crps, sigma, n_test -- and per-split values)"
    )


def save_csv(per_split, path: Path):
    rows = []
    for method, splits in per_split.items():
        for split, values in splits.items():
            rows.append({"method": method, "split": split, **values})

    if not rows:
        print("No results found to save.")
        return

    # _PRO_KEYS is a superset of exact_gp's fields (it also has the *_gauss variants),
    # so it's used as the canonical fieldname list -- any exact_gp row just leaves
    # those columns blank.
    fieldnames = ["method", "split", *_PRO_KEYS.keys()]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-split results to {path}")


def print_pit_summary(pit_records, *, num_bins: int = 10, show_bars: bool = False):
    """One line per method: chi-square departure from a uniform PIT (pooled across
    every test point and split). Larger = worse marginal calibration; as a rough
    rule of thumb, values well above num_bins-1 indicate a clear departure from
    uniform (no p-value, to avoid a scipy dependency). Pass show_bars=True for the
    full per-bin ASCII histogram (10 extra lines per method) when you actually want
    to see the shape (e.g. under-vs-over-dispersion) rather than just the summary
    number.
    """
    if not pit_records:
        return

    print(f"\n{'=' * 60}\n  PIT calibration (pooled across splits)\n{'=' * 60}")
    method_col = max((len(m) for m in pit_records), default=6) + 2
    for method, values in sorted(pit_records.items()):
        arr = np.asarray(values)
        n = len(arr)
        if n == 0:
            continue
        counts, edges = np.histogram(arr, bins=num_bins, range=(0.0, 1.0))
        expected = n / num_bins
        chi2_stat = float(np.sum((counts - expected) ** 2 / expected))
        print(
            f"  {method:<{method_col}} chi2={chi2_stat:7.2f}  (df={num_bins - 1}, n={n})"
        )
        if show_bars:
            max_count = counts.max() if counts.max() > 0 else 1
            for i, c in enumerate(counts):
                bar = "#" * round(40 * c / max_count)
                print(f"      [{edges[i]:.1f}, {edges[i + 1]:.1f}) {c:4d} {bar}")


if __name__ == "__main__":
    records, per_split_records, pit_records = collect()
    summary = summarise(records)
    print_table(summary)
    save_csv(per_split_records, RESULTS_ROOT / "summary.csv")
    print_pit_summary(pit_records)
