"""PrO-GP against the OMGP baselines on the exact-GP or inducing UCI datasets: table and figure."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from run_uci import EXACT_DATASETS, INDUCING_DATASETS

from non_parametric_pro.data.uci.uci import UCI_REGRESSION_DATASET_SIZES

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

KS = (1, 2, 3, 5, 10, 20)
PREFIX = "omgp"
INDUCING = False

# (method dir, metrics file, NLPD key, label)
EXACT_BASELINES = {
    "gp": ("exact_gp", "gp_metrics.json", "gp_nlpd", "Bayes GP"),
    "pro": ("pro_gp_gibbs", "pro_metrics.json", "pro_nlpd", "PrO-GP"),
}
INDUCING_BASELINES = {
    "gp": ("vgp_noncollapsed", "gp_metrics.json", "vgp_nlpd", "SVGP"),
    "pro": ("inducing_pro_gp_gibbs", "pro_metrics.json", "pro_nlpd", "Inducing PrO-GP"),
}
# Integer-target datasets are shown in their dequantised form.
INDUCING_PLOTTED = [ds for ds in INDUCING_DATASETS if f"{ds}_dq" not in INDUCING_DATASETS]


def baselines() -> dict[str, tuple[str, str, str, str]]:
    return INDUCING_BASELINES if INDUCING else EXACT_BASELINES


def methods() -> dict[str, tuple[str, str]]:
    return {
        **{method: (fname, key) for method, fname, key, _ in baselines().values()},
        **{
            f"{PREFIX}_{v}": ("mixture_metrics.json", "mixture_nlpd")
            for v in (*(f"k{k}" for k in KS), "valk")
        },
    }


def labels() -> dict[str, str]:
    return {
        **{method: label for method, _, _, label in baselines().values()},
        **{f"{PREFIX}_k{k}": f"OMGP K={k}" for k in KS},
        f"{PREFIX}_valk": "OMGP val-K",
    }


# Same PrO-GP / GP colours as the other figures (plot_convergence.py, example_grid.py).
PRO_COLOR = "#2ca58d"
GP_COLOR = "#e8974e"
OMGP_COLOR = "#3b6fb6"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def collect(dataset: str, pro_dir: str | None = None) -> dict[str, dict[int, float]]:
    """{method: {split: nlpd}} for the methods with any results, restricted to splits where
    all of them have run. `pro_dir` reads the PrO baseline from another run directory."""
    records = {m: {} for m in methods()}
    pro_method = baselines()["pro"][0]
    for split_dir in (RESULTS_ROOT / dataset).glob("split_*"):
        split = int(split_dir.name.split("_")[1])
        for method, (fname, key) in methods().items():
            run_dir = pro_dir if method == pro_method and pro_dir else method
            path = split_dir / run_dir / fname
            if path.exists():
                records[method][split] = json.loads(path.read_text())[key]
    records = {m: v for m, v in records.items() if v}
    if not records:
        return {}
    common = set.intersection(*(set(v) for v in records.values()))
    return {m: {s: v[s] for s in sorted(common)} for m, v in records.items()}


def selected_ks(dataset: str) -> list[int]:
    paths = sorted((RESULTS_ROOT / dataset).glob(f"split_*/{PREFIX}_valk/mixture_metrics.json"))
    return [json.loads(p.read_text())["selected_k"] for p in paths]


def mean_se(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(se)


def main(
    prefix: str,
    *,
    inducing: bool,
    pro_name: str | None = None,
    raw: bool = False,
    datasets: list[str] | None = None,
) -> None:
    global PREFIX, INDUCING
    PREFIX, INDUCING = prefix, inducing
    gp_method, pro_method = baselines()["gp"][0], baselines()["pro"][0]
    pro_label, gp_label = baselines()["pro"][3], baselines()["gp"][3]
    suffix = ("_inducing" if inducing else "") + (f"_pro_{pro_name}" if pro_name else "")
    # pro_out_dir() names runs `pro_gp_<name>` / `inducing_pro_gp_<name>`.
    pro_override = pro_method.removesuffix("gibbs") + pro_name if pro_name else None
    overridden = set()
    datasets = sorted(
        datasets or (INDUCING_PLOTTED if inducing else EXACT_DATASETS),
        key=lambda d: UCI_REGRESSION_DATASET_SIZES[d][0],
    )
    rows, ranks = [], {m: [] for m in methods()}
    deltas, raws = {}, {}
    for dataset in datasets:
        use_override = pro_override and any((RESULTS_ROOT / dataset).glob(f"split_*/{pro_override}"))
        records = collect(dataset, pro_override if use_override else None)
        if use_override:
            overridden.add(dataset)
        ks = [k for k in KS if f"{PREFIX}_k{k}" in records]
        if pro_method not in records or not ks or not records[pro_method]:
            continue
        splits = list(records[pro_method])
        pro = np.array(list(records[pro_method].values()))
        row = {"dataset": dataset, "num_splits": len(splits)}
        means = {}
        for method, by_split in records.items():
            values = np.array(list(by_split.values()))
            means[method], row[f"{method}_se"] = mean_se(values)
            row[f"{method}_mean"] = means[method]
            row[f"{method}_delta_vs_pro"], row[f"{method}_delta_se"] = mean_se(values - pro)
        row["oracle_k"] = min(ks, key=lambda k: means[f"{PREFIX}_k{k}"])
        row["valk_selected"] = json.dumps(selected_ks(dataset))
        order = sorted(means, key=means.get)
        for method in records:
            ranks[method].append(order.index(method) + 1)
        deltas[dataset] = {
            k: (row[f"{PREFIX}_k{k}_delta_vs_pro"], row[f"{PREFIX}_k{k}_delta_se"]) for k in ks
        } | {"gp": (row[f"{gp_method}_delta_vs_pro"], row[f"{gp_method}_delta_se"])}
        if f"{PREFIX}_valk" in records:
            deltas[dataset]["valk"] = (row[f"{PREFIX}_valk_delta_vs_pro"], row[f"{PREFIX}_valk_delta_se"])
        raws[dataset] = {
            key: (row[f"{method}_mean"], row[f"{method}_se"])
            for key, method in (
                *((k, f"{PREFIX}_k{k}") for k in ks),
                ("gp", gp_method), ("pro", pro_method), ("valk", f"{PREFIX}_valk"),
            )
            if f"{method}_mean" in row
        }
        rows.append(row)

    if not rows:
        print("No OMGP results found; run run_uci.py --mixture-only first.")
        return

    print(f"{'NLPD':<14}" + "".join(f"{labels()[m]:>16}" for m in methods()) + "   oracle K")
    for row in rows:
        line = f"{row['dataset']:<14}"
        for method in methods():
            cell = (f"{row[f'{method}_mean']:>9.3f}±{row[f'{method}_se']:.3f}"
                    if f"{method}_mean" in row else f"{'—':>16}")
            line += cell
        print(line + f"   {row['oracle_k']}  ({row['num_splits']} splits)")
    print(f"{'mean rank':<14}" + "".join(
        f"{np.mean(ranks[m]):>16.2f}" if ranks[m] else f"{'—':>16}" for m in methods()
    ))

    path = RESULTS_ROOT / f"{PREFIX}_summary{suffix}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")

    plot_deltas(raws if raw else deltas, overridden, pro_label, gp_label, pro_override,
                FIGURES_DIR / f"{PREFIX}_uci{suffix}{'_raw' if raw else ''}.png", raw=raw)


def plot_deltas(deltas, overridden, pro_label, gp_label, pro_override, path: Path, *,
                raw: bool = False) -> None:
    """ΔNLPD vs PrO-GP per K, or with `raw` the NLPD itself (mean ± SE over splits) with
    PrO-GP and the GP as bands."""
    ncols = 4 if len(deltas) > 6 else 3
    nrows = -(-len(deltas) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows), sharex=True, squeeze=False)
    for ax, dataset in zip(axes.flat, deltas, strict=False):
        by = deltas[dataset]
        ks = [k for k in KS if k in by]
        mean, se = zip(*(by[k] for k in ks), strict=True)
        ax.errorbar(ks, mean, yerr=se, color=OMGP_COLOR, marker="o", capsize=3,
                    linewidth=1.5, label="OMGP, M components")
        pro_mean = by["pro"][0] if raw else 0.0
        ax.axhline(pro_mean, color=PRO_COLOR, linewidth=1.5,
                   label=pro_label if raw else f"{pro_label} (reference)")
        ax.axhline(by["gp"][0], color=GP_COLOR, linestyle="--", linewidth=1.2, label=gp_label)
        if raw:
            mean, se = by["pro"]
            ax.axhspan(mean - se, mean + se, color=PRO_COLOR, alpha=0.12, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.tick_params(labelsize=13)
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_title(f"{dataset}†" if dataset in overridden else dataset, fontsize=16)
    for ax in axes.flat[len(deltas):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("NLPD" if raw
                      else f"ΔNLPD vs {pro_label}\n(>0: PrO better)")
    legend = {}
    for ax in axes.flat[: len(deltas)]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            legend.setdefault(label, handle)
    fig.legend(legend.values(), legend.keys(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07),
               fontsize=16)
    if overridden:
        fig.text(0.5, 1.0, f"† PrO-GP reference from {pro_override}", ha="center", fontsize=10)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="omgp", choices=("omgp", "omgp_shared"))
    parser.add_argument("--inducing", action="store_true")
    parser.add_argument(
        "--pro-name", default=None,
        help="Use pro_gp_<name> as the PrO reference where it exists, e.g. gibbs_long.",
    )
    parser.add_argument("--raw", action="store_true", help="Plot NLPD rather than ΔNLPD vs PrO-GP.")
    parser.add_argument("--datasets", default=None, help="Comma-separated datasets to plot.")
    args = parser.parse_args()
    main(args.prefix, inducing=args.inducing, pro_name=args.pro_name, raw=args.raw,
         datasets=args.datasets.split(",") if args.datasets else None)
