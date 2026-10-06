"""
Tuning regret: how much PrO-GP and the OMGP baselines lose by not tuning their
hyperparameter (alpha for PrO-GP, the number of components K for OMGP).

Regret = NLPD(choice) - NLPD(best option of the same method in that setting), computed
per setting (best mean over splits) and per split (best chosen split by split).
Writes CSVs and figures to experiments/tuning_regret/ and prints markdown summaries.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "tuning_regret"

KS = (1, 2, 3, 5, 10)
ALPHAS = (0.5, 1.4, 3.0, 6.0)
DEFAULT_ALPHA = 1.4
REGRET_THRESHOLD = 0.05

SYNTHETIC_REGIMES = ("block_outliers", "heteroskedastic", "multimodal", "well_specified")
SYNTHETIC_NS = (50, 100, 200)
UCI_EXACT = ("concreteslump", "servo", "machine", "autompg", "housing", "stock", "energy", "concrete")
# Integer-target datasets are used in their dequantised form (see uci.DEQUANTIZED_DATASETS).
UCI_INDUCING = ("wine", "skillcraft", "abalone_dq", "whitewine_dq", "parkinsons", "airquality")
# Reduced runs: 3 splits, 500 Gibbs steps, no val-K.
UCI_LARGE = ("elevators", "protein")
PEMS_NS = (200, 225, 250, 275)
# stock's alpha sweep was stopped part-way.
ALPHA_DATASETS = ("concreteslump", "servo", "machine", "autompg", "housing")

PRO_COLOR = "#e8974e"
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


class Setting(NamedTuple):
    group: str
    label: str
    n: int | None
    pro: np.ndarray
    per_k: dict[int, np.ndarray]
    valk: np.ndarray | None


def _read(path: Path, key: str):
    return json.loads(path.read_text())[key]


def _instance_nlpds(setting_dir: Path, algo: str) -> np.ndarray:
    return np.array(_read(setting_dir / algo / "metrics.json", "nlpds"))


def _split_nlpds(splits: list[Path], run: str, fname: str, key: str) -> np.ndarray:
    return np.array([_read(s / run / fname, key) for s in splits])


def synthetic_settings(prefix: str):
    root = ROOT / "synthetic" / "results"
    for regime in SYNTHETIC_REGIMES:
        for n in SYNTHETIC_NS:
            d = root / regime / f"n_{n}"
            if not (d / f"{prefix}_valk" / "metrics.json").exists():
                continue
            per_k = {
                k: _instance_nlpds(d, f"{prefix}_k{k}")
                for k in KS
                if (d / f"{prefix}_k{k}" / "metrics.json").exists()
            }
            yield Setting("synthetic", f"{regime} n={n}", n, _instance_nlpds(d, "pro_gp"), per_k,
                          _instance_nlpds(d, f"{prefix}_valk"))


def _split_settings(group, unit_dirs, pro_dir, prefix):
    for label, n, d in unit_dirs:
        splits = sorted(
            (s for s in d.glob("split_*") if (s / f"{prefix}_k1" / "mixture_metrics.json").exists()),
            key=lambda p: int(p.name.split("_")[1]),
        )
        if not splits:
            continue

        def has(algo, splits=splits):
            return all((s / algo / "mixture_metrics.json").exists() for s in splits)

        def mix(algo, splits=splits):
            return _split_nlpds(splits, algo, "mixture_metrics.json", "mixture_nlpd")

        pro = _split_nlpds(splits, pro_dir, "pro_metrics.json", "pro_nlpd")
        per_k = {k: mix(f"{prefix}_k{k}") for k in KS if has(f"{prefix}_k{k}")}
        valk = mix(f"{prefix}_valk") if has(f"{prefix}_valk") else None
        yield Setting(group, label, n, pro, per_k, valk)


def all_settings(prefix: str) -> list[Setting]:
    uci = ROOT / "uci" / "results"
    pems = ROOT / "pems" / "results"
    return [
        *synthetic_settings(prefix),
        *_split_settings("uci_exact", [(ds, None, uci / ds) for ds in UCI_EXACT],
                         "pro_gp_gibbs", prefix),
        *_split_settings("uci_inducing", [(ds, None, uci / ds) for ds in UCI_INDUCING],
                         "inducing_pro_gp_gibbs", prefix),
        *_split_settings("uci_large", [(ds, None, uci / ds) for ds in UCI_LARGE],
                         "inducing_pro_gp_gibbs", prefix),
        *_split_settings("pems", [(f"pems n={n}", n, pems / f"num_train_{n}") for n in PEMS_NS],
                         "pro_gp_gibbs", prefix),
    ]


def _own_regret(s: Setting, k: int) -> float:
    return s.per_k[k].mean() - min(v.mean() for v in s.per_k.values())


def mixture_regrets(settings: list[Setting]) -> list[dict]:
    """Per-setting and per-split regret of each mixture choice, against the best K that
    setting was run with. Transferred K is restricted to the Ks every setting has."""
    common_ks = [k for k in KS if all(k in s.per_k for s in settings)]
    rows = []
    for i, s in enumerate(settings):
        ks = sorted(s.per_k)
        grid = np.stack([s.per_k[k] for k in ks])
        means = grid.mean(axis=1)
        best_mean, best_split = means.min(), grid.min(axis=0)
        others = settings[:i] + settings[i + 1:]
        transferred = min(common_ks, key=lambda k: np.mean([_own_regret(o, k) for o in others]))
        choices = {
            **{f"K={k}": s.per_k[k] for k in ks},
            "random K": grid.mean(axis=0),
            "transferred K": s.per_k[transferred],
        }
        if s.valk is not None:
            choices["val-K"] = s.valk
        for choice, values in choices.items():
            rows.append({
                "group": s.group, "setting": s.label, "choice": choice,
                "regret": float(values.mean() - best_mean),
                "regret_per_split": float(np.mean(values - best_split)),
                "best_k": ks[int(means.argmin())],
                "ks_run": ",".join(map(str, ks)),
                "transferred_k": transferred,
                "pro_minus_best_k": float(s.pro.mean() - best_mean),
            })
    return rows


def pro_alpha_regrets() -> list[dict]:
    uci = ROOT / "uci" / "results"
    rows = []
    for ds in ALPHA_DATASETS:
        splits = sorted(
            (s for s in (uci / ds).glob("split_*") if (s / "pro_gp_valalpha" / "pro_metrics.json").exists()),
            key=lambda p: int(p.name.split("_")[1]),
        )
        def pro(run, splits=splits):
            return _split_nlpds(splits, run, "pro_metrics.json", "pro_nlpd")

        per_alpha = {a: pro(f"pro_gp_alpha{a:g}") for a in ALPHAS}
        grid = np.stack([per_alpha[a] for a in ALPHAS])
        best_mean, best_split = grid.mean(axis=1).min(), grid.min(axis=0)
        choices = {
            **{f"alpha={a:g}": per_alpha[a] for a in ALPHAS},
            "val-alpha": pro("pro_gp_valalpha"),
            "d_eff rule": pro("pro_gp_deffalpha"),
        }
        for choice, values in choices.items():
            rows.append({
                "group": "uci_exact", "setting": ds, "choice": choice,
                "regret": float(values.mean() - best_mean),
                "regret_per_split": float(np.mean(values - best_split)),
                "best_alpha": ALPHAS[int(grid.mean(axis=1).argmin())],
            })
    return rows


def summarise(rows: list[dict], order: list[str]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[(r["group"], r["choice"])].append(r)
        by[("all", r["choice"])].append(r)
    out = []
    for group in ("all", "synthetic", "uci_exact", "uci_inducing", "uci_large", "pems"):
        for choice in order:
            rs = by.get((group, choice))
            if not rs:
                continue
            regret = np.array([r["regret"] for r in rs])
            out.append({
                "group": group, "choice": choice, "settings": len(rs),
                "mean_regret": float(regret.mean()), "max_regret": float(regret.max()),
                "num_over_threshold": int((regret > REGRET_THRESHOLD).sum()),
                "mean_regret_per_split": float(np.mean([r["regret_per_split"] for r in rs])),
            })
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def print_summary(title: str, summary: list[dict]) -> None:
    print(f"\n### {title}\n")
    print(f"| group | choice | settings | mean regret | max regret | # > {REGRET_THRESHOLD} | mean per-split regret |")
    print("|---|---|---|---|---|---|---|")
    for r in summary:
        print(f"| {r['group']} | {r['choice']} | {r['settings']} | {r['mean_regret']:.3f} | "
              f"{r['max_regret']:.3f} | {r['num_over_threshold']} | {r['mean_regret_per_split']:.3f} |")


def best_k_vs_n(settings: list[Setting]) -> list[dict]:
    rows = []
    for s in settings:
        if s.n is None:
            continue
        means = {k: v.mean() for k, v in s.per_k.items()}
        best = min(means, key=means.get)
        rows.append({
            "series": s.label.rsplit(" n=", 1)[0], "n": s.n, "best_k": best,
            "pro_minus_best_k": float(s.pro.mean() - means[best]),
        })
    return rows


def plot_best_k_vs_n(rows: list[dict], path: Path) -> None:
    """Synthetic regimes only; PeMS's narrow n range is reported in the table."""
    rows = [r for r in rows if r["series"] in SYNTHETIC_REGIMES]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    # Small vertical offsets keep regimes with the same best K visible.
    offsets = np.linspace(-0.12, 0.12, len(SYNTHETIC_REGIMES))
    for series, offset in zip(SYNTHETIC_REGIMES, offsets, strict=True):
        pts = sorted((r["n"], r["best_k"], r["pro_minus_best_k"]) for r in rows if r["series"] == series)
        if not pts:
            continue
        n, k, gap = zip(*pts, strict=True)
        label = series.replace("_", " ")
        axes[0].plot(n, np.array(k) + offset, marker="o", linewidth=1.5, label=label)
        axes[1].plot(n, gap, marker="o", linewidth=1.5, label=label)
    axes[0].set_ylabel("best mixture K")
    axes[0].set_yticks(KS)
    axes[1].axhline(0.0, color=PRO_COLOR, linewidth=1.2)
    axes[1].set_ylabel("PrO-GP (default) $-$ best-K OMGP\nNLPD (<0: PrO better)")
    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xticks(SYNTHETIC_NS, [str(n) for n in SYNTHETIC_NS])
        ax.set_xlabel("training points n")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")


def plot_sensitivity(mixture_rows: list[dict], alpha_rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, len(ALPHA_DATASETS), figsize=(3.2 * len(ALPHA_DATASETS), 5.6),
                             sharey="col")
    for j, ds in enumerate(ALPHA_DATASETS):
        alpha = {r["choice"]: r["regret"] for r in alpha_rows if r["setting"] == ds}
        mix = {r["choice"]: r["regret"] for r in mixture_rows if r["setting"] == ds}
        top, bottom = axes[0, j], axes[1, j]
        top.plot(ALPHAS, [alpha[f"alpha={a:g}"] for a in ALPHAS], color=PRO_COLOR,
                 marker="^", linewidth=1.5)
        top.axvline(DEFAULT_ALPHA, color=PRO_COLOR, linestyle=":", linewidth=1.2)
        top.set_xlabel(r"$\alpha$ (default 1.4 dotted)")
        top.set_title(ds, fontsize=12)
        ks = [k for k in KS if f"K={k}" in mix]
        bottom.plot(ks, [mix[f"K={k}"] for k in ks], color=OMGP_COLOR, marker="o", linewidth=1.5)
        bottom.axvline(5, color=OMGP_COLOR, linestyle=":", linewidth=1.2)
        bottom.set_xlabel("K (best fixed K=5 dotted)")
        for ax, ticks in ((top, ALPHAS), (bottom, ks)):
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    axes[0, 0].set_ylabel("PrO-GP regret\n" + r"(NLPD $-$ best $\alpha$)")
    axes[1, 0].set_ylabel("OMGP regret\n" + r"(NLPD $-$ best K)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")


def main(prefix: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    settings = all_settings(prefix)
    mixture_rows = mixture_regrets(settings)
    alpha_rows = pro_alpha_regrets()

    write_csv(mixture_rows, OUT_DIR / f"{prefix}_regret.csv")
    write_csv(alpha_rows, OUT_DIR / "pro_alpha_regret.csv")
    mixture_summary = summarise(
        mixture_rows,
        [*(f"K={k}" for k in KS), "random K", "transferred K", "val-K"],
    )
    alpha_summary = summarise(
        alpha_rows, [*(f"alpha={a:g}" for a in ALPHAS), "val-alpha", "d_eff rule"]
    )
    write_csv(mixture_summary, OUT_DIR / f"{prefix}_regret_summary.csv")
    write_csv(alpha_summary, OUT_DIR / "pro_alpha_regret_summary.csv")

    print_summary(f"OMGP ({prefix}) regret vs its best K among those run (up to {KS})", mixture_summary)
    print_summary(f"PrO-GP regret vs its best alpha in {ALPHAS}", alpha_summary)

    best_k = best_k_vs_n(settings)
    write_csv(best_k, OUT_DIR / f"{prefix}_best_k_vs_n.csv")
    print("\n### Best K vs n\n")
    print("| series | n | best K | PrO - best-K OMGP |")
    print("|---|---|---|---|")
    for r in best_k:
        print(f"| {r['series']} | {r['n']} | {r['best_k']} | {r['pro_minus_best_k']:+.3f} |")

    plot_best_k_vs_n(best_k, OUT_DIR / f"{prefix}_best_k_vs_n.png")
    plot_sensitivity(mixture_rows, alpha_rows, OUT_DIR / f"{prefix}_sensitivity.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="omgp_shared", choices=("omgp", "omgp_shared"))
    main(parser.parse_args().prefix)
