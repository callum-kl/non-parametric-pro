"""
Normality diagnostics for a fitted exact GP.

Reloads the state saved by ``fit_exact_gp.py`` and checks the Gaussian
assumption via three residual sets:

``whitened``
    z = L^-1 y with L = chol(K + sigma^2 I) on the training data. Under the
    model these are *exactly* i.i.d. N(0, 1), so formal normality tests are
    valid here and nowhere else.
``loo``
    Closed-form leave-one-out z-scores (Rasmussen & Williams eq. 5.12).
    Marginally N(0, 1) but correlated -- good for spotting outliers, not for
    p-values.
``test``
    Held-out standardised predictive residuals. Marginally N(0, 1), correlated.
"""

import json
import logging
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import hydra
import matplotlib
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from scipy.linalg import cho_factor, cho_solve, solve_triangular

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from util import gp_state_dir, load_gp_state

from non_parametric_pro.data.uci.uci import load_uci_regression_dataset

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdbd6"
SERIES = {"whitened": "#2a78d6", "loo": "#eb6834", "test": "#1baf7a"}
REF = "#8a8984"


def out_dir(cfg: DictConfig) -> Path:
    return gp_state_dir(cfg) / "normality"


def _blom(n: int) -> np.ndarray:
    return (np.arange(1, n + 1) - 0.375) / (n + 0.25)


def _qq_envelope(n: int, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise band for N(0,1) order statistics via their Beta distribution."""
    i = np.arange(1, n + 1)
    lo_p = (1.0 - level) / 2.0
    lo = stats.norm.ppf(stats.beta.ppf(lo_p, i, n - i + 1))
    hi = stats.norm.ppf(stats.beta.ppf(1.0 - lo_p, i, n - i + 1))
    return lo, hi


def compute_residuals(
    kernel, sigma: float, x_train, y_train, x_test, y_test, jitter: float
) -> dict[str, dict[str, np.ndarray]]:
    k_ff = np.asarray(kernel.gram(x_train).as_matrix(), dtype=float)
    n = k_ff.shape[0]
    k_tilde = k_ff + (sigma**2 + jitter) * np.eye(n)

    chol_lower = np.linalg.cholesky(k_tilde)
    z_white = solve_triangular(chol_lower, y_train, lower=True)

    c = cho_factor(k_tilde, lower=True)
    k_inv = cho_solve(c, np.eye(n))
    alpha = cho_solve(c, y_train)
    loo_var = 1.0 / np.diag(k_inv)
    loo_mean = y_train - alpha * loo_var
    z_loo = alpha / np.sqrt(np.diag(k_inv))

    k_sf = np.asarray(kernel.cross_covariance(x_test, x_train), dtype=float)
    test_mean = k_sf @ alpha
    k_ss_diag = np.asarray(kernel.gram(x_test).as_matrix(), dtype=float).diagonal()
    v = cho_solve(c, k_sf.T)
    latent_var = k_ss_diag - np.einsum("ij,ji->i", k_sf, v)
    test_std = np.sqrt(np.maximum(latent_var, 0.0) + sigma**2)
    z_test = (y_test - test_mean) / test_std

    return {
        "whitened": {"z": z_white, "covariate": np.arange(n, dtype=float)},
        "loo": {"z": z_loo, "covariate": loo_mean},
        "test": {"z": z_test, "covariate": test_mean},
    }


def normality_stats(z: np.ndarray) -> dict[str, float]:
    z = np.asarray(z, dtype=float)
    n = z.size
    shapiro = stats.shapiro(z)
    dagostino = stats.normaltest(z)
    jarque = stats.jarque_bera(z)
    ks = stats.kstest(z, "norm")
    anderson = stats.anderson(z, dist="norm", method="interpolate")
    return {
        "n": int(n),
        "mean": float(z.mean()),
        "std": float(z.std(ddof=1)),
        "skew": float(stats.skew(z)),
        "excess_kurtosis": float(stats.kurtosis(z)),
        "shapiro_W": float(shapiro.statistic),
        "shapiro_p": float(shapiro.pvalue),
        "dagostino_p": float(dagostino.pvalue),
        "jarque_bera_p": float(jarque.pvalue),
        "ks_standard_normal_D": float(ks.statistic),
        "ks_standard_normal_p": float(ks.pvalue),
        "anderson_A2": float(anderson.statistic),
        "anderson_p": float(anderson.pvalue),
        "frac_beyond_2sigma": float(np.mean(np.abs(z) > 2.0)),
        "frac_beyond_3sigma": float(np.mean(np.abs(z) > 3.0)),
    }


def target_summary(y: np.ndarray) -> dict[str, float]:
    """A large atom in y caps how Gaussian any residual can be."""
    values, counts = np.unique(y, return_counts=True)
    return {
        "n": int(y.size),
        "num_unique": int(values.size),
        "modal_value": float(values[counts.argmax()]),
        "modal_fraction": float(counts.max() / y.size),
        "skew": float(stats.skew(y)),
        "excess_kurtosis": float(stats.kurtosis(y)),
    }


def whitening_order_check(
    k_tilde: np.ndarray, y: np.ndarray, num_permutations: int, seed: int
) -> dict[str, float]:
    """Cholesky whitening depends on point order; resample it to check stability."""
    rng = np.random.default_rng(seed)
    n = k_tilde.shape[0]
    p_values = []
    for _ in range(num_permutations):
        perm = rng.permutation(n)
        chol_lower = np.linalg.cholesky(k_tilde[np.ix_(perm, perm)])
        z = solve_triangular(chol_lower, y[perm], lower=True)
        p_values.append(float(stats.shapiro(z).pvalue))
    p_values = np.asarray(p_values)
    return {
        "num_permutations": int(num_permutations),
        "shapiro_p_median": float(np.median(p_values)),
        "shapiro_p_q05": float(np.quantile(p_values, 0.05)),
        "shapiro_p_q95": float(np.quantile(p_values, 0.95)),
        "frac_reject_at_5pct": float(np.mean(p_values < 0.05)),
    }


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def _qq_points(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(z)
    theoretical = stats.norm.ppf(_blom(ordered.size))
    return theoretical, ordered


def draw_qq(ax, z: np.ndarray, colour: str) -> None:
    theoretical, ordered = _qq_points(z)
    lo, hi = _qq_envelope(ordered.size)
    ax.fill_between(
        theoretical,
        lo,
        hi,
        color=REF,
        alpha=0.15,
        linewidth=0,
        label="95% pointwise band",
    )
    ax.plot(theoretical, theoretical, color=REF, linewidth=2, zorder=2, label="N(0, 1)")
    ax.scatter(
        theoretical,
        ordered,
        s=14,
        color=colour,
        edgecolor=SURFACE,
        linewidth=0.5,
        zorder=3,
    )
    _style_axes(ax)


def draw_histogram(ax, z: np.ndarray, colour: str) -> None:
    ax.hist(
        z,
        bins=min(40, max(12, int(np.sqrt(z.size) * 1.5))),
        density=True,
        color=colour,
        alpha=0.55,
        edgecolor=SURFACE,
        linewidth=0.5,
    )
    grid = np.linspace(min(-4.0, z.min()), max(4.0, z.max()), 400)
    ax.plot(grid, stats.norm.pdf(grid), color=REF, linewidth=2, label="N(0, 1)")
    _style_axes(ax)


def draw_residual_scatter(
    ax, z: np.ndarray, covariate: np.ndarray, colour: str
) -> None:
    ax.axhline(0.0, color=REF, linewidth=2, zorder=2)
    for level in (-2.0, 2.0):
        ax.axhline(level, color=REF, linewidth=1, linestyle="--", alpha=0.7, zorder=2)
    ax.scatter(
        covariate,
        z,
        s=14,
        color=colour,
        edgecolor=SURFACE,
        linewidth=0.5,
        zorder=3,
    )
    _style_axes(ax)


def plot_panel(residuals: dict, path: Path) -> None:
    payload = residuals["whitened"]
    z = payload["z"]
    colour = SERIES["whitened"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9), facecolor=SURFACE)

    draw_qq(axes[0], z, colour)
    axes[0].set_title("Q-Q vs N(0, 1)", fontsize=10, color=INK)
    axes[0].set_xlabel("Theoretical quantile", fontsize=9, color=INK_MUTED)
    axes[0].set_ylabel("Observed quantile", fontsize=9, color=INK_MUTED)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left")

    draw_histogram(axes[1], z, colour)
    axes[1].set_title("Density vs N(0, 1)", fontsize=10, color=INK)
    axes[1].set_xlabel("z", fontsize=9, color=INK_MUTED)
    axes[1].set_ylabel("Density", fontsize=9, color=INK_MUTED)

    draw_residual_scatter(axes[2], z, payload["covariate"], colour)
    axes[2].set_title("Residual vs index", fontsize=10, color=INK)
    axes[2].set_xlabel("Training index", fontsize=9, color=INK_MUTED)
    axes[2].set_ylabel("z", fontsize=9, color=INK_MUTED)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../conf", config_name="gp_normality")
def main(cfg: DictConfig) -> None:
    log.info("GP normality: dataset=%s split=%d", cfg.dataset, cfg.split)

    kernel, sigma, _, scaler_x, scaler_y = load_gp_state(cfg)

    example = load_uci_regression_dataset(cfg.dataset, split=cfg.split)
    x_train = scaler_x.transform(example.x_train)
    y_train = scaler_y.transform(example.y_train).squeeze(-1)
    x_test = scaler_x.transform(example.x_test)
    y_test = scaler_y.transform(example.y_test).squeeze(-1)
    log.info(
        "N_train=%d  N_test=%d  D=%d  sigma=%.4f",
        x_train.shape[0],
        x_test.shape[0],
        x_train.shape[1],
        sigma,
    )

    residuals = compute_residuals(
        kernel, sigma, x_train, y_train, x_test, y_test, cfg.jitter
    )
    stats_by_type = {name: normality_stats(p["z"]) for name, p in residuals.items()}

    k_ff = np.asarray(kernel.gram(x_train).as_matrix(), dtype=float)
    k_tilde = k_ff + (sigma**2 + cfg.jitter) * np.eye(k_ff.shape[0])
    order_check = whitening_order_check(
        k_tilde, y_train, cfg.num_permutations, cfg.seed
    )
    target = target_summary(y_train)

    for name, s in stats_by_type.items():
        log.info(
            "%-8s n=%4d sd=%.3f skew=%+.3f ex.kurt=%+.3f shapiro_p=%.3e ks_p=%.3e",
            name,
            s["n"],
            s["std"],
            s["skew"],
            s["excess_kurtosis"],
            s["shapiro_p"],
            s["ks_standard_normal_p"],
        )
    log.info(
        "Target y_train: %d unique of %d, modal value %.3f holds %.1f%% of mass, skew=%+.3f",
        target["num_unique"],
        target["n"],
        target["modal_value"],
        100 * target["modal_fraction"],
        target["skew"],
    )
    log.info(
        "Whitening-order check: median Shapiro p=%.3e, rejects at 5%% in %.0f%% of %d orders",
        order_check["shapiro_p_median"],
        100 * order_check["frac_reject_at_5pct"],
        order_check["num_permutations"],
    )

    target_dir = out_dir(cfg)
    target_dir.mkdir(parents=True, exist_ok=True)
    panel_path = (
        Path(cfg.results_root) / f"normality_panel_{cfg.dataset}_split{cfg.split}.png"
    )
    plot_panel(residuals, panel_path)

    payload = {
        "dataset": cfg.dataset,
        "split": cfg.split,
        "sigma": float(sigma),
        "residuals": stats_by_type,
        "target": target,
        "whitening_order_check": order_check,
    }
    with open(target_dir / "normality_stats.json", "w") as f:
        json.dump(payload, f, indent=2)

    np.savez(
        target_dir / "residuals.npz",
        **{f"z_{name}": p["z"] for name, p in residuals.items()},
    )
    log.info("Saved diagnostics to %s and panel to %s", target_dir, panel_path)


if __name__ == "__main__":
    main()
