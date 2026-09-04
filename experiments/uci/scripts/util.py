"""PRO GP sampling, seeded from an exact GP or VGP depending on cfg.inducing."""

import logging
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from non_parametric_pro.inducing import PointInducingBasis

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)


_VGP_VARIANT_SUBDIRS = {
    "collapsed": "vgp",
    "noncollapsed": "vgp_noncollapsed",
    "ppgpr": "ppgpr"
}


def gp_state_dir(cfg: DictConfig) -> Path:
    """
    Locate the state saved by ``fit_vgp.py`` (``cfg.inducing=True``) or
    ``fit_exact_gp.py`` (``cfg.inducing=False``).
    """
    if not cfg.inducing:
        subdir = f"exact_gp_{cfg.gp_name}" if cfg.gp_name else "exact_gp"
        return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / subdir

    if cfg.vgp_variant not in _VGP_VARIANT_SUBDIRS:
        msg = (
            f"Unknown vgp_variant={cfg.vgp_variant!r}; "
            f"expected one of {sorted(_VGP_VARIANT_SUBDIRS)}."
        )
        raise ValueError(msg)
    subdir = _VGP_VARIANT_SUBDIRS[cfg.vgp_variant]
    if cfg.vgp_name:
        subdir = f"{subdir}_{cfg.vgp_name}"
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / subdir


def pro_out_dir(cfg: DictConfig) -> Path:
    subdir = "inducing_pro_gp" if cfg.inducing else "pro_gp"
    if cfg.name:
        subdir = f"{subdir}_{cfg.name}"
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}" / subdir


def load_gp_state(cfg: DictConfig):
    """Load state from fit_vgp.py (inducing=true) or fit_exact_gp.py (inducing=false)."""
    path = gp_state_dir(cfg) / "gp_state.npz"
    script = "fit_vgp.py" if cfg.inducing else "fit_exact_gp.py"
    if not path.exists():
        raise FileNotFoundError(f"No state at {path} — run {script} first.")

    state = np.load(path)
    if "kernel_type" in state:
        kernel_type = str(state["kernel_type"])
    else:
        kernel_type = "Matern32"
        log.warning(
            "%s has no 'kernel_type' (pre-fix save); assuming Matern32. "
            "Re-run %s to save kernel_type explicitly.",
            path,
            script,
        )
    kernel_cls = getattr(gpx.kernels, kernel_type)
    kernel = kernel_cls(
        lengthscale=jnp.array(state["lengthscale"]),
        variance=jnp.array(state["variance"]),
    )
    sigma_val = float(state["sigma"])
    inducing_basis = PointInducingBasis(jnp.array(state["z"])) if cfg.inducing else None

    scaler_x = StandardScaler()
    scaler_x.mean_ = state["scaler_x_mean"]
    scaler_x.scale_ = state["scaler_x_scale"]
    scaler_y = StandardScaler()
    scaler_y.mean_ = state["scaler_y_mean"]
    scaler_y.scale_ = state["scaler_y_scale"]

    return kernel, sigma_val, inducing_basis, scaler_x, scaler_y
