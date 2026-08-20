"""PRO GP sampling, seeded from an exact GP or VGP depending on cfg.inducing."""

import json
import logging
import os
from pathlib import Path

# Must be set before `import jax` (and before any transitive jax import, e.g. via
# gpjax) -- jax.config.update("jax_enable_x64", True) here isn't enough, since under
# `-m hydra/launcher=joblib` the fit runs in a joblib worker process that doesn't
# reliably replay this module's own top-level statements before jax's backend
# initializes, silently leaving that worker on float32.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
import paramax as px
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from non_parametric_pro import ula
from non_parametric_pro.parameter_adaptation import parameter_adaptation
from non_parametric_pro.data.uci import load_uci_regression_dataset
from non_parametric_pro.density import ProParameters, pro_logdensity_fn, regularised_score
from non_parametric_pro.inducing import PointInducingBasis, compute_inducing_basis
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import crps_pro, nlpd_pro, prediction_basis, run_inference_algorithm_with_burn_in

log = logging.getLogger(__name__)

# Anchors results_root/hydra.run.dir/hydra.sweep.dir to this script's own directory
# (experiments/uci/), regardless of the caller's current working directory.
OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)


_VGP_VARIANT_SUBDIRS = {
    "collapsed": "vgp",
    "noncollapsed": "vgp_noncollapsed",
}


def gp_state_dir(cfg: DictConfig) -> Path:
    """
    Locate the state saved by ``fit_vgp.py`` (``cfg.inducing=True``) or
    ``fit_exact_gp.py`` (``cfg.inducing=False``).

    ``cfg.vgp_variant`` picks which of ``fit_vgp.py``'s training methods to load --
    ``"collapsed"`` (``CollapsedVariationalGaussian`` + ``fit_scipy``, the default) or
    ``"noncollapsed"`` (``VariationalGaussian``, fit with either plain Adam or natural
    gradients on ``q(u)`` -- see ``non_parametric_pro.gp.natural_gradient_svgp_fit``;
    those two share the same ``vgp_noncollapsed`` directory since natural gradients is
    just a different optimiser for the same non-collapsed model, not a separate one --
    whichever ``fit_vgp.py`` run happened most recently is what's here). This must match
    whichever ``fit_vgp.py collapsed=...`` value was actually run -- see that script's
    ``state_dir`` for the exact same subdirectory names.
    ``cfg.vgp_name`` mirrors ``fit_vgp.py``'s own ``name`` override, for loading a
    specifically-named run rather than the bare variant directory. ``cfg.gp_name``
    is the same idea for the non-inducing case, mirroring ``fit_exact_gp.py``'s own
    ``name`` override -- e.g. ``gp_name=loo`` loads ``exact_gp_loo`` (as saved by
    ``fit_exact_gp.py objective=loocv name=loo``) instead of the bare ``exact_gp``.
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


def _cholesky_basis(kernel, x, jitter=1e-6):
    k = kernel.gram(x).as_matrix()
    return jnp.linalg.cholesky(k + jitter * jnp.eye(k.shape[0]))


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
        # Pre-existing states saved before kernel_type was recorded -- these were all
        # fit with Matern32, so this matches their actual behaviour; re-run
        # fit_exact_gp.py/fit_vgp.py to pick up kernel_type for new fits.
        kernel_type = "Matern32"
        log.warning(
            "%s has no 'kernel_type' (pre-fix save); assuming Matern32. "
            "Re-run %s to save kernel_type explicitly.",
            path, script,
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