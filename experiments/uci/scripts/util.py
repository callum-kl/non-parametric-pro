import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import jax.numpy as jnp
import numpy as np
import paramax as px
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from non_parametric_pro.inducing import PointInducingBasis

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)


_VGP_VARIANT_SUBDIRS = {"noncollapsed": "vgp_noncollapsed", "ppgpr": "ppgpr"}


def split_dir(cfg: DictConfig) -> Path:
    return Path(cfg.results_root) / cfg.dataset / f"split_{cfg.split}"


def gp_state_dir(cfg: DictConfig) -> Path:
    if not cfg.inducing:
        return split_dir(cfg) / "exact_gp"
    subdir = _VGP_VARIANT_SUBDIRS[cfg.vgp_variant]
    if cfg.vgp_name:
        subdir = f"{subdir}_{cfg.vgp_name}"
    return split_dir(cfg) / subdir


def pro_out_dir(cfg: DictConfig) -> Path:
    subdir = "inducing_pro_gp" if cfg.inducing else "pro_gp"
    if cfg.name:
        subdir = f"{subdir}_{cfg.name}"
    return split_dir(cfg) / subdir


def load_gp_state(cfg: DictConfig):
    """Load state from fit_vgp.py (inducing=true) or fit_exact_gp.py (inducing=false)."""
    path = gp_state_dir(cfg) / "gp_state.npz"
    script = "fit_vgp.py" if cfg.inducing else "fit_exact_gp.py"
    if not path.exists():
        raise FileNotFoundError(f"No state at {path} — run {script} first.")

    state = np.load(path)
    kernel_cls = getattr(gpx.kernels, str(state["kernel_type"]))
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


def build_vgp_posterior(x_train: jnp.ndarray, cfg: DictConfig):
    """Zero-mean RBF prior times Gaussian likelihood, initialised as in fit_vgp.py."""
    num_features = x_train.shape[1]
    lengthscale = gpx.parameters.SigmoidBounded(
        jnp.sqrt(num_features) * jnp.ones((num_features,)),
        low=cfg.lengthscale_min,
        high=cfg.lengthscale_max,
    )
    variance = gpx.parameters.PositiveReal(jnp.array([1.0]))
    kernel = gpx.kernels.RBF(lengthscale=lengthscale, variance=variance)
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=x_train.shape[0], obs_stddev=jnp.sqrt(0.01)
    )
    return prior * likelihood


def save_gp_state(out_dir: Path, variational_family, scaler_x, scaler_y) -> None:
    kernel = variational_family.posterior.prior.kernel
    sigma = variational_family.posterior.likelihood.obs_stddev
    np.savez(
        out_dir / "gp_state.npz",
        kernel_type=type(kernel).__name__,
        lengthscale=np.array(px.unwrap(kernel.lengthscale)),
        variance=np.array(px.unwrap(kernel.variance)),
        sigma=np.array(px.unwrap(sigma)).reshape(()),
        z=np.array(px.unwrap(variational_family.inducing_inputs)),
        scaler_x_mean=scaler_x.mean_,
        scaler_x_scale=scaler_x.scale_,
        scaler_y_mean=scaler_y.mean_,
        scaler_y_scale=scaler_y.scale_,
    )
