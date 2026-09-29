from pathlib import Path

import gpjax as gpx
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from non_parametric_pro.data.pems.pems import pems_regression_split

OmegaConf.register_new_resolver(
    "script_dir", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)


def split_dir(cfg: DictConfig) -> Path:
    return Path(cfg.results_root) / f"num_train_{cfg.num_train}" / f"split_{cfg.split}"


def gp_state_dir(cfg: DictConfig) -> Path:
    return split_dir(cfg) / "exact_gp"


def pro_out_dir(cfg: DictConfig) -> Path:
    return split_dir(cfg) / (f"pro_gp_{cfg.name}" if cfg.name else "pro_gp")


def load_split(cfg: DictConfig, graph_data):
    return pems_regression_split(
        graph_data, cfg.split, num_train=cfg.num_train, seed=cfg.split + cfg.split_seed_offset
    )


def load_gp_state(cfg: DictConfig):
    """Load the fixed GraphKernel + sigma + y-scaler saved by fit_exact_gp.py."""
    path = gp_state_dir(cfg) / "gp_state.npz"
    if not path.exists():
        raise FileNotFoundError(f"No state at {path} — run fit_exact_gp.py first.")

    state = np.load(path)
    kernel = gpx.kernels.GraphKernel(
        laplacian=jnp.array(state["laplacian"]),
        lengthscale=jnp.array(state["lengthscale"]),
        variance=jnp.array(state["variance"]),
        smoothness=jnp.array(state["smoothness"]),
    )
    sigma_val = float(state["sigma"])

    scaler_y = StandardScaler()
    scaler_y.mean_ = state["scaler_y_mean"]
    scaler_y.scale_ = state["scaler_y_scale"]

    return kernel, sigma_val, scaler_y
