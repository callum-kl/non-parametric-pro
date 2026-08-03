import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import gpjax as gpx
import optax as ox
import paramax as px

from blackjax.util import run_inference_algorithm
from sklearn.preprocessing import StandardScaler
from scipy import stats

from non_parametric_pro import ula
from non_parametric_pro.density import ProParameters, pro_logdensity_fn, regularised_score
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import prediction_basis, nlpd_gp, nlpd_pro, crps_gp, crps_pro
from non_parametric_pro.data.uci import load_uci_regression_dataset
from non_parametric_pro.adaptation.parameter_adaptation import parameter_adaptation, cross_validated_parameter_adaptation
from non_parametric_pro.gp import _full_gp_basis
from non_parametric_pro.util import train_val_split, run_inference_algorithm_with_burn_in

