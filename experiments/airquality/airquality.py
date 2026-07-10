import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax.scipy.linalg import solve_triangular

import gpjax as gpx
import optax as ox
import paramax as px

from non_parametric_pro import ula

from non_parametric_pro.density import ProParameters, pro_logdensity_fn, pro_score_fn
from non_parametric_pro.ula import parametric_ula
from non_parametric_pro.util import posterior_function_draws
from non_parametric_pro.data.synthetic import make_contaminated_data, make_mixture_data
from non_parametric_pro.gp import _full_gp_basis
from non_parametric_pro.adaptation.parameter_adaptation import parameter_adaptation
from non_parametric_pro.data.kampala_airquality import load_kampala_airquality_records, kampala_forecasting_split, kampala_site_ids
from non_parametric_pro.inducing import PointInducingBasis, compute_inducing_basis, kmeans_inducing_points

from scipy import stats
from blackjax.util import run_inference_algorithm

from non_parametric_pro.util import nlpd_gp, nlpd_pro

jax.config.update("jax_enable_x64", True)

from non_parametric_pro.data.kampala_airquality import load_kampala_airquality_records, kampala_forecasting_split, kampala_site_ids


if __name__ == "__main__":
    key = jr.PRNGKey(0)

    records = load_kampala_airquality_records()
    site_ids = kampala_site_ids(records)
    data = kampala_forecasting_split(records, site_ids[0], max_train=100000)

    x_train, y_train = jnp.array(data.x_train), jnp.array(data.y_train)
    x_test, y_test = jnp.array(data.x_test), jnp.array(data.y_test)

    M = 1000
    D = x_train.shape[1]
    N = x_train.shape[0]


    gpx_data = gpx.Dataset(X=x_train, y=y_train)
    kernel = gpx.kernels.RBF(lengthscale=jnp.ones((D,)), variance=px.NonTrainable(jnp.array(1.0)))
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=gpx_data.n)
    posterior = prior * likelihood
    optim = ox.adam(learning_rate=0.01)

    z_init = kmeans_inducing_points(key, x_train, M).z 

    variational_family = gpx.variational_families.VariationalGaussian(
        posterior=posterior,
        inducing_inputs=z_init,
    )

    opt_variational_family, _ = gpx.fit(
        model=variational_family,
        objective=lambda p, d: -gpx.objectives.elbo(p, d),
        train_data=gpx_data,
        optim=optim,
        num_iters=500,
        verbose=True,
        batch_size=500
    )


    v_gp_latent_dist = opt_variational_family.predict(x_test)
    v_gp_predictive_dist = opt_variational_family.posterior.likelihood(v_gp_latent_dist)

    v_gp_predictive_mean = v_gp_predictive_dist.mean
    v_gp_predictive_std = jnp.sqrt(v_gp_predictive_dist.variance)

    z_opt = px.unwrap(opt_variational_family.inducing_inputs)

    y_test_raw = data.y_std * y_test.squeeze() + data.y_mean
    v_gp_predictive_mean_raw = data.y_std * v_gp_predictive_mean + data.y_mean


    print(nlpd_gp(y_test, v_gp_predictive_mean, v_gp_predictive_std))
    print(jnp.sqrt(jnp.mean((y_test_raw - v_gp_predictive_mean_raw)**2)))

    

