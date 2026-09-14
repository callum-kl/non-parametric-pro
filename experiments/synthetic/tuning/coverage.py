"""Empirical coverage of the plotted PrO regions, and its breakdown across x."""

import numpy as np


def enclosed_mass_at(y_query, particle_predictions, sigma_eff, *, num_y=2000, k=8.0):
    """For each test point, the mass of the smallest HPD region containing it --
    the same field `_pro_density_bands` contours, evaluated at the observed y."""
    out = np.empty(y_query.shape[0])
    for i in range(y_query.shape[0]):
        mu = particle_predictions[i]
        s = sigma_eff[i]
        lo, hi = mu.min() - k * s, mu.max() + k * s
        grid = np.linspace(lo, hi, num_y)
        dens = np.exp(-0.5 * ((grid[:, None] - mu[None, :]) / s) ** 2).mean(1)
        dens /= dens.sum()
        order = np.argsort(-dens)
        cum = np.cumsum(dens[order])
        enclosed = np.empty_like(cum)
        enclosed[order] = cum
        out[i] = enclosed[np.argmin(np.abs(grid - y_query[i]))]
    return out


def coverage(result, y_test, levels=(0.5, 0.95)):
    pp = np.asarray(result.particle_predictions)
    se = np.asarray(result.sigma_eff)
    m = enclosed_mass_at(np.asarray(y_test).ravel(), pp, se)
    return {lev: float((m <= lev).mean()) for lev in levels}, m
