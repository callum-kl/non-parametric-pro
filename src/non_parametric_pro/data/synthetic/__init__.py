"""Synthetic regression datasets for benchmarking.

Being migrated from the single ``_synthetic_legacy`` module into one file per
scenario under this package -- ``heteroscedastic`` is the first to move.
Everything is re-exported here so existing ``from non_parametric_pro.data.synthetic
import ...`` call sites don't need to change during the migration.
"""

from non_parametric_pro.data._synthetic_legacy import (
    BlockOutlierCase,
    CleanRegressionCase,
    ContaminatedCase,
    HuberData,
    HuberSplit,
    MixtureData,
    contaminated_outlier_pattern,
    make_block_outlier_case,
    make_clean_regression_case,
    make_contaminated_data,
    make_huber_data,
    make_mixture_data,
    regression_truth,
)
from non_parametric_pro.data.synthetic.heteroskedastic import (
    HeteroskedasticCase,
    NoiseRegions,
    heteroskedastic_noise_std,
    make_heteroskedastic_instance,
    sample_noise_regions,
)

__all__ = [
    "BlockOutlierCase",
    "CleanRegressionCase",
    "ContaminatedCase",
    "HeteroskedasticCase",
    "HuberData",
    "HuberSplit",
    "MixtureData",
    "NoiseRegions",
    "contaminated_outlier_pattern",
    "heteroskedastic_noise_std",
    "make_block_outlier_case",
    "make_clean_regression_case",
    "make_contaminated_data",
    "make_heteroskedastic_instance",
    "make_huber_data",
    "make_mixture_data",
    "regression_truth",
    "sample_noise_regions",
]
