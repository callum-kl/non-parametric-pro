"""
Kampala PM2.5 air-quality dataset and train/test splits.

Ports the data loading and train/test split logic from
https://github.com/claramst/gps-kampala-airquality (GPflow/TensorFlow) so the same
comparisons can be run from this repo's JAX/gpjax code. Only ``nov-data.csv`` (the file
that repo's ``nowcasting.py``/``forecasting.py`` scripts actually train on) is fetched;
parsing is done with the standard library (``csv``/``datetime``) rather than pandas, to
avoid adding a new dependency for what's otherwise a UCI-style flat-file loader.

**No license.** The source repository has no ``LICENSE`` file and states no license or
attribution for either the code or ``nov-data.csv``, beyond site/device IDs whose format
matches the AirQo (Uganda) network. That's fine for local reproducibility/comparison, but
worth flagging if you plan to redistribute the raw data itself.

**Two split protocols**, both ported from the source repo:

- :func:`kampala_nowcasting_split` -- leave-one-site-out: test is all rows for one
  ``site_id``, train is every other site's rows (subsampled to ``max_train``).
- :func:`kampala_forecasting_split` -- date-cutoff: train is all rows before the last day
  (2021-11-30), test is that day's rows for one ``site_id``.

**One intentional deviation**: the source repo's ``nowcasting.py`` standardizes the target
using ``pm2_5_raw_value``'s mean/std while the target itself is ``pm2_5_calibrated_value``
(a leftover from an alternate, commented-out line -- ``forecasting.py`` uses the consistent,
presumably-intended calibrated-value mean/std). Both splits here use the calibrated-value
statistics consistently; if you need bit-for-bit parity with ``nowcasting.py`` specifically,
recompute ``y_mean``/``y_std`` from ``pm2_5_raw`` instead.

**Not bit-for-bit identical**: the original code resamples training rows via
``pandas.DataFrame.sample(n=1000, random_state=i)``; this module uses
``numpy.random.default_rng(fold)`` instead, since the two libraries' RNG algorithms differ.
Both use ``fold in range(4)`` to match the original's 4-repeat averaging, but the specific
rows selected for a given fold will differ between the two.
"""

import csv
import datetime as dt
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np

_DATA_URL = (
    "https://raw.githubusercontent.com/claramst/gps-kampala-airquality/main/nov-data.csv"
)
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S UTC"
_DEFAULT_CUTOFF_DAY = "2021-11-30"
_DEFAULT_MAX_TRAIN = 1000
_NUM_FOLDS = 4  # matches the source repo's `for i in range(4)` resampling


def package_data_dir(*parts: str) -> Path:
    """Return a path under the repository's top-level ``data/`` directory."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root.joinpath("data", *parts)


def _require_http_url(url: str) -> None:
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        msg = f"Unsupported URL scheme: {scheme!r}"
        raise ValueError(msg)


def download_kampala_airquality(
    *,
    directory: Path | None = None,
    force: bool = False,
    source_url: str = _DATA_URL,
) -> Path:
    """
    Download ``nov-data.csv`` from ``claramst/gps-kampala-airquality``.

    Returns the local path to the CSV file (cached under ``data/kampala_airquality/`` by
    default; re-downloads only if ``force=True`` or the file is missing).
    """
    directory = directory if directory is not None else package_data_dir("kampala_airquality")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "nov-data.csv"
    if path.is_file() and not force:
        return path

    _require_http_url(source_url)
    urllib.request.urlretrieve(source_url, path)  # noqa: S310
    return path


class KampalaAirQualityRecords(NamedTuple):
    """Every row of ``nov-data.csv``, as parallel arrays (one entry per row)."""

    site_id: np.ndarray  # (N,) str
    day: np.ndarray  # (N,) str "YYYY-MM-DD", for date-cutoff splits
    index_day: np.ndarray  # (N,) int, weekday 0=Mon..6=Sun
    index_time: np.ndarray  # (N,) int, hour 0..23
    latitude: np.ndarray  # (N,) float
    longitude: np.ndarray  # (N,) float
    pm2_5_raw: np.ndarray  # (N,) float
    pm2_5_calibrated: np.ndarray  # (N,) float


def load_kampala_airquality_records(
    *,
    directory: Path | None = None,
    force: bool = False,
) -> KampalaAirQualityRecords:
    """Download (if needed) and parse ``nov-data.csv`` into parallel numpy arrays."""
    path = download_kampala_airquality(directory=directory, force=force)

    site_id, day, index_day, index_time = [], [], [], []
    latitude, longitude, pm2_5_raw, pm2_5_calibrated = [], [], [], []

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            timestamp = dt.datetime.strptime(row["timestamp"], _TIMESTAMP_FORMAT)  # noqa: DTZ007
            site_id.append(row["site_id"])
            day.append(timestamp.strftime("%Y-%m-%d"))
            index_day.append(timestamp.weekday())
            index_time.append(timestamp.hour)
            latitude.append(float(row["latitude"]))
            longitude.append(float(row["longitude"]))
            pm2_5_raw.append(float(row["pm2_5_raw_value"]))
            pm2_5_calibrated.append(float(row["pm2_5_calibrated_value"]))

    return KampalaAirQualityRecords(
        site_id=np.array(site_id),
        day=np.array(day),
        index_day=np.array(index_day, dtype=np.int64),
        index_time=np.array(index_time, dtype=np.int64),
        latitude=np.array(latitude),
        longitude=np.array(longitude),
        pm2_5_raw=np.array(pm2_5_raw),
        pm2_5_calibrated=np.array(pm2_5_calibrated),
    )


def kampala_site_ids(records: KampalaAirQualityRecords) -> list[str]:
    """Return the sorted, unique ``site_id``s present in ``records``."""
    return sorted(set(records.site_id.tolist()))


def kampala_outlier_mask(
    records: KampalaAirQualityRecords, *, iqr_multiplier: float = 1.5
) -> np.ndarray:
    """
    Per-site IQR keep-mask over ``pm2_5_calibrated_value`` (``True`` = keep).

    Ports ``sparse_approximations/sparse_gp.py``'s ``df_no_outliers`` construction:
    bounds are computed **per site_id**, using that site's full set of readings (train and
    test rows combined, matching the source script, which builds this mask before
    splitting). Pass the result as ``outlier_mask`` to :func:`kampala_nowcasting_split`/
    :func:`kampala_forecasting_split` to remove these rows from *training* data only --
    the source script never filters test rows, and neither do these split functions.
    """
    keep = np.ones(records.pm2_5_calibrated.shape[0], dtype=bool)
    for site in kampala_site_ids(records):
        idx = np.flatnonzero(records.site_id == site)
        values = records.pm2_5_calibrated[idx]
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        keep[idx] = (values >= lo) & (values <= hi)
    return keep


class KampalaSplit(NamedTuple):
    """One train/test split: features are ``[index_day/7, index_time/24, lat*, lon*]``."""

    x_train: np.ndarray  # (n_train, 4)
    y_train: np.ndarray  # (n_train, 1), standardized
    x_test: np.ndarray  # (n_test, 4)
    y_test: np.ndarray  # (n_test, 1), standardized using train mean/std
    y_mean: float
    y_std: float
    site_id: str
    fold: int


def _features(
    records: KampalaAirQualityRecords,
    idx: np.ndarray,
    *,
    lat_mean: float,
    lat_std: float,
    lon_mean: float,
    lon_std: float,
) -> np.ndarray:
    """``[index_day/7, index_time/24, lat*, lon*]``, matching the source repo's scaling."""
    return np.stack(
        [
            records.index_day[idx] / 7.0,
            records.index_time[idx] / 24.0,
            (records.latitude[idx] - lat_mean) / lat_std,
            (records.longitude[idx] - lon_mean) / lon_std,
        ],
        axis=1,
    )


def _subsample_train_idx(
    train_idx: np.ndarray, *, max_train: int, fold: int
) -> np.ndarray:
    if train_idx.size <= max_train:
        return train_idx
    rng = np.random.default_rng(fold)
    return rng.choice(train_idx, size=max_train, replace=False)


def _finalize_split(
    records: KampalaAirQualityRecords,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    site_id: str,
    fold: int,
) -> KampalaSplit:
    lat_mean, lat_std = records.latitude[train_idx].mean(), records.latitude[train_idx].std()
    lon_mean, lon_std = records.longitude[train_idx].mean(), records.longitude[train_idx].std()
    y_mean = records.pm2_5_calibrated[train_idx].mean()
    y_std = records.pm2_5_calibrated[train_idx].std()

    stats = {"lat_mean": lat_mean, "lat_std": lat_std, "lon_mean": lon_mean, "lon_std": lon_std}
    x_train = _features(records, train_idx, **stats)
    x_test = _features(records, test_idx, **stats)
    y_train = ((records.pm2_5_calibrated[train_idx] - y_mean) / y_std).reshape(-1, 1)
    y_test = ((records.pm2_5_calibrated[test_idx] - y_mean) / y_std).reshape(-1, 1)

    return KampalaSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        y_mean=float(y_mean),
        y_std=float(y_std),
        site_id=site_id,
        fold=fold,
    )


def kampala_nowcasting_split(
    records: KampalaAirQualityRecords,
    site_id: str,
    *,
    fold: int = 0,
    max_train: int = _DEFAULT_MAX_TRAIN,
    outlier_mask: np.ndarray | None = None,
) -> KampalaSplit:
    """
    Leave-one-site-out split: test is all rows for ``site_id``, train is every other row.

    Ports ``nowcasting.py``'s ``train_test_gp``: training rows are subsampled to at most
    ``max_train`` (matching a single one of its ``for i in range(4)`` repeats -- pass
    ``fold`` in ``0..3`` to reproduce a specific repeat; see the module docstring for why
    this isn't bit-for-bit identical to the original's ``pandas``-based resampling).

    ``outlier_mask`` (e.g. from :func:`kampala_outlier_mask`), if given, is intersected
    with the training rows *before* subsampling; test rows are never filtered by it.
    """
    test_mask = records.site_id == site_id
    if not test_mask.any():
        msg = f"No rows found for site_id={site_id!r}"
        raise ValueError(msg)

    test_idx = np.flatnonzero(test_mask)
    train_candidates = ~test_mask
    if outlier_mask is not None:
        train_candidates = train_candidates & outlier_mask
    train_idx = _subsample_train_idx(
        np.flatnonzero(train_candidates), max_train=max_train, fold=fold
    )
    return _finalize_split(records, train_idx, test_idx, site_id=site_id, fold=fold)


def kampala_forecasting_split(
    records: KampalaAirQualityRecords,
    site_id: str,
    *,
    fold: int = 0,
    max_train: int = _DEFAULT_MAX_TRAIN,
    cutoff_day: str = _DEFAULT_CUTOFF_DAY,
    outlier_mask: np.ndarray | None = None,
) -> KampalaSplit:
    """
    Date-cutoff split: train is all rows before ``cutoff_day``, test is ``site_id``'s rows on it.

    Ports ``forecasting.py``'s ``train_test_forecast_gp``. Unlike the nowcasting split,
    ``train`` doesn't depend on ``site_id`` (only on the date cutoff), matching the source
    script computing it once outside its per-site loop.

    ``outlier_mask`` (e.g. from :func:`kampala_outlier_mask`), if given, is intersected
    with the training rows *before* subsampling; test rows are never filtered by it.
    """
    test_day_mask = records.day == cutoff_day
    test_mask = test_day_mask & (records.site_id == site_id)
    if not test_mask.any():
        msg = f"No rows found for site_id={site_id!r} on {cutoff_day!r}"
        raise ValueError(msg)

    test_idx = np.flatnonzero(test_mask)
    train_candidates = ~test_day_mask
    if outlier_mask is not None:
        train_candidates = train_candidates & outlier_mask
    train_idx = _subsample_train_idx(
        np.flatnonzero(train_candidates), max_train=max_train, fold=fold
    )
    return _finalize_split(records, train_idx, test_idx, site_id=site_id, fold=fold)
