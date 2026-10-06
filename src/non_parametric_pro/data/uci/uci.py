import csv
import io
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

UCI_REGRESSION_DATASET_SIZES: dict[str, tuple[int, int]] = {
    "abalone": (4_177, 10),
    "airquality": (6_941, 11),
    "autompg": (392, 7),
    "autos": (159, 25),
    "breastcancer": (194, 33),
    "concrete": (1_030, 8),
    "concreteslump": (103, 7),
    "elevators": (16_599, 18),
    "energy": (768, 8),
    "forest": (517, 12),
    "housing": (506, 13),
    "kin40k": (40_000, 8),
    "machine": (209, 7),
    "parkinsons": (5_875, 20),
    "protein": (45_730, 9),
    "servo": (167, 4),
    "skillcraft": (3_338, 19),
    "stock": (536, 11),
    "whitewine": (4_898, 11),
    "wine": (1_599, 11),
    "yacht": (308, 6),
    "abalone_dq": (4_177, 10),
    "whitewine_dq": (4_898, 11),
}

# Integer-valued targets give continuous densities an unbounded likelihood (a mixture can
# put a spike on every level), so these variants add U(-1/2, 1/2) noise to the target:
# the standard dequantisation, under which NLPD scores the mass of each integer's bin.
DEQUANTIZED_DATASETS = {"abalone_dq": "abalone", "whitewine_dq": "whitewine"}
DEQUANTIZE_SEED = 0


NUM_RAW_UCI_SPLITS = 10
UCI_TEST_FRACTION = 0.2
MIN_DATA_COLUMNS = 2


class UCIRegressionDataset(NamedTuple):
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    name: str
    split: int


def package_data_dir(*parts: str) -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root.joinpath("data", *parts)


def _require_http_url(url: str) -> None:
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        msg = f"Unsupported URL scheme: {scheme!r}"
        raise ValueError(msg)


def download_uci_regression_datasets(
    *,
    directory: Path | None = None,
    force: bool = False,
    source_url: str = "https://github.com/treforevans/uci_datasets/archive/refs/heads/master.tar.gz",
) -> Path:
    directory = directory if directory is not None else package_data_dir("uci_datasets")
    if directory.is_dir() and not force:
        return directory

    _require_http_url(source_url)
    directory.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        archive = tmp_path / "uci_datasets.tar.gz"
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        urllib.request.urlretrieve(source_url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(extract_dir, filter="data")

        source_dir = _find_uci_data_root(extract_dir)
        if source_dir is None:
            msg = "Downloaded archive did not contain UCI regression dataset files"
            raise FileNotFoundError(msg)

        if directory.is_dir():
            if not force:
                msg = f"Destination already exists: {directory}"
                raise FileExistsError(msg)
            shutil.rmtree(directory)

        shutil.copytree(source_dir, directory)

    return directory


def _write_uci_regression_dataset(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    directory: Path | None = None,
    seed: int = 0,
) -> Path:
    directory = directory if directory is not None else package_data_dir("uci_datasets")
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), NUM_RAW_UCI_SPLITS)
    mask = np.zeros((n, NUM_RAW_UCI_SPLITS), dtype=np.int64)
    for i, fold in enumerate(folds):
        mask[fold, i] = 1

    dataset_dir = directory / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data = np.hstack([x, y.reshape(-1, 1)])
    np.savetxt(dataset_dir / "data.csv.gz", data, delimiter=",")
    np.savetxt(dataset_dir / "test_mask.csv.gz", mask, delimiter=",", fmt="%d")
    return dataset_dir


def download_wine_quality_white(
    *,
    directory: Path | None = None,
    force: bool = False,
    source_url: str = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
) -> Path:
    directory = directory if directory is not None else package_data_dir("uci_datasets")
    dataset_dir = directory / "whitewine"
    if dataset_dir.is_dir() and not force:
        return dataset_dir

    _require_http_url(source_url)
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "wine-quality.zip"
        urllib.request.urlretrieve(source_url, archive)
        with zipfile.ZipFile(archive) as zf, zf.open("winequality-white.csv") as f:
            rows = list(
                csv.reader(io.TextIOWrapper(f, encoding="utf-8"), delimiter=";")
            )

    data = np.array(rows[1:], dtype=np.float64)
    x, y = data[:, :-1], data[:, -1]
    return _write_uci_regression_dataset("whitewine", x, y, directory=directory)


def download_abalone(
    *,
    directory: Path | None = None,
    force: bool = False,
    source_url: str = "https://archive.ics.uci.edu/static/public/1/abalone.zip",
) -> Path:
    directory = directory if directory is not None else package_data_dir("uci_datasets")
    dataset_dir = directory / "abalone"
    if dataset_dir.is_dir() and not force:
        return dataset_dir

    _require_http_url(source_url)
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "abalone.zip"
        urllib.request.urlretrieve(source_url, archive)
        with zipfile.ZipFile(archive) as zf, zf.open("abalone.data") as f:
            rows = list(csv.reader(io.TextIOWrapper(f, encoding="utf-8")))

    sex = np.array([r[0] for r in rows])
    numeric = np.array([r[1:] for r in rows], dtype=np.float64)
    sex_onehot = np.stack([sex == cat for cat in ("M", "F", "I")], axis=1).astype(
        np.float64
    )
    x = np.hstack([sex_onehot, numeric[:, :-1]])
    y = numeric[:, -1]
    return _write_uci_regression_dataset("abalone", x, y, directory=directory)


def download_air_quality(
    *,
    directory: Path | None = None,
    force: bool = False,
    source_url: str = "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
) -> Path:
    directory = directory if directory is not None else package_data_dir("uci_datasets")
    dataset_dir = directory / "airquality"
    if dataset_dir.is_dir() and not force:
        return dataset_dir

    _require_http_url(source_url)
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "air-quality.zip"
        urllib.request.urlretrieve(source_url, archive)
        with zipfile.ZipFile(archive) as zf, zf.open("AirQualityUCI.csv") as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"), delimiter=";")
            header = next(reader)
            rows = [r for r in reader if r and r[0].strip()]

    drop_cols = {"Date", "Time", "NMHC(GT)", ""}
    keep = [i for i, col_name in enumerate(header) if col_name not in drop_cols]
    keep_names = [header[i] for i in keep]
    data = np.array(
        [[float(r[i].replace(",", ".")) for i in keep] for r in rows], dtype=np.float64
    )
    data = data[~(data == -200).any(axis=1)]

    target_idx = keep_names.index("CO(GT)")
    feature_idx = [i for i in range(data.shape[1]) if i != target_idx]
    x, y = data[:, feature_idx], data[:, target_idx]
    return _write_uci_regression_dataset("airquality", x, y, directory=directory)


def load_uci_regression_dataset(
    dataset: str,
    *,
    split: int = 1,
    directory: Path | None = None,
) -> UCIRegressionDataset:
    """Random UCI_TEST_FRACTION train/test split, reproducible from `split` alone."""
    if split < 1:
        msg = f"split must be >= 1, got {split}."
        raise ValueError(msg)

    directory = directory if directory is not None else package_data_dir("uci_datasets")
    name = _canonical_uci_dataset_name(dataset)
    source = DEQUANTIZED_DATASETS.get(name, name)
    data_root = _find_uci_data_root(directory, source)
    if data_root is None:
        msg = (
            f"Could not find UCI regression data under {directory}. "
            f"Run download_uci_regression_datasets(directory={directory!r}) first."
        )
        raise FileNotFoundError(msg)

    dataset_dir = data_root / source
    data_path = dataset_dir / "data.csv.gz"
    if not data_path.is_file():
        msg = f"Missing data file: {data_path}"
        raise FileNotFoundError(msg)

    data = _read_gzip_csv(data_path, np.float64)
    if data.shape[1] < MIN_DATA_COLUMNS:
        msg = f"Expected at least one feature and one target column in {data_path}"
        raise ValueError(msg)

    n = data.shape[0]
    if name in DEQUANTIZED_DATASETS:
        data[:, -1] += np.random.default_rng(DEQUANTIZE_SEED).uniform(-0.5, 0.5, size=n)
    perm = np.random.default_rng(split).permutation(n)
    num_test = round(UCI_TEST_FRACTION * n)
    test_idx, train_idx = perm[:num_test], perm[num_test:]
    x = data[:, :-1]
    y = data[:, -1:]

    return UCIRegressionDataset(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_test=x[test_idx],
        y_test=y[test_idx],
        name=name,
        split=split,
    )


def _canonical_uci_dataset_name(dataset: str) -> str:
    if dataset not in UCI_REGRESSION_DATASET_SIZES:
        available = ", ".join(sorted(UCI_REGRESSION_DATASET_SIZES))
        msg = f"Unknown UCI regression dataset {dataset!r}. Available: {available}"
        raise ValueError(msg)
    return dataset


def _find_uci_data_root(directory: Path, dataset: str = "concrete") -> Path | None:
    candidates = (
        directory,
        directory / "uci_datasets",
        directory / "uci_datasets" / "uci_datasets",
    )
    for candidate in candidates:
        if (candidate / dataset / "data.csv.gz").is_file() and (
            candidate / dataset / "test_mask.csv.gz"
        ).is_file():
            return candidate

    if directory.is_dir():
        for root, dirs, _files in os.walk(directory):
            if dataset in dirs:
                root_path = Path(root)
                if (root_path / dataset / "data.csv.gz").is_file() and (
                    root_path / dataset / "test_mask.csv.gz"
                ).is_file():
                    return root_path

    return None


def _read_gzip_csv(path: Path, dtype: type) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=dtype)
    return _as_matrix(data)


def _as_matrix(data: np.ndarray) -> np.ndarray:
    return data.reshape(-1, 1) if data.ndim == 1 else data
