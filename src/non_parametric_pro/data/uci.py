"""Download and load the standardized UCI regression datasets (ported from Julia)."""

import os
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np

UCI_REGRESSION_DATASET_SIZES: dict[str, tuple[int, int]] = {
    "challenger": (23, 4),
    "fertility": (100, 9),
    "concreteslump": (103, 7),
    "autos": (159, 25),
    "servo": (167, 4),
    "breastcancer": (194, 33),
    "machine": (209, 7),
    "yacht": (308, 6),
    "autompg": (392, 7),
    "housing": (506, 13),
    "forest": (517, 12),
    "stock": (536, 11),
    "pendulum": (630, 9),
    "energy": (768, 8),
    "concrete": (1_030, 8),
    "solar": (1_066, 10),
    "airfoil": (1_503, 5),
    "wine": (1_599, 11),
    "gas": (2_565, 128),
    "skillcraft": (3_338, 19),
    "sml": (4_137, 26),
    "parkinsons": (5_875, 20),
    "pumadyn32nm": (8_192, 32),
    "pol": (15_000, 26),
    "elevators": (16_599, 18),
    "bike": (17_379, 17),
    "kin40k": (40_000, 8),
    "protein": (45_730, 9),
    "tamielectric": (45_781, 3),
    "keggdirected": (48_827, 20),
    "slice": (53_500, 385),
    "keggundirected": (63_608, 27),
    "3droad": (434_874, 3),
    "song": (515_345, 90),
    "buzz": (583_250, 77),
    "houseelectric": (2_049_280, 11),
}

UCI_REGRESSION_DATASET_ALIASES: dict[str, str] = {
    "automobile": "autos",
    "cancer": "breastcancer",
    "ctslice": "slice",
    "electric": "houseelectric",
    "forestfires": "forest",
    "gassensor": "gas",
    "hardware": "machine",
    "kegg": "keggdirected",
    "keggu": "keggundirected",
    "poletele": "pol",
    "pumadyn": "pumadyn32nm",
    "slump": "concreteslump",
    "solarflare": "solar",
}


NUM_RAW_UCI_SPLITS = 10
NUM_UCI_SPLITS = 5  # each merges a consecutive pair of the 10 raw masks -> larger test sets
MIN_DATA_COLUMNS = 2


class UCIRegressionDataset(NamedTuple):
    """One standardized UCI regression train/test split."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    name: str
    split: int


def package_data_dir(*parts: str) -> Path:
    """Return a path under the repository's top-level ``data/`` directory."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root.joinpath("data", *parts)


def uci_regression_datasets(*, include_aliases: bool = False) -> list[str]:
    """
    Return the available standardized UCI regression dataset names.

    The values are the directory names used by ``uci_datasets``; aliases such
    as ``"electric"`` can be included with ``include_aliases=True``.
    """
    names = sorted(UCI_REGRESSION_DATASET_SIZES)
    if not include_aliases:
        return names
    return sorted(set(names) | set(UCI_REGRESSION_DATASET_ALIASES))


def _require_http_url(url: str) -> None:
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        msg = f"Unsupported URL scheme: {scheme!r}"
        raise ValueError(msg)


def download_kin40k(
    *,
    directory: Path | None = None,
    force: bool = False,
    source_url: str = "https://github.com/trungngv/fgp/archive/refs/heads/master.tar.gz",
) -> Path:
    """
    Download the kin40k dataset from the fgp repository.

    Returns its local directory.
    """
    directory = directory if directory is not None else package_data_dir("kin40k")
    if directory.is_dir() and not force:
        return directory

    _require_http_url(source_url)
    directory.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        archive = tmp_path / "fgp-master.tar.gz"
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        urllib.request.urlretrieve(source_url, archive)  # noqa: S310
        with tarfile.open(archive) as tar:
            tar.extractall(extract_dir, filter="data")

        source_dir = extract_dir / "fgp-master" / "data" / "kin40k"
        if not source_dir.is_dir():
            msg = "Downloaded archive did not contain data/kin40k"
            raise FileNotFoundError(msg)

        if directory.is_dir():
            if not force:
                msg = f"Destination already exists: {directory}"
                raise FileExistsError(msg)
            shutil.rmtree(directory)

        shutil.copytree(source_dir, directory)

    return directory


def download_uci_regression_datasets(
    *,
    directory: Path | None = None,
    force: bool = False,
    source_url: str = "https://github.com/treforevans/uci_datasets/archive/refs/heads/master.tar.gz",
) -> Path:
    """
    Download the standardized UCI regression datasets from ``treforevans/uci_datasets``.

    The corpus contains 10 fixed train/test masks per dataset, stored as
    gzipped CSV files. Returns the local directory containing the dataset
    folders.
    """
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

        urllib.request.urlretrieve(source_url, archive)  # noqa: S310
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


def load_uci_regression_dataset(
    dataset: str,
    *,
    split: int = 1,
    directory: Path | None = None,
) -> UCIRegressionDataset:
    """
    Load one standardized UCI regression train/test split.

    ``split`` is one-based and must be in ``1..5``. Each of these 5 splits
    merges a consecutive pair of the 10 raw ``uci_datasets`` mask columns —
    split 1 unions raw masks 1+2, split 2 unions raw masks 3+4, and so on —
    so the test set is the union of both raw test masks (~20% of the data)
    and the train set is everything else. This halves the number of splits
    but doubles the test-set size, reducing the split-to-split metric
    variance that the raw 10-way 90/10 masks are prone to on small datasets.
    """
    if not 1 <= split <= NUM_UCI_SPLITS:
        msg = f"split must be in 1..{NUM_UCI_SPLITS} (merged pairs of the 10 raw splits)."
        raise ValueError(msg)

    directory = directory if directory is not None else package_data_dir("uci_datasets")
    name = _canonical_uci_dataset_name(dataset)
    data_root = _find_uci_data_root(directory, name)
    if data_root is None:
        msg = (
            f"Could not find UCI regression data under {directory}. "
            f"Run download_uci_regression_datasets(directory={directory!r}) first."
        )
        raise FileNotFoundError(msg)

    dataset_dir = data_root / name
    data_path = dataset_dir / "data.csv.gz"
    mask_path = dataset_dir / "test_mask.csv.gz"
    if not data_path.is_file():
        msg = f"Missing data file: {data_path}"
        raise FileNotFoundError(msg)
    if not mask_path.is_file():
        msg = f"Missing split mask file: {mask_path}"
        raise FileNotFoundError(msg)

    data = _read_gzip_csv(data_path, np.float64)
    if name == "song":
        data1_path = dataset_dir / "data1.csv.gz"
        if not data1_path.is_file():
            msg = f"Missing second song data file: {data1_path}"
            raise FileNotFoundError(msg)
        data = np.vstack([data, _read_gzip_csv(data1_path, np.float64)])
    masks = _read_gzip_csv(mask_path, np.int64)

    if data.shape[1] < MIN_DATA_COLUMNS:
        msg = f"Expected at least one feature and one target column in {data_path}"
        raise ValueError(msg)
    if masks.shape[0] != data.shape[0]:
        msg = f"Mask row count does not match data row count for {name}"
        raise ValueError(msg)
    raw_col_a, raw_col_b = 2 * (split - 1), 2 * (split - 1) + 1
    if masks.shape[1] < raw_col_b + 1:
        msg = (
            f"Requested split {split} (raw mask columns {raw_col_a + 1}+{raw_col_b + 1}), "
            f"but {mask_path} only has {masks.shape[1]} raw split columns"
        )
        raise ValueError(msg)

    test_mask = masks[:, raw_col_a].astype(bool) | masks[:, raw_col_b].astype(bool)
    train_mask = ~test_mask
    x = data[:, :-1]
    y = data[:, -1:]  # keep 2-D, matching gpjax's Dataset shape convention

    return UCIRegressionDataset(
        x_train=x[train_mask],
        y_train=y[train_mask],
        x_test=x[test_mask],
        y_test=y[test_mask],
        name=name,
        split=split,
    )


def _canonical_uci_dataset_name(dataset: str) -> str:
    key = re.sub(r"[\s_]", "", dataset.lower())
    name = UCI_REGRESSION_DATASET_ALIASES.get(key, key)
    if name not in UCI_REGRESSION_DATASET_SIZES:
        available = ", ".join(uci_regression_datasets())
        msg = f"Unknown UCI regression dataset {dataset!r}. Available: {available}"
        raise ValueError(msg)
    return name


def _find_uci_data_root(directory: Path, dataset: str = "yacht") -> Path | None:
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
