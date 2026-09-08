import pickle
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple

import networkx as nx
import numpy as np


def package_data_dir(*parts: str) -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root.joinpath("data", *parts)


def _require_http_url(url: str) -> None:
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        msg = f"Unsupported URL scheme: {scheme!r}"
        raise ValueError(msg)


_PEMS_SOURCE_URL = (
    "https://raw.githubusercontent.com/vabor112/pems-regression/main/"
    "pems_regression/resources/processed_pems_data.pkl"
)


def download_pems_regression_data(
    *,
    directory: Path | None = None,
    force: bool = False,
    source_url: str = _PEMS_SOURCE_URL,
) -> Path:
    """Download and cache the pre-processed PeMS-Bay road-network pickle.

    Contains a ``networkx.Graph`` (1016 nodes / 1173 edges, San Jose road
    network) and the Monday-17:30 traffic-speed reading at 325 sensor nodes,
    from Borovitskiy et al. (2021), "Matern Gaussian Processes on Graphs".
    """
    directory = directory if directory is not None else package_data_dir("pems")
    path = directory / "processed_pems_data.pkl"
    if path.is_file() and not force:
        return path

    _require_http_url(source_url)
    directory.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(source_url, path)
    return path


class PemsGraphData(NamedTuple):
    graph: nx.Graph
    laplacian: np.ndarray
    node_index: np.ndarray
    speed: np.ndarray
    num_nodes: int
    num_sensors: int


def load_pems_graph_data(*, force: bool = False) -> PemsGraphData:
    path = download_pems_regression_data(force=force)
    with path.open("rb") as f:
        graph, (node_index, speed) = pickle.load(f)  # noqa: S301

    # Explicit nodelist so row/col `v` always means "vertex labelled `v`",
    # matching the integer labels used to index into node_index -- rather than
    # relying on whatever order graph.nodes() happens to iterate in.
    nodelist = list(range(graph.number_of_nodes()))
    laplacian = nx.laplacian_matrix(graph, nodelist=nodelist, weight="weight").toarray()
    node_index = np.asarray(node_index, dtype=np.int64).reshape(-1, 1)
    speed = np.asarray(speed, dtype=np.float64).reshape(-1, 1)

    return PemsGraphData(
        graph=graph,
        laplacian=laplacian,
        node_index=node_index,
        speed=speed,
        num_nodes=laplacian.shape[0],
        num_sensors=node_index.shape[0],
    )


class PemsRegressionSplit(NamedTuple):
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    split: int


def pems_regression_split(
    data: PemsGraphData,
    split: int,
    *,
    num_train: int = 250,
    seed: int | None = None,
) -> PemsRegressionSplit:
    """Random train/test split of the 325 sensors.

    Reproducible via `seed` (defaults to `split` when not given). `seed` is
    decoupled from `split` so callers can hold the output directory labeling
    (`split`) fixed while generating a different partition (`seed`) -- e.g. to
    rerun the same nominal splits with an independent set of random partitions.
    """
    rng = np.random.default_rng(seed if seed is not None else split)
    perm = rng.permutation(data.num_sensors)
    train_idx, test_idx = perm[:num_train], perm[num_train:]

    return PemsRegressionSplit(
        x_train=data.node_index[train_idx],
        y_train=data.speed[train_idx],
        x_test=data.node_index[test_idx],
        y_test=data.speed[test_idx],
        split=split,
    )
