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


def bridge_region_nodes(graph: nx.Graph, edge: tuple[int, int]) -> set[int]:
    """Nodes in the smaller component obtained by cutting a bridge edge.

    `edge` must be a bridge (its removal disconnects the graph) -- used to isolate
    a single, graph-contiguous road segment as a "localized region" for
    misspecification testing, e.g. simulating an accident affecting one road.

    Only ever isolates small dead-end spurs: this graph's sensor-bearing roads sit
    in one 876-node/300-sensor biconnected core (opposite carriageways, ramps, and
    frontage roads all give an alternate path around any single edge), so no bridge
    cut can carve out a stretch of an actual highway corridor -- see
    `road_segment_nodes` for that.
    """
    without_edge = graph.copy()
    without_edge.remove_edge(*edge)
    components = list(nx.connected_components(without_edge))
    return {int(v) for v in min(components, key=len)}


def road_segment_nodes(
    graph: nx.Graph, seed_node: int, radius_m: float, *, weight: str = "length"
) -> set[int]:
    """Nodes within `radius_m` driving distance of `seed_node`.

    A physically meaningful "stretch of road" for misspecification testing: unlike
    `bridge_region_nodes` (which only reaches small dead-end spurs), this follows
    real road distance (edge `length` in metres) outward from a seed, so it traces
    a contiguous corridor along the actual highway network -- e.g. a segment of
    I-680 or US-101 -- rather than requiring a graph-disconnecting cut.
    """
    distances = nx.single_source_dijkstra_path_length(graph, seed_node, weight=weight)
    return {int(n) for n, d in distances.items() if d <= radius_m}


def apply_regime_shift(
    data: PemsGraphData, affected_nodes: set[int], shift_factor: float
) -> PemsGraphData:
    """Multiply the speed of every sensor whose node is in `affected_nodes` by
    `shift_factor` (e.g. 0.6 for a 40% congestion-scale slowdown).

    Simulates a persistent, localized event (e.g. an accident) that a smooth graph
    kernel isn't built to represent as a sharp boundary. Multiplicative rather than
    an additive mph offset so it stays physically plausible (speeds can't go
    negative) regardless of a region's baseline speed -- an additive -25mph shift
    is a mild congestion-scale change on a ~64mph free-flowing segment but can push
    an already-slow ~27mph arterial segment negative.
    """
    affected_mask = np.isin(data.node_index.reshape(-1), list(affected_nodes))
    speed = data.speed.copy()
    speed[affected_mask] *= shift_factor
    return data._replace(speed=speed)


class PemsRegressionSplit(NamedTuple):
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    split: int
    train_affected: np.ndarray
    test_affected: np.ndarray


def pems_regression_split(
    data: PemsGraphData,
    split: int,
    *,
    num_train: int = 250,
    affected_nodes: set[int] | None = None,
) -> PemsRegressionSplit:
    """Random train/test split of the 325 sensors, reproducible via `split`.

    `affected_nodes`, if given, marks sensors inside a misspecification-test region
    (see `bridge_region_nodes`/`apply_regime_shift`) via the returned boolean
    `train_affected`/`test_affected` masks; all-`False` when `None`.
    """
    rng = np.random.default_rng(split)
    perm = rng.permutation(data.num_sensors)
    train_idx, test_idx = perm[:num_train], perm[num_train:]

    if affected_nodes:
        affected = np.isin(data.node_index.reshape(-1), list(affected_nodes))
    else:
        affected = np.zeros(data.num_sensors, dtype=bool)

    return PemsRegressionSplit(
        x_train=data.node_index[train_idx],
        y_train=data.speed[train_idx],
        x_test=data.node_index[test_idx],
        y_test=data.speed[test_idx],
        split=split,
        train_affected=affected[train_idx],
        test_affected=affected[test_idx],
    )
