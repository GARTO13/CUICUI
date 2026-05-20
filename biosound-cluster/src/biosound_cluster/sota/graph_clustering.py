"""k-NN graph construction + Leiden clustering with stability scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GraphClusteringResult:
    """Output of one Leiden run on a k-NN graph."""

    labels: np.ndarray  # (n,) int — -1 means filtered noise
    stability: np.ndarray  # (n,) float in [0,1]
    n_clusters: int


def _build_knn(
    embeddings: np.ndarray,
    k: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    n = embeddings.shape[0]
    k_eff = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    return distances, indices


def _distances_to_weights(distances: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return np.clip(1.0 - distances, 0.0, 2.0)
    return 1.0 / (1.0 + distances)


def _leiden_partition(
    n: int,
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    resolution: float,
    n_iterations: int,
    seed: int,
) -> np.ndarray:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise ImportError(
            "Leiden clustering requires python-igraph and leidenalg. "
            "Install with: pip install 'biosound-cluster[sota]'"
        ) from exc

    edges = [(int(a), int(b)) for a, b in edge_index]
    graph = ig.Graph(n=n, edges=edges, directed=False)
    graph.es["weight"] = edge_weights.tolist()
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        n_iterations=int(n_iterations),
        seed=int(seed),
    )
    return np.asarray(partition.membership, dtype=np.int64)


def _edges_from_knn(indices: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, k_plus_one = indices.shape
    src = np.repeat(np.arange(n), k_plus_one - 1)
    dst = indices[:, 1:].reshape(-1)
    w = weights[:, 1:].reshape(-1)
    mask = src != dst
    return np.stack([src[mask], dst[mask]], axis=1), w[mask]


def cluster_embeddings(
    embeddings: np.ndarray,
    knn_neighbors: int,
    metric: str,
    resolution: float,
    n_iterations: int,
    seed: int,
    min_cluster_size: int,
    stability_subsamples: int,
    stability_subsample_fraction: float,
) -> GraphClusteringResult:
    """Run Leiden on a k-NN graph and score per-point stability via subsampling."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2-D")
    n = embeddings.shape[0]
    if n == 0:
        return GraphClusteringResult(
            labels=np.array([], dtype=np.int64),
            stability=np.array([], dtype=np.float32),
            n_clusters=0,
        )
    if n == 1:
        return GraphClusteringResult(
            labels=np.array([-1], dtype=np.int64),
            stability=np.array([0.0], dtype=np.float32),
            n_clusters=0,
        )

    distances, indices = _build_knn(embeddings, knn_neighbors, metric)
    weights = _distances_to_weights(distances, metric)
    edge_index, edge_weights = _edges_from_knn(indices, weights)
    raw_labels = _leiden_partition(
        n=n,
        edge_index=edge_index,
        edge_weights=edge_weights,
        resolution=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )

    cluster_sizes = np.bincount(raw_labels)
    keep_mask = cluster_sizes[raw_labels] >= min_cluster_size
    labels = np.where(keep_mask, raw_labels, -1).astype(np.int64)

    unique = sorted({int(label) for label in labels if label >= 0})
    relabel = {old: new for new, old in enumerate(unique)}
    labels = np.array(
        [relabel[int(label)] if int(label) >= 0 else -1 for label in labels],
        dtype=np.int64,
    )

    stability = _stability_via_subsampling(
        embeddings=embeddings,
        base_labels=labels,
        knn_neighbors=knn_neighbors,
        metric=metric,
        resolution=resolution,
        n_iterations=n_iterations,
        seed=seed,
        n_subsamples=stability_subsamples,
        subsample_fraction=stability_subsample_fraction,
        min_cluster_size=min_cluster_size,
    )
    return GraphClusteringResult(
        labels=labels,
        stability=stability,
        n_clusters=len(unique),
    )


def _stability_via_subsampling(
    embeddings: np.ndarray,
    base_labels: np.ndarray,
    knn_neighbors: int,
    metric: str,
    resolution: float,
    n_iterations: int,
    seed: int,
    n_subsamples: int,
    subsample_fraction: float,
    min_cluster_size: int,
) -> np.ndarray:
    n = embeddings.shape[0]
    if n_subsamples <= 0 or n < 8:
        return np.where(base_labels >= 0, 1.0, 0.0).astype(np.float32)

    rng = np.random.default_rng(seed)
    counts = np.zeros(n, dtype=np.int32)
    agreement = np.zeros(n, dtype=np.float32)
    sub_size = max(8, int(round(subsample_fraction * n)))

    for sub_idx in range(n_subsamples):
        sample = rng.choice(n, size=sub_size, replace=False)
        sub_embed = embeddings[sample]
        distances, indices = _build_knn(sub_embed, knn_neighbors, metric)
        weights = _distances_to_weights(distances, metric)
        edge_index, edge_weights = _edges_from_knn(indices, weights)
        sub_labels = _leiden_partition(
            n=sub_embed.shape[0],
            edge_index=edge_index,
            edge_weights=edge_weights,
            resolution=resolution,
            n_iterations=n_iterations,
            seed=int(seed + sub_idx + 1),
        )
        sizes = np.bincount(sub_labels)
        sub_labels = np.where(sizes[sub_labels] >= min_cluster_size, sub_labels, -1)
        agreement_sub = _co_assignment_agreement(base_labels[sample], sub_labels)
        counts[sample] += 1
        agreement[sample] += agreement_sub.astype(np.float32)

    stability = np.where(counts > 0, agreement / np.maximum(counts, 1), 0.0).astype(np.float32)
    stability[base_labels < 0] = 0.0
    return stability


def _co_assignment_agreement(base: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """For each point i, fraction of other points whose (base==base_i) <=> (sub==sub_i)."""
    n = base.shape[0]
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    agreement = np.empty(n, dtype=np.float32)
    base_eq = base[:, None] == base[None, :]
    sub_eq = sub[:, None] == sub[None, :]
    match = base_eq == sub_eq
    np.fill_diagonal(match, False)
    agreement = match.sum(axis=1) / float(n - 1)
    return agreement.astype(np.float32)
