"""Unsupervised clustering for acoustic embeddings."""

from __future__ import annotations

import logging
from collections import Counter

import hdbscan
import numpy as np
import umap
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.schemas import AudioEvent, ClusterSummary

LOGGER = logging.getLogger(__name__)


def compute_representatives(
    embeddings_space: np.ndarray,
    events: list[AudioEvent],
    labels: np.ndarray,
    max_representatives: int = 5,
) -> dict[int, list[str]]:
    """Choose representative events closest to each cluster centroid."""
    representatives: dict[int, list[str]] = {}
    for label in sorted(int(x) for x in set(labels) if int(x) >= 0):
        indices = np.flatnonzero(labels == label)
        if indices.size == 0:
            continue
        cluster_points = embeddings_space[indices]
        centroid = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        chosen = indices[np.argsort(distances)[:max_representatives]]
        representatives[label] = [events[int(i)].event_id for i in chosen]
    return representatives


def _assign_labels(events: list[AudioEvent], labels: np.ndarray, probabilities: np.ndarray | None) -> None:
    for idx, event in enumerate(events):
        label = int(labels[idx])
        event.cluster_id = None if label < 0 else label
        event.is_noise = label < 0
        if probabilities is not None and idx < len(probabilities):
            event.cluster_probability = float(probabilities[idx])
        else:
            event.cluster_probability = None if label < 0 else 1.0


def _summaries(events: list[AudioEvent], labels: np.ndarray, representatives: dict[int, list[str]]) -> list[ClusterSummary]:
    summaries: list[ClusterSummary] = []
    counts = Counter(int(x) for x in labels if int(x) >= 0)
    for label in sorted(counts):
        cluster_events = [event for event in events if event.cluster_id == label]
        probabilities = [
            event.cluster_probability
            for event in cluster_events
            if event.cluster_probability is not None
        ]
        purity_scores = [
            event.purity_score
            for event in cluster_events
            if event.purity_score is not None
        ]
        mean_probability = float(np.mean(probabilities)) if probabilities else None
        mean_purity_score = float(np.mean(purity_scores)) if purity_scores else None
        summaries.append(
            ClusterSummary(
                cluster_id=label,
                size=counts[label],
                folder_name=f"cluster_{label:03d}_size_{counts[label]:03d}",
                mean_probability=mean_probability,
                representative_event_ids=representatives.get(label, []),
                mean_purity_score=mean_purity_score,
                n_component_events=sum(1 for event in cluster_events if event.is_component),
                n_original_events=sum(1 for event in cluster_events if event.source_type == "original"),
            )
        )
    return summaries


def _fallback_cluster(scaled: np.ndarray, config: BioSoundConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_events = scaled.shape[0]
    n_components = min(max(2, min(config.umap_components, n_events - 1)), scaled.shape[1])
    reduced = PCA(n_components=n_components, random_state=config.random_state).fit_transform(scaled)
    try:
        labels = DBSCAN(eps=2.0, min_samples=max(2, config.min_samples or config.min_cluster_size // 2)).fit_predict(reduced)
        if len(set(labels)) <= 1 and n_events >= config.min_cluster_size:
            n_clusters = max(1, min(4, n_events // config.min_cluster_size))
            labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(reduced)
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Fallback clustering failed; assigning one cluster: %s", exc)
        labels = np.zeros(n_events, dtype=int)
    probabilities = np.where(labels >= 0, 1.0, 0.0).astype(float)
    return labels.astype(int), probabilities, reduced


def cluster_embeddings(
    embeddings: np.ndarray,
    events: list[AudioEvent],
    config: BioSoundConfig,
) -> tuple[list[AudioEvent], list[ClusterSummary], np.ndarray]:
    """Cluster event embeddings with UMAP + HDBSCAN, with robust fallbacks."""
    if len(events) == 0:
        return events, [], np.empty((0, 0), dtype=np.float32)

    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    if len(events) <= max(4, config.min_cluster_size):
        labels = np.zeros(len(events), dtype=int)
        probabilities = np.ones(len(events), dtype=float)
        reduced = scaled
    else:
        try:
            n_neighbors = min(config.umap_neighbors, len(events) - 1)
            n_components = min(config.umap_components, max(2, len(events) - 2), embeddings.shape[1])
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                metric=config.umap_metric,
                min_dist=config.umap_min_dist,
                random_state=config.random_state,
            )
            reduced = reducer.fit_transform(scaled)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=config.min_cluster_size,
                min_samples=config.min_samples or max(2, config.min_cluster_size // 2),
                cluster_selection_method="eom",
                prediction_data=True,
            )
            labels = clusterer.fit_predict(reduced).astype(int)
            probabilities = getattr(clusterer, "probabilities_", np.ones(len(events), dtype=float))
            if np.all(labels < 0):
                LOGGER.warning("HDBSCAN marked all events as noise; using fallback clustering.")
                labels, probabilities, reduced = _fallback_cluster(scaled, config)
        except Exception as exc:
            LOGGER.warning("UMAP/HDBSCAN failed; using fallback clustering: %s", exc)
            labels, probabilities, reduced = _fallback_cluster(scaled, config)

    _assign_labels(events, labels, probabilities)
    representatives = compute_representatives(reduced, events, labels)
    summaries = _summaries(events, labels, representatives)
    return events, summaries, reduced
