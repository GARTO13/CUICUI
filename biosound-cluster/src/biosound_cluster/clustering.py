"""Unsupervised clustering for acoustic embeddings."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace

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
    """Choose clean, typical representatives for each cluster."""
    representatives: dict[int, list[str]] = {}
    for label in sorted(int(x) for x in set(labels) if int(x) >= 0):
        indices = np.flatnonzero(labels == label)
        if indices.size == 0:
            continue
        cluster_points = embeddings_space[indices]
        centroid = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        max_distance = float(np.max(distances)) if distances.size else 0.0
        centroid_similarity = 1.0 - (distances / max(max_distance, 1e-9))
        scores = []
        for pos, event_index in enumerate(indices):
            event = events[int(event_index)]
            score = _representative_score(event, float(centroid_similarity[pos]))
            event.representative_score = score
            scores.append(score)
        chosen = indices[np.argsort(scores)[::-1][:max_representatives]]
        representatives[label] = [events[int(i)].event_id for i in chosen]
    return representatives


def _representative_score(event: AudioEvent, centroid_similarity: float) -> float:
    eventness = _value(event.eventness_score, 0.55)
    local_snr = _value(event.local_snr_score, 0.55)
    stability = _value(event.embedding_stability_score, 0.70)
    duration = _value(event.duration_confidence, 0.65)
    overlap_penalty = _value(event.overlap_penalty, 0.0)
    score = (
        0.30 * centroid_similarity
        + 0.25 * eventness
        + 0.20 * local_snr
        + 0.15 * stability
        + 0.10 * duration
        - 0.20 * overlap_penalty
    )
    return float(np.clip(score, 0.0, 1.0))


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
        clusterability_scores = [
            event.clusterability_score
            for event in cluster_events
            if event.clusterability_score is not None
        ]
        stability_scores = [
            event.embedding_stability_score
            for event in cluster_events
            if event.embedding_stability_score is not None
        ]
        prefamilies = [event.acoustic_prefamily or "unknown" for event in cluster_events]
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
                mean_clusterability_score=float(np.mean(clusterability_scores)) if clusterability_scores else None,
                mean_stability_score=float(np.mean(stability_scores)) if stability_scores else None,
                acoustic_prefamily=Counter(prefamilies).most_common(1)[0][0] if prefamilies else "unknown",
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


def _cluster_embedding_block(
    embeddings: np.ndarray,
    config: BioSoundConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    n_events = embeddings.shape[0]

    if n_events <= max(4, config.min_cluster_size):
        return np.zeros(n_events, dtype=int), np.ones(n_events, dtype=float), scaled

    try:
        n_neighbors = min(config.umap_neighbors, n_events - 1)
        n_components = min(config.umap_components, max(2, n_events - 2), embeddings.shape[1])
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
        probabilities = getattr(clusterer, "probabilities_", np.ones(n_events, dtype=float))
        if np.all(labels < 0):
            LOGGER.warning("HDBSCAN marked all events as noise; using fallback clustering.")
            labels, probabilities, reduced = _fallback_cluster(scaled, config)
    except Exception as exc:
        LOGGER.warning("UMAP/HDBSCAN failed; using fallback clustering: %s", exc)
        labels, probabilities, reduced = _fallback_cluster(scaled, config)
    return labels.astype(int), probabilities.astype(float), reduced


def _prefamily_groups(events: list[AudioEvent], config: BioSoundConfig) -> list[np.ndarray]:
    if not config.enable_acoustic_prefamilies:
        return [np.arange(len(events), dtype=int)]
    groups: list[np.ndarray] = []
    small: list[int] = []
    family_names = sorted({event.acoustic_prefamily or "unknown" for event in events})
    for family in family_names:
        indices = np.asarray(
            [idx for idx, event in enumerate(events) if (event.acoustic_prefamily or "unknown") == family],
            dtype=int,
        )
        if indices.size >= config.prefamily_min_events:
            groups.append(indices)
        else:
            small.extend(indices.tolist())
    if small:
        small_indices = np.asarray(small, dtype=int)
        if small_indices.size >= config.prefamily_min_events or not groups:
            groups.append(small_indices)
        else:
            groups[-1] = np.concatenate([groups[-1], small_indices])
    return groups or [np.arange(len(events), dtype=int)]


def _assign_cluster_stability(events: list[AudioEvent], labels: np.ndarray, probabilities: np.ndarray) -> None:
    for idx, event in enumerate(events):
        if labels[idx] < 0:
            event.cluster_stability_score = 0.0
        else:
            event.cluster_stability_score = float(np.clip(probabilities[idx], 0.0, 1.0))


def _assign_ensemble_stability(
    embeddings: np.ndarray,
    events: list[AudioEvent],
    labels: np.ndarray,
    probabilities: np.ndarray,
    config: BioSoundConfig,
) -> None:
    """Estimate cluster stability from a small optional clustering ensemble."""
    if config.cluster_ensemble_runs <= 1:
        _assign_cluster_stability(events, labels, probabilities)
        return

    variants = _ensemble_configs(config)[: config.cluster_ensemble_runs]
    alt_labels: list[np.ndarray] = []
    for variant in variants:
        try:
            run_labels, _, _ = _cluster_embedding_block(embeddings, variant)
            alt_labels.append(run_labels)
        except Exception as exc:  # pragma: no cover - defensive optional path
            LOGGER.warning("Cluster ensemble run failed: %s", exc)

    if not alt_labels:
        _assign_cluster_stability(events, labels, probabilities)
        return

    for idx, event in enumerate(events):
        if labels[idx] < 0:
            event.cluster_stability_score = 0.0
            continue
        peers = np.flatnonzero(labels == labels[idx])
        peers = peers[peers != idx]
        if peers.size == 0:
            event.cluster_stability_score = float(np.clip(probabilities[idx], 0.0, 1.0))
            continue
        run_scores = []
        for run_labels in alt_labels:
            same = run_labels[peers] == run_labels[idx]
            run_scores.append(float(np.mean(same)) if same.size else 1.0)
        event.cluster_stability_score = float(np.clip(np.mean(run_scores), 0.0, 1.0))


def _ensemble_configs(config: BioSoundConfig) -> list[BioSoundConfig]:
    neighbor_values = [10, 15, 25, config.umap_neighbors]
    min_dist_values = [0.0, 0.05, config.umap_min_dist]
    min_cluster_values = [5, 8, 12, config.min_cluster_size]
    min_sample_values = [1, 3, 5, config.min_samples or max(2, config.min_cluster_size // 2)]
    variants: list[BioSoundConfig] = []
    seen: set[tuple[int, float, int, int]] = set()
    for idx in range(max(1, config.cluster_ensemble_runs)):
        params = (
            neighbor_values[idx % len(neighbor_values)],
            min_dist_values[idx % len(min_dist_values)],
            max(2, min_cluster_values[idx % len(min_cluster_values)]),
            max(1, min_sample_values[idx % len(min_sample_values)]),
        )
        if params in seen:
            continue
        seen.add(params)
        variants.append(
            replace(
                config,
                umap_neighbors=params[0],
                umap_min_dist=params[1],
                min_cluster_size=params[2],
                min_samples=params[3],
                cluster_ensemble_runs=1,
            )
        )
    return variants


def cluster_embeddings(
    embeddings: np.ndarray,
    events: list[AudioEvent],
    config: BioSoundConfig,
) -> tuple[list[AudioEvent], list[ClusterSummary], np.ndarray]:
    """Cluster event embeddings with UMAP + HDBSCAN, with robust fallbacks."""
    if len(events) == 0:
        return events, [], np.empty((0, 0), dtype=np.float32)

    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.full(len(events), -1, dtype=int)
    probabilities = np.zeros(len(events), dtype=float)
    reduced_parts: list[np.ndarray] = []
    next_label = 0
    for indices in _prefamily_groups(events, config):
        block_labels, block_probabilities, block_reduced = _cluster_embedding_block(embeddings[indices], config)
        unique_positive = sorted(int(label) for label in set(block_labels) if int(label) >= 0)
        label_map = {label: next_label + pos for pos, label in enumerate(unique_positive)}
        remapped = np.asarray([label_map.get(int(label), -1) for label in block_labels], dtype=int)
        labels[indices] = remapped
        probabilities[indices] = block_probabilities
        next_label += len(unique_positive)
        padded = np.zeros((len(events), block_reduced.shape[1]), dtype=np.float32)
        padded[indices] = block_reduced.astype(np.float32)
        reduced_parts.append(padded)
    reduced = np.concatenate(reduced_parts, axis=1) if reduced_parts else embeddings

    _assign_labels(events, labels, probabilities)
    _assign_ensemble_stability(embeddings, events, labels, probabilities, config)
    representatives = compute_representatives(reduced, events, labels)
    summaries = _summaries(events, labels, representatives)
    return events, summaries, reduced


def _value(value: float | None, default: float) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(value)
