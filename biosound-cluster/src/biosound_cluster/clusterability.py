"""Clusterability scoring and routing for candidate acoustic events."""

from __future__ import annotations

from dataclasses import replace

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.embeddings import compute_logmel_embedding, extract_event_clip
from biosound_cluster.schemas import AudioEvent


def analyze_and_route_clusterability(
    audio: np.ndarray,
    sr: int,
    events: list[AudioEvent],
    config: BioSoundConfig,
) -> tuple[list[AudioEvent], list[AudioEvent]]:
    """
    Score whether events are stable and structured enough for normal clustering.

    Events below the clustering threshold are kept for review with full metadata;
    the audio is never deleted or replaced.
    """
    if not events or not config.enable_clusterability_filtering:
        return events, []

    clusterable: list[AudioEvent] = []
    review: list[AudioEvent] = []
    for event in events:
        attach_clusterability_scores(audio, sr, event, config)
        score = event.clusterability_score or 0.0
        if score >= config.min_clusterability_for_clustering:
            clusterable.append(event)
        else:
            review.append(_mark_clusterability_review(event, config))
    return clusterable, review


def attach_clusterability_scores(
    audio: np.ndarray,
    sr: int,
    event: AudioEvent,
    config: BioSoundConfig,
) -> None:
    """Attach clusterability component scores to one event."""
    local_snr = _local_snr_score(event, config)
    structure = _spectral_structure_score(event)
    duration = _duration_confidence(event.duration_sec, config)
    stability = (
        compute_embedding_stability_score(audio, sr, event, config)
        if config.enable_embedding_stability
        else None
    )
    stability_for_score = 0.75 if stability is None else stability
    broadband_penalty = _broadband_noise_penalty(event)
    overlap_penalty = _overlap_penalty(event)
    edge_penalty = _edge_case_penalty(event, config)

    event.local_snr_score = local_snr
    event.spectral_structure_score = structure
    event.duration_confidence = duration
    event.embedding_stability_score = stability
    event.broadband_noise_penalty = broadband_penalty
    event.overlap_penalty = overlap_penalty
    event.edge_case_penalty = edge_penalty
    score = (
        0.24 * _value(event.eventness_score, 0.55)
        + 0.22 * local_snr
        + 0.20 * structure
        + 0.14 * duration
        + 0.14 * stability_for_score
        - 0.18 * broadband_penalty
        - 0.08 * overlap_penalty
        - 0.06 * edge_penalty
    )
    if stability is not None and stability < config.embedding_stability_min_for_clustering:
        score -= 0.08 * (1.0 - stability)
    event.clusterability_score = float(np.clip(score, 0.0, 1.0))


def compute_embedding_stability_score(
    audio: np.ndarray,
    sr: int,
    event: AudioEvent,
    config: BioSoundConfig,
) -> float:
    """Measure embedding consistency across non-destructive auxiliary views."""
    clip = extract_event_clip(audio, sr, event)
    if clip.size == 0:
        return 0.0

    views = _build_embedding_views(clip, sr, event, config)
    vectors: list[np.ndarray] = []
    for view in views:
        if view.size == 0:
            continue
        vectors.append(compute_logmel_embedding(view.astype(np.float32), sr, config))
    if len(vectors) < 2:
        return 0.75

    base = vectors[0]
    similarities = [_cosine_similarity(base, vector) for vector in vectors[1:]]
    return float(np.clip(np.mean(similarities), 0.0, 1.0)) if similarities else 0.75


def _build_embedding_views(
    clip: np.ndarray,
    sr: int,
    event: AudioEvent,
    config: BioSoundConfig,
) -> list[np.ndarray]:
    views: list[np.ndarray] = []
    for name in config.embedding_stability_views:
        if name == "original":
            views.append(clip)
        elif name == "center_crop":
            views.append(_center_crop(clip, keep_fraction=0.70))
        elif name == "band_limited":
            views.append(_band_limited_view(clip, sr, event))
    return views or [clip]


def _center_crop(clip: np.ndarray, keep_fraction: float) -> np.ndarray:
    if clip.size < 8:
        return clip
    keep = max(8, int(round(clip.size * keep_fraction)))
    start = max(0, (clip.size - keep) // 2)
    return clip[start : start + keep]


def _band_limited_view(clip: np.ndarray, sr: int, event: AudioEvent) -> np.ndarray:
    if clip.size < 32:
        return clip
    center = event.spectral_centroid if event.spectral_centroid and event.spectral_centroid > 0 else sr / 4.0
    nyquist = sr / 2.0
    low = max(80.0, center / 3.0)
    high = min(nyquist * 0.96, center * 3.0)
    if high <= low:
        low, high = 80.0, nyquist * 0.96
    try:
        sos = butter(3, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
        if clip.size <= 3 * sos.shape[0] * 2:
            return clip
        return sosfiltfilt(sos, clip).astype(np.float32)
    except ValueError:
        return clip


def _mark_clusterability_review(event: AudioEvent, config: BioSoundConfig) -> AudioEvent:
    review = replace(event)
    review.cluster_id = None
    review.cluster_probability = None
    review.is_noise = True
    review.is_low_confidence_noise = review.clusterability_score is not None and (
        review.clusterability_score < config.min_clusterability_for_review
    )
    review.is_ambiguous_review = not review.is_low_confidence_noise
    review.source_type = "low_confidence_noise" if review.is_low_confidence_noise else "ambiguous_review"
    reason = (
        "clusterability:low"
        if review.is_low_confidence_noise
        else "clusterability:ambiguous"
    )
    review.candidate_route_reason = _append_reason(review.candidate_route_reason, reason)
    return review


def _append_reason(existing: str | None, reason: str) -> str:
    return reason if not existing else f"{existing};{reason}"


def _local_snr_score(event: AudioEvent, config: BioSoundConfig) -> float:
    snr = max(_value(event.snr_db, 0.0), _value(event.peak_band_snr_db, 0.0))
    return _smoothstep(config.noise_min_snr_db, config.noise_min_snr_db + 18.0, snr)


def _spectral_structure_score(event: AudioEvent) -> float:
    tonality = _value(event.tonality_score, 0.35)
    flatness_score = 1.0 - _value(event.spectral_flatness, 0.5)
    peak_band = _smoothstep(0.0, 18.0, _value(event.peak_band_snr_db, 6.0))
    return float(np.clip(0.45 * tonality + 0.35 * flatness_score + 0.20 * peak_band, 0.0, 1.0))


def _duration_confidence(duration_sec: float, config: BioSoundConfig) -> float:
    short = _smoothstep(config.min_event_duration * 0.6, max(config.min_event_duration * 2.0, 0.12), duration_sec)
    long = 1.0 - _smoothstep(config.max_event_duration * 0.75, config.max_event_duration, duration_sec)
    return float(np.clip(short * long, 0.0, 1.0))


def _broadband_noise_penalty(event: AudioEvent) -> float:
    flatness = _value(event.spectral_flatness, 0.0)
    bandwidth = _value(event.bandwidth_hz, 0.0)
    centroid = max(_value(event.spectral_centroid, 0.0), 1.0)
    bandwidth_ratio = float(np.clip(bandwidth / max(centroid * 2.0, 1.0), 0.0, 1.0))
    low_tonality = 1.0 - _value(event.tonality_score, 0.5)
    return float(np.clip(0.50 * flatness + 0.30 * bandwidth_ratio + 0.20 * low_tonality, 0.0, 1.0))


def _overlap_penalty(event: AudioEvent) -> float:
    if event.is_mixed:
        return 1.0
    base = 0.4 if event.is_overlapping else 0.0
    polyphony = _value(event.polyphony_score, 0.0)
    purity_penalty = 1.0 - _value(event.purity_score, 1.0)
    return float(np.clip(max(base, 0.5 * polyphony, 0.5 * purity_penalty), 0.0, 1.0))


def _edge_case_penalty(event: AudioEvent, config: BioSoundConfig) -> float:
    penalty = 0.0
    if event.duration_sec < config.min_review_event_duration:
        penalty += 0.45
    if event.duration_sec > config.max_event_duration * 0.90:
        penalty += 0.35
    if event.active_ratio is not None and event.active_ratio > config.eventness_max_active_ratio:
        penalty += 0.25
    return float(np.clip(penalty, 0.0, 1.0))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float((np.dot(a, b) / denom + 1.0) / 2.0)


def _value(value: float | None, default: float) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(value)


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    x = float(np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)
