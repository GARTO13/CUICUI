"""Event usefulness scoring for candidate acoustic events."""

from __future__ import annotations

from dataclasses import dataclass, replace

import librosa
import numpy as np

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.embeddings import extract_event_clip
from biosound_cluster.schemas import AudioEvent


@dataclass(slots=True)
class EventnessFeatures:
    """Temporal salience features for deciding whether a candidate should enter clustering."""

    eventness_score: float
    temporal_contrast_db: float
    active_ratio: float


def analyze_and_route_eventness(
    audio: np.ndarray,
    sr: int,
    events: list[AudioEvent],
    config: BioSoundConfig,
) -> tuple[list[AudioEvent], list[AudioEvent]]:
    """
    Split events into clusterable and low-confidence candidates using unsupervised salience.

    Low-confidence candidates are not deleted. They are routed to the same review bucket as
    low-confidence noise so they do not pollute normal acoustic-family clusters.
    """
    if not events or not config.enable_eventness_filtering:
        return events, []

    clusterable: list[AudioEvent] = []
    low_confidence: list[AudioEvent] = []
    for event in events:
        features = compute_eventness_features(audio, sr, event, config)
        _attach_eventness(event, features)
        threshold = (
            config.min_component_eventness_for_clustering
            if event.is_component
            else config.min_eventness_for_clustering
        )
        if is_low_confidence_event(features, threshold, config):
            low_confidence.append(_mark_low_confidence_event(event, features))
        else:
            clusterable.append(event)
    return clusterable, low_confidence


def compute_eventness_features(
    audio: np.ndarray,
    sr: int,
    event: AudioEvent,
    config: BioSoundConfig,
) -> EventnessFeatures:
    """Compute an eventness score from envelope contrast and temporal occupancy."""
    clip = extract_event_clip(audio, sr, event)
    if clip.size == 0:
        return EventnessFeatures(0.0, 0.0, 1.0)

    n_fft = min(config.frame_length, max(256, 2 ** int(np.floor(np.log2(max(256, clip.size))))))
    hop_length = min(config.hop_length, max(64, n_fft // 4))
    rms = librosa.feature.rms(y=clip, frame_length=n_fft, hop_length=hop_length)[0]
    if rms.size == 0:
        return EventnessFeatures(0.0, 0.0, 1.0)

    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0, top_db=None)
    floor_db = float(np.percentile(rms_db, 20))
    peak_db = float(np.percentile(rms_db, 95))
    contrast_db = max(0.0, peak_db - floor_db)
    active = rms_db >= floor_db + config.eventness_min_contrast_db
    active_ratio = float(np.count_nonzero(active) / len(active)) if len(active) else 1.0

    contrast_score = _smoothstep(
        config.eventness_min_contrast_db,
        config.eventness_min_contrast_db + 14.0,
        contrast_db,
    )
    occupancy_score = 1.0 - _smoothstep(0.55, config.eventness_max_active_ratio, active_ratio)
    duration_score = _duration_score(event.duration_sec, config)
    score = 0.55 * contrast_score + 0.25 * occupancy_score + 0.20 * duration_score
    if event.is_component and event.purity_score is not None:
        score = 0.75 * score + 0.25 * float(np.clip(event.purity_score, 0.0, 1.0))
    return EventnessFeatures(
        eventness_score=float(np.clip(score, 0.0, 1.0)),
        temporal_contrast_db=float(contrast_db),
        active_ratio=float(np.clip(active_ratio, 0.0, 1.0)),
    )


def is_low_confidence_event(
    features: EventnessFeatures,
    threshold: float,
    config: BioSoundConfig,
) -> bool:
    """Return true for weak, low-contrast, or nearly continuous candidates."""
    continuous_low_contrast = (
        features.active_ratio > config.eventness_max_active_ratio
        and features.temporal_contrast_db < config.eventness_min_contrast_db + 3.0
    )
    return features.eventness_score < threshold or continuous_low_contrast


def _attach_eventness(event: AudioEvent, features: EventnessFeatures) -> None:
    event.eventness_score = features.eventness_score
    event.temporal_contrast_db = features.temporal_contrast_db
    event.active_ratio = features.active_ratio


def _mark_low_confidence_event(event: AudioEvent, features: EventnessFeatures) -> AudioEvent:
    low = replace(event)
    _attach_eventness(low, features)
    low.cluster_id = None
    low.cluster_probability = None
    low.is_noise = True
    low.is_low_confidence_noise = True
    low.is_low_confidence_event = True
    low.source_type = "low_confidence_noise"
    return low


def _duration_score(duration_sec: float, config: BioSoundConfig) -> float:
    short = _smoothstep(config.min_event_duration * 0.8, config.min_event_duration * 2.0, duration_sec)
    long = 1.0 - _smoothstep(config.max_event_duration * 0.75, config.max_event_duration, duration_sec)
    return float(np.clip(short * long, 0.0, 1.0))


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    x = float(np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)
