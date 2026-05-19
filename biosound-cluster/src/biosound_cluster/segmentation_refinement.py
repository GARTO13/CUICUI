"""Temporal refinement for broad candidate acoustic events."""

from __future__ import annotations

from dataclasses import replace

import librosa
import numpy as np
from scipy.ndimage import binary_closing

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.schemas import AudioEvent
from biosound_cluster.segmentation import compute_event_stats


def refine_candidate_events(
    audio: np.ndarray,
    sr: int,
    events: list[AudioEvent],
    config: BioSoundConfig,
) -> list[AudioEvent]:
    """
    Refine broad detections by tightening temporal boundaries and removing duplicates.

    This remains unsupervised: it uses only local signal activity, not species labels.
    """
    if not events or not config.enable_segmentation_refinement:
        return events

    duration_sec = float(len(audio) / sr) if sr else 0.0
    refined = [
        _refine_one_event(audio, sr, event, config, duration_sec)
        for event in events
    ]
    filtered = [
        event for event in refined
        if event.duration_sec >= config.min_event_duration
    ]
    deduped = temporal_non_max_suppression(filtered, iou_threshold=config.refinement_nms_iou)
    return _renumber_events(sorted(deduped, key=lambda item: item.start_sec))


def temporal_non_max_suppression(events: list[AudioEvent], iou_threshold: float) -> list[AudioEvent]:
    """Remove near-duplicate events that strongly overlap in time."""
    if len(events) <= 1 or iou_threshold >= 1:
        return events

    ordered = sorted(events, key=_event_score, reverse=True)
    kept: list[AudioEvent] = []
    for event in ordered:
        if all(_interval_iou(event, kept_event) < iou_threshold for kept_event in kept):
            kept.append(event)
    return kept


def _refine_one_event(
    audio: np.ndarray,
    sr: int,
    event: AudioEvent,
    config: BioSoundConfig,
    duration_sec: float,
) -> AudioEvent:
    start_sample = max(0, int(round(event.start_sec * sr)))
    end_sample = min(len(audio), int(round(event.end_sec * sr)))
    clip = audio[start_sample:end_sample]
    if clip.size < max(16, int(config.min_event_duration * sr * 0.25)):
        return event

    frame_length = min(config.frame_length, max(256, 2 ** int(np.floor(np.log2(max(256, clip.size))))))
    hop_length = min(config.hop_length, max(64, frame_length // 4))
    rms = librosa.feature.rms(y=clip, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0:
        return event

    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0, top_db=None)
    floor_db = float(np.percentile(rms_db, 20))
    peak_db = float(np.max(rms_db))
    threshold_db = max(
        floor_db + config.refinement_activity_db,
        peak_db - config.refinement_peak_drop_db,
    )
    active = rms_db >= threshold_db
    if not np.any(active):
        return event

    close_frames = max(1, int(round(0.04 * sr / hop_length)))
    active = binary_closing(active, structure=np.ones(close_frames, dtype=bool))
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        return event

    rel_start = float(librosa.frames_to_time(int(active_indices[0]), sr=sr, hop_length=hop_length))
    rel_end = float(librosa.frames_to_time(int(active_indices[-1]) + 1, sr=sr, hop_length=hop_length))
    new_start = max(0.0, event.start_sec + rel_start - config.refinement_padding)
    new_end = min(duration_sec, event.start_sec + rel_end + config.refinement_padding)

    if new_end <= new_start:
        return event
    if (new_end - new_start) < config.min_event_duration:
        center = (new_start + new_end) / 2.0
        half = config.min_event_duration / 2.0
        new_start = max(0.0, center - half)
        new_end = min(duration_sec, center + half)
    if (
        abs(new_start - event.start_sec) < config.refinement_min_trim_sec
        and abs(new_end - event.end_sec) < config.refinement_min_trim_sec
    ):
        return event

    stats = compute_event_stats(audio, sr, new_start, new_end)
    return replace(
        event,
        start_sec=float(new_start),
        end_sec=float(new_end),
        duration_sec=float(new_end - new_start),
        rms_db=stats["rms_db"],
        peak_db=stats["peak_db"],
        spectral_centroid=stats["spectral_centroid"],
    )


def _interval_iou(a: AudioEvent, b: AudioEvent) -> float:
    intersection = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    union = max(a.end_sec, b.end_sec) - min(a.start_sec, b.start_sec)
    return float(intersection / union) if union > 0 else 0.0


def _event_score(event: AudioEvent) -> float:
    if event.peak_db is not None:
        return float(event.peak_db)
    if event.rms_db is not None:
        return float(event.rms_db)
    return float(event.duration_sec)


def _renumber_events(events: list[AudioEvent]) -> list[AudioEvent]:
    return [
        replace(event, event_id=f"event_{idx:06d}")
        for idx, event in enumerate(events)
    ]
