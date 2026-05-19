"""Candidate selection after segmentation, polyphony, and eventness scoring."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.schemas import AudioEvent


def select_clusterable_candidates(
    events: list[AudioEvent],
    config: BioSoundConfig,
) -> tuple[list[AudioEvent], list[AudioEvent]]:
    """
    Reduce over-produced candidates while preserving reviewability.

    The selected events continue to embeddings/clustering. Pruned candidates are routed
    to the low-confidence review bucket instead of being deleted.
    """
    if not events or not config.enable_candidate_selection:
        return events, []

    scored = [_attach_selection_score(event) for event in events]
    kept_after_parent, pruned_parent = _limit_components_per_parent(scored, config.max_components_per_parent)
    kept_after_nms, pruned_nms = _temporal_nms(kept_after_parent, config.candidate_nms_iou)
    return sorted(kept_after_nms, key=lambda event: event.start_sec), pruned_parent + pruned_nms


def _limit_components_per_parent(
    events: list[AudioEvent],
    max_components_per_parent: int,
) -> tuple[list[AudioEvent], list[AudioEvent]]:
    components_by_parent: dict[str, list[AudioEvent]] = defaultdict(list)
    passthrough: list[AudioEvent] = []
    for event in events:
        if event.is_component and event.parent_event_id:
            components_by_parent[event.parent_event_id].append(event)
        else:
            passthrough.append(event)

    kept = list(passthrough)
    pruned: list[AudioEvent] = []
    for parent_id, group in components_by_parent.items():
        ordered = sorted(group, key=_selection_score, reverse=True)
        kept.extend(ordered[:max_components_per_parent])
        pruned.extend(
            _mark_pruned(event, f"component_limit:{parent_id}")
            for event in ordered[max_components_per_parent:]
        )
    return kept, pruned


def _temporal_nms(
    events: list[AudioEvent],
    iou_threshold: float,
) -> tuple[list[AudioEvent], list[AudioEvent]]:
    if len(events) <= 1 or iou_threshold >= 1:
        return events, []

    ordered = sorted(events, key=_selection_score, reverse=True)
    kept: list[AudioEvent] = []
    pruned: list[AudioEvent] = []
    for event in ordered:
        if any(_should_suppress(event, kept_event, iou_threshold) for kept_event in kept):
            pruned.append(_mark_pruned(event, "temporal_nms"))
        else:
            kept.append(event)
    return kept, pruned


def _should_suppress(candidate: AudioEvent, kept: AudioEvent, iou_threshold: float) -> bool:
    if candidate.parent_event_id and kept.parent_event_id and candidate.parent_event_id == kept.parent_event_id:
        return False
    return _interval_iou(candidate, kept) >= iou_threshold


def _attach_selection_score(event: AudioEvent) -> AudioEvent:
    event.selection_score = _selection_score(event)
    return event


def _selection_score(event: AudioEvent) -> float:
    eventness = _value(event.eventness_score, 0.5)
    purity = _value(event.purity_score, 0.7 if not event.is_component else 0.5)
    quality = _value(event.quality_score, 0.7)
    duration = _duration_preference(event.duration_sec)
    peak = 0.5 if event.peak_db is None else float(np.clip((event.peak_db + 60.0) / 60.0, 0.0, 1.0))
    return float(
        0.40 * eventness
        + 0.25 * purity
        + 0.20 * quality
        + 0.10 * duration
        + 0.05 * peak
    )


def _duration_preference(duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    short = _smoothstep(0.08, 0.25, duration_sec)
    long = 1.0 - _smoothstep(5.0, 8.0, duration_sec)
    return float(np.clip(short * long, 0.0, 1.0))


def _mark_pruned(event: AudioEvent, reason: str) -> AudioEvent:
    pruned = replace(event)
    pruned.cluster_id = None
    pruned.cluster_probability = None
    pruned.is_noise = True
    pruned.is_low_confidence_noise = True
    pruned.is_low_confidence_event = True
    pruned.is_pruned_candidate = True
    pruned.candidate_route_reason = reason
    pruned.source_type = "low_confidence_noise"
    return pruned


def _interval_iou(a: AudioEvent, b: AudioEvent) -> float:
    intersection = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    union = max(a.end_sec, b.end_sec) - min(a.start_sec, b.start_sec)
    return float(intersection / union) if union > 0 else 0.0


def _value(value: float | None, default: float) -> float:
    if value is None:
        return default
    return float(np.clip(value, 0.0, 1.0))


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    x = float(np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)
