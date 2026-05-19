"""Review-focused routing for events that should not enter normal clusters."""

from __future__ import annotations

from dataclasses import replace

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.schemas import AudioEvent


def route_short_review_events(
    events: list[AudioEvent],
    config: BioSoundConfig,
) -> tuple[list[AudioEvent], list[AudioEvent]]:
    """
    Route very short events to a secondary review bucket.

    Short sounds may be biologically real, but they are often poor representatives for
    human cluster review. They are kept and exported separately rather than deleted.
    """
    if not events or not config.enable_short_event_review or config.min_review_event_duration <= 0:
        return events, []

    clusterable: list[AudioEvent] = []
    short_review: list[AudioEvent] = []
    for event in events:
        if event.duration_sec < config.min_review_event_duration:
            short_review.append(_mark_short_review(event, config.min_review_event_duration))
        else:
            clusterable.append(event)
    return clusterable, short_review


def _mark_short_review(event: AudioEvent, threshold: float) -> AudioEvent:
    short = replace(event)
    short.cluster_id = None
    short.cluster_probability = None
    short.is_noise = False
    short.is_short_review_event = True
    short.candidate_route_reason = f"short_event:<{threshold:.3f}s"
    short.source_type = "short_review"
    return short
