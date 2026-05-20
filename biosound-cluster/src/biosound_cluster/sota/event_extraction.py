"""Convert per-window cluster labels into temporal events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class WindowEvent:
    """Intermediate event derived from contiguous windows of the same cluster."""

    cluster_id: int
    start_sec: float
    end_sec: float
    window_indices: list[int]
    mean_stability: float
    mean_rms_db: float


def runs_to_events(
    start_times: np.ndarray,
    window_sec: float,
    hop_sec: float,
    labels: np.ndarray,
    stability: np.ndarray,
    rms_db: np.ndarray,
    min_event_duration: float,
    max_event_duration: float,
    max_event_gap: float,
) -> list[WindowEvent]:
    """Group contiguous windows with the same label into events.

    A run breaks when:
        - the label changes,
        - the next window starts more than ``max_event_gap`` seconds after the
          previous one ended,
        - extending the run would exceed ``max_event_duration``.
    """
    events: list[WindowEvent] = []
    n = len(labels)
    if n == 0:
        return events

    current_label: int | None = None
    current_indices: list[int] = []

    def _flush() -> None:
        if current_label is None or not current_indices:
            return
        idxs = current_indices
        first_start = float(start_times[idxs[0]])
        last_start = float(start_times[idxs[-1]])
        end = last_start + window_sec
        duration = end - first_start
        if duration < min_event_duration:
            return
        stab = stability[idxs].astype(np.float64)
        rms = rms_db[idxs].astype(np.float64)
        events.append(
            WindowEvent(
                cluster_id=int(current_label),
                start_sec=first_start,
                end_sec=end,
                window_indices=list(idxs),
                mean_stability=float(stab.mean()) if stab.size else 0.0,
                mean_rms_db=float(rms.mean()) if rms.size else -120.0,
            )
        )

    for i in range(n):
        label = int(labels[i])
        if label < 0:
            _flush()
            current_label = None
            current_indices = []
            continue
        if current_label is None:
            current_label = label
            current_indices = [i]
            continue
        prev_end = float(start_times[current_indices[-1]]) + window_sec
        gap = float(start_times[i]) - prev_end
        candidate_duration = (float(start_times[i]) + window_sec) - float(start_times[current_indices[0]])
        if (
            label != current_label
            or gap > max_event_gap
            or candidate_duration > max_event_duration
        ):
            _flush()
            current_label = label
            current_indices = [i]
        else:
            current_indices.append(i)
    _flush()
    return events


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0.0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0.0:
        return 0.0
    return inter / union


def temporal_nms(events: list[WindowEvent], iou_threshold: float) -> list[WindowEvent]:
    """Greedy NMS keeping the most stable event in each conflict group."""
    if not events:
        return []
    sorted_events = sorted(
        events,
        key=lambda e: (-e.mean_stability, -(e.end_sec - e.start_sec)),
    )
    kept: list[WindowEvent] = []
    for event in sorted_events:
        clash = any(
            _iou(event.start_sec, event.end_sec, k.start_sec, k.end_sec) >= iou_threshold
            for k in kept
        )
        if not clash:
            kept.append(event)
    return sorted(kept, key=lambda e: e.start_sec)


def refine_boundaries(
    audio: np.ndarray,
    sample_rate: int,
    events: list[WindowEvent],
    activity_db: float,
    padding_sec: float,
    smoothing_sec: float,
) -> list[WindowEvent]:
    """Tighten each event's [start, end] using a smoothed |audio| envelope."""
    if not events or audio.size == 0:
        return events

    smooth_n = max(1, int(round(smoothing_sec * sample_rate)))
    kernel = np.ones(smooth_n, dtype=np.float32) / smooth_n
    refined: list[WindowEvent] = []
    for event in events:
        s = max(0, int(round(event.start_sec * sample_rate)))
        e = min(audio.shape[0], int(round(event.end_sec * sample_rate)))
        if e <= s:
            refined.append(event)
            continue
        clip = audio[s:e].astype(np.float32, copy=False)
        envelope = np.abs(clip)
        if envelope.size >= smooth_n:
            envelope = np.convolve(envelope, kernel, mode="same")
        peak = float(envelope.max()) if envelope.size else 0.0
        if peak <= 1e-9:
            refined.append(event)
            continue
        threshold = peak * (10.0 ** (-activity_db / 20.0))
        active = envelope > threshold
        if not bool(active.any()):
            refined.append(event)
            continue
        first = int(np.argmax(active))
        last = int(len(active) - np.argmax(active[::-1]) - 1)
        new_start_sec = max(
            event.start_sec,
            event.start_sec + first / sample_rate - padding_sec,
        )
        new_end_sec = min(
            event.end_sec,
            event.start_sec + last / sample_rate + padding_sec,
        )
        if new_end_sec <= new_start_sec:
            refined.append(event)
            continue
        refined.append(
            WindowEvent(
                cluster_id=event.cluster_id,
                start_sec=new_start_sec,
                end_sec=new_end_sec,
                window_indices=list(event.window_indices),
                mean_stability=event.mean_stability,
                mean_rms_db=event.mean_rms_db,
            )
        )
    return refined
