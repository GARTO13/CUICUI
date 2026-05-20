from __future__ import annotations

import numpy as np

from biosound_cluster.sota.event_extraction import (
    refine_boundaries,
    runs_to_events,
    temporal_nms,
    WindowEvent,
)


def test_runs_to_events_groups_contiguous_labels() -> None:
    start_times = np.arange(10, dtype=np.float64) * 1.0
    labels = np.array([-1, 0, 0, 0, -1, 1, 1, -1, 0, 0], dtype=np.int64)
    stability = np.ones(10, dtype=np.float32)
    rms = np.full(10, -20.0, dtype=np.float32)
    events = runs_to_events(
        start_times=start_times,
        window_sec=1.0,
        hop_sec=1.0,
        labels=labels,
        stability=stability,
        rms_db=rms,
        min_event_duration=0.5,
        max_event_duration=30.0,
        max_event_gap=0.1,
    )
    assert len(events) == 3
    assert events[0].cluster_id == 0
    assert events[0].start_sec == 1.0
    assert events[1].cluster_id == 1
    assert events[2].cluster_id == 0


def test_runs_to_events_breaks_on_long_gap() -> None:
    start_times = np.array([0.0, 1.0, 5.0, 6.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 0], dtype=np.int64)
    stability = np.ones(4, dtype=np.float32)
    rms = np.full(4, -10.0, dtype=np.float32)
    events = runs_to_events(
        start_times=start_times,
        window_sec=1.0,
        hop_sec=1.0,
        labels=labels,
        stability=stability,
        rms_db=rms,
        min_event_duration=0.5,
        max_event_duration=30.0,
        max_event_gap=0.5,
    )
    assert len(events) == 2


def test_temporal_nms_keeps_most_stable() -> None:
    a = WindowEvent(cluster_id=0, start_sec=0.0, end_sec=2.0, window_indices=[0], mean_stability=0.9, mean_rms_db=-20)
    b = WindowEvent(cluster_id=0, start_sec=0.5, end_sec=2.5, window_indices=[1], mean_stability=0.5, mean_rms_db=-20)
    kept = temporal_nms([a, b], iou_threshold=0.3)
    assert len(kept) == 1
    assert kept[0].mean_stability == 0.9


def test_refine_boundaries_tightens_silent_tails() -> None:
    sr = 16_000
    audio = np.zeros(sr * 3, dtype=np.float32)
    burst_start = int(0.8 * sr)
    burst_end = int(1.2 * sr)
    audio[burst_start:burst_end] = 0.5 * np.sin(
        2 * np.pi * 1500 * np.arange(burst_end - burst_start) / sr
    )
    event = WindowEvent(
        cluster_id=0,
        start_sec=0.0,
        end_sec=3.0,
        window_indices=[0, 1],
        mean_stability=1.0,
        mean_rms_db=-20.0,
    )
    refined = refine_boundaries(
        audio=audio,
        sample_rate=sr,
        events=[event],
        activity_db=12.0,
        padding_sec=0.02,
        smoothing_sec=0.01,
    )
    assert refined[0].start_sec > 0.5
    assert refined[0].end_sec < 1.5
