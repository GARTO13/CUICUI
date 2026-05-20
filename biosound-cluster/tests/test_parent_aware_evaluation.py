from __future__ import annotations

import pandas as pd

from biosound_cluster.evaluation.dcase2024 import GroundTruthEvent
from biosound_cluster.evaluation.metrics import compute_detection_metrics


def test_parent_aware_collapses_components_for_detection() -> None:
    events = pd.DataFrame(
        [
            {"event_id": "p1_c0", "parent_event_id": "p1", "start_sec": 1.00, "end_sec": 1.30, "source_type": "component", "is_component": True},
            {"event_id": "p1_c1", "parent_event_id": "p1", "start_sec": 1.25, "end_sec": 1.60, "source_type": "component", "is_component": True},
            {"event_id": "p1_c2", "parent_event_id": "p1", "start_sec": 1.55, "end_sec": 1.90, "source_type": "component", "is_component": True},
        ]
    )
    gt = [GroundTruthEvent(file_id="f", start_sec=0.95, end_sec=1.95, label="POS")]

    raw = compute_detection_metrics(gt, events, parent_aware=False, iou_threshold=0.1, overlap_threshold=0.1)
    parent = compute_detection_metrics(gt, events, parent_aware=True, iou_threshold=0.1, overlap_threshold=0.1)

    assert raw.n_pred == 3
    assert parent.n_pred == 1
    assert parent.precision > raw.precision
