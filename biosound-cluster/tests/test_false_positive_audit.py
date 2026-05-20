from __future__ import annotations

from pathlib import Path

import pandas as pd

from biosound_cluster.evaluation.error_audit import audit_false_positives


def test_false_positive_audit_writes_csv_and_markdown(tmp_path: Path) -> None:
    matched = pd.DataFrame(
        [
            {"event_id": "a", "matched_gt_label": "POS", "match_iou": 0.5, "match_overlap_ratio": 0.5, "duration_sec": 0.5, "detection_score": 0.8, "source_type": "original"},
            {"event_id": "b", "matched_gt_label": "", "match_iou": 0.0, "match_overlap_ratio": 0.0, "duration_sec": 0.1, "detection_score": 0.2, "source_type": "component"},
        ]
    )
    result = audit_false_positives(matched, pd.DataFrame(), tmp_path)
    assert Path(result["csv"]).exists()
    assert Path(result["markdown"]).exists()
    audit = pd.read_csv(result["csv"])
    assert "feature" in audit.columns
