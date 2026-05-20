from __future__ import annotations

from pathlib import Path

import pandas as pd

from biosound_cluster.evaluation.calibration import calibrate_routing_thresholds_from_dcase


def test_routing_calibration_writes_outputs(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pd.DataFrame(
        [
            {"event_id": "a", "matched_gt_label": "POS", "match_iou": 0.5, "match_overlap_ratio": 0.5, "detection_score": 0.8, "component_quality_score": 0.9, "duration_sec": 0.4},
            {"event_id": "b", "matched_gt_label": "", "match_iou": 0.0, "match_overlap_ratio": 0.0, "detection_score": 0.2, "component_quality_score": 0.3, "duration_sec": 0.1},
        ]
    ).to_csv(eval_dir / "matched_predictions.csv", index=False)

    result = calibrate_routing_thresholds_from_dcase(eval_dir, tmp_path / "calibration", min_recall_retention=0.5)

    assert Path(result["recommended_thresholds_json"]).exists()
    assert Path(result["results_csv"]).exists()
