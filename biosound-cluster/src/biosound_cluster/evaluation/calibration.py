"""Threshold calibration from an existing DCASE evaluation run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


THRESHOLDS = {
    "min_clusterable_detection_score": ("detection_score", [0.20, 0.30, 0.35, 0.40, 0.50], "min"),
    "min_component_quality_for_clustering": ("component_quality_score", [0.45, 0.55, 0.62, 0.70, 0.80], "min"),
    "min_component_purity_for_clustering_strict": ("purity_score", [0.55, 0.65, 0.72, 0.80], "min"),
    "min_component_energy_ratio_for_clustering": ("component_energy_ratio", [0.06, 0.10, 0.12, 0.18, 0.25], "min"),
    "min_component_snr_db_for_clustering": ("component_snr_db", [4.0, 6.0, 8.0, 10.0, 12.0], "min"),
    "min_component_compactness": ("component_compactness", [0.08, 0.12, 0.15, 0.22, 0.30], "min"),
    "min_duration": ("duration_sec", [0.08, 0.12, 0.20, 0.25, 0.35], "min"),
    "max_stationarity_score": ("stationarity_score", [0.70, 0.80, 0.90, 0.95], "max"),
}


def calibrate_routing_thresholds_from_dcase(
    eval_dir: Path,
    output_dir: Path,
    min_recall_retention: float = 0.85,
) -> dict:
    """Test simple routing thresholds on an existing DCASE run."""
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matched_csv = eval_dir / "matched_predictions.csv"
    if not matched_csv.exists():
        raise FileNotFoundError(f"matched_predictions.csv not found in {eval_dir}")
    df = pd.read_csv(matched_csv)
    usable = (
        df.get("matched_gt_label", pd.Series([""] * len(df))).fillna("").astype(str).ne("")
        & (
            pd.to_numeric(df.get("match_iou"), errors="coerce").fillna(0.0).ge(0.3)
            | pd.to_numeric(df.get("match_overlap_ratio"), errors="coerce").fillna(0.0).ge(0.3)
        )
    )
    base_tp = int(usable.sum())
    base_pred = len(df)
    base_precision = base_tp / base_pred if base_pred else 0.0
    rows = []
    for name, (feature, values, direction) in THRESHOLDS.items():
        if feature not in df.columns:
            continue
        series = pd.to_numeric(df[feature], errors="coerce")
        for value in values:
            keep = series.ge(value) if direction == "min" else series.le(value)
            keep = keep | series.isna()
            tp = int((usable & keep).sum())
            pred = int(keep.sum())
            recall_retention = tp / base_tp if base_tp else 0.0
            precision = tp / pred if pred else 0.0
            rows.append(
                {
                    "threshold_name": name,
                    "feature": feature,
                    "value": value,
                    "direction": direction,
                    "kept_predictions": pred,
                    "precision_proxy": precision,
                    "precision_gain": precision - base_precision,
                    "recall_retention": recall_retention,
                }
            )
    results = pd.DataFrame(rows)
    results_csv = output_dir / "routing_calibration_results.csv"
    results.to_csv(results_csv, index=False)
    recommended = {}
    if not results.empty:
        feasible = results[results["recall_retention"] >= min_recall_retention].copy()
        for name, group in feasible.groupby("threshold_name"):
            best = group.sort_values(["precision_gain", "precision_proxy"], ascending=False).iloc[0]
            recommended[name] = float(best["value"])
    rec_json = output_dir / "recommended_thresholds.json"
    rec_json.write_text(json.dumps(recommended, indent=2), encoding="utf-8")
    report = output_dir / "routing_calibration_report.md"
    lines = [
        "# Routing calibration",
        "",
        f"Base precision proxy: {base_precision:.3f}",
        f"Minimum recall retention: {min_recall_retention:.3f}",
        "",
        "## Recommended thresholds",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in recommended.items())
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"results_csv": str(results_csv), "recommended_thresholds_json": str(rec_json), "report_md": str(report), "recommended_thresholds": recommended}
