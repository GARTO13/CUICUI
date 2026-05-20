"""False-positive audit helpers for DCASE evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_FEATURES = [
    "duration_sec",
    "detection_score",
    "event_snr_db",
    "active_band_fraction",
    "spectral_flux_score",
    "stationarity_score",
    "purity_score",
    "component_energy_ratio",
    "component_snr_db",
    "component_compactness",
    "component_quality_score",
]


def audit_false_positives(
    matched_predictions: pd.DataFrame,
    events_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Compare matched and unmatched predictions to explain false positives."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if matched_predictions.empty:
        rows: list[dict[str, object]] = []
        audit_df = pd.DataFrame(rows, columns=["feature", "matched_mean", "unmatched_mean", "matched_median", "unmatched_median", "suggested_threshold", "direction", "separation_score"])
        audit_df.to_csv(output_dir / "false_positive_audit.csv", index=False)
        (output_dir / "false_positive_audit.md").write_text("# False positive audit\n\nNo predictions available.\n", encoding="utf-8")
        return {"csv": str(output_dir / "false_positive_audit.csv"), "markdown": str(output_dir / "false_positive_audit.md")}

    df = matched_predictions.copy()
    if not events_df.empty and "event_id" in events_df.columns:
        extra_cols = [col for col in events_df.columns if col not in df.columns or col == "event_id"]
        df = df.merge(events_df[extra_cols].drop_duplicates("event_id"), on="event_id", how="left")
    matched_mask = (
        df.get("matched_gt_label", pd.Series([""] * len(df))).fillna("").astype(str).ne("")
        & (
            pd.to_numeric(df.get("match_iou"), errors="coerce").fillna(0.0).ge(0.3)
            | pd.to_numeric(df.get("match_overlap_ratio"), errors="coerce").fillna(0.0).ge(0.3)
        )
    )

    rows = []
    for feature in AUDIT_FEATURES:
        if feature not in df.columns:
            continue
        matched = pd.to_numeric(df.loc[matched_mask, feature], errors="coerce").dropna()
        unmatched = pd.to_numeric(df.loc[~matched_mask, feature], errors="coerce").dropna()
        if matched.empty or unmatched.empty:
            continue
        separation = abs(float(matched.median() - unmatched.median())) / (float(matched.std() + unmatched.std()) + 1e-9)
        direction = "min" if matched.median() > unmatched.median() else "max"
        threshold = float((matched.median() + unmatched.median()) / 2.0)
        rows.append(
            {
                "feature": feature,
                "matched_mean": float(matched.mean()),
                "unmatched_mean": float(unmatched.mean()),
                "matched_median": float(matched.median()),
                "unmatched_median": float(unmatched.median()),
                "suggested_threshold": threshold,
                "direction": direction,
                "separation_score": separation,
            }
        )

    audit_df = pd.DataFrame(rows, columns=["feature", "matched_mean", "unmatched_mean", "matched_median", "unmatched_median", "suggested_threshold", "direction", "separation_score"])
    audit_csv = output_dir / "false_positive_audit.csv"
    audit_md = output_dir / "false_positive_audit.md"
    audit_df.to_csv(audit_csv, index=False)

    source = df.get("source_type", pd.Series(["original"] * len(df))).fillna("original")
    unmatched_source = source[~matched_mask]
    component_fp_rate = float(unmatched_source.isin(["component", "component_review"]).mean()) if len(unmatched_source) else 0.0
    top = audit_df.sort_values("separation_score", ascending=False).head(6) if not audit_df.empty else audit_df
    lines = [
        "# False positive audit",
        "",
        f"Unmatched predictions: {int((~matched_mask).sum())}",
        f"Matched predictions: {int(matched_mask.sum())}",
        f"False positives from components/component_review: {component_fp_rate:.3f}",
        "",
        "## Signals",
        "",
    ]
    for _, row in top.iterrows():
        direction = "raise minimum" if row["direction"] == "min" else "lower maximum"
        lines.append(
            f"- `{row['feature']}` separates matched/unmatched events; suggested {direction} around {row['suggested_threshold']:.3f}."
        )
    lines.extend(
        [
            "",
            "Check whether false positives have weaker detection scores, short duration, high stationarity, low energy ratio, or poor component quality before tightening thresholds.",
            "",
        ]
    )
    audit_md.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(audit_csv), "markdown": str(audit_md)}
