"""Metrics for unsupervised bioacoustic annotation-assistant evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log1p
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

if TYPE_CHECKING:
    from biosound_cluster.evaluation.dcase2024 import GroundTruthEvent


@dataclass(slots=True)
class DetectionMetrics:
    n_gt: int
    n_pred: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    median_time_error_sec: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ClusteringMetrics:
    n_matched_events: int
    n_clusters: int
    n_noise: int
    cluster_purity: float
    weighted_cluster_purity: float
    normalized_mutual_info: float | None
    adjusted_rand_index: float | None
    annotation_compression_ratio: float
    mean_cluster_size: float
    median_cluster_size: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PolyphonyMetrics:
    enabled: bool
    n_mixed_events: int
    n_component_events: int
    n_original_events: int
    mixed_rate: float
    component_rate: float
    mean_purity_score: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationScore:
    detection_f1: float
    weighted_cluster_purity: float
    annotation_compression_score: float
    temporal_quality_score: float
    polyphony_quality_score: float
    final_score_100: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute intersection-over-union for two time intervals."""
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return float(intersection / union) if union > 0 else 0.0


def interval_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return overlap divided by the ground-truth interval duration."""
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    duration = max(0.0, a_end - a_start)
    return float(intersection / duration) if duration > 0 else 0.0


def _normal_events(pred_events: pd.DataFrame) -> pd.DataFrame:
    if pred_events.empty:
        return pred_events.copy()
    events = pred_events.copy()
    if "source_type" in events.columns:
        events = events[~events["source_type"].fillna("original").isin({"mixed", "low_confidence_noise", "short_review"})]
    if "is_mixed" in events.columns:
        events = events[~events["is_mixed"].fillna(False).astype(bool)]
    if "is_low_confidence_noise" in events.columns:
        events = events[~events["is_low_confidence_noise"].fillna(False).astype(bool)]
    return _collapse_component_predictions(events.reset_index(drop=True))


def _collapse_component_predictions(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse separated components back to their parent for temporal detection metrics."""
    if events.empty or "source_type" not in events.columns or "parent_event_id" not in events.columns:
        return events
    component_mask = (events["source_type"].fillna("") == "component") & events["parent_event_id"].fillna("").astype(str).ne("")
    if not component_mask.any():
        return events

    originals = events[~component_mask].copy()
    rows: list[dict[str, object]] = []
    for parent_id, group in events[component_mask].groupby("parent_event_id", dropna=True):
        first = group.iloc[0].to_dict()
        first["event_id"] = str(parent_id)
        first["start_sec"] = float(pd.to_numeric(group["start_sec"], errors="coerce").min())
        first["end_sec"] = float(pd.to_numeric(group["end_sec"], errors="coerce").max())
        first["duration_sec"] = first["end_sec"] - first["start_sec"]
        first["source_type"] = "component_parent"
        first["is_component"] = False
        rows.append(first)
    collapsed = pd.DataFrame(rows, columns=events.columns) if rows else events.iloc[0:0].copy()
    return pd.concat([originals, collapsed], ignore_index=True)


def match_events(
    gt_events: list["GroundTruthEvent"],
    pred_events: pd.DataFrame,
    iou_threshold: float = 0.3,
    overlap_threshold: float = 0.3,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedily match ground-truth events to predictions by temporal overlap."""
    preds = _normal_events(pred_events)
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt in enumerate(gt_events):
        for pred_idx, pred in preds.iterrows():
            iou = interval_iou(gt.start_sec, gt.end_sec, float(pred["start_sec"]), float(pred["end_sec"]))
            overlap = interval_overlap_ratio(gt.start_sec, gt.end_sec, float(pred["start_sec"]), float(pred["end_sec"]))
            score = max(iou, overlap)
            if iou >= iou_threshold or overlap >= overlap_threshold:
                candidates.append((score, gt_idx, int(pred_idx)))

    candidates.sort(reverse=True, key=lambda item: item[0])
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, float(score)))

    unmatched_gt = [idx for idx in range(len(gt_events)) if idx not in used_gt]
    unmatched_pred = [idx for idx in range(len(preds)) if idx not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def compute_detection_metrics(
    gt_events: list["GroundTruthEvent"],
    pred_events: pd.DataFrame,
    iou_threshold: float = 0.3,
    overlap_threshold: float = 0.3,
) -> DetectionMetrics:
    """Compute temporal event detection metrics."""
    preds = _normal_events(pred_events)
    matches, unmatched_gt, unmatched_pred = match_events(
        gt_events,
        preds,
        iou_threshold=iou_threshold,
        overlap_threshold=overlap_threshold,
    )
    true_positives = len(matches)
    false_positives = len(unmatched_pred)
    false_negatives = len(unmatched_gt)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ious: list[float] = []
    time_errors: list[float] = []
    for gt_idx, pred_idx, _ in matches:
        gt = gt_events[gt_idx]
        pred = preds.iloc[pred_idx]
        ious.append(interval_iou(gt.start_sec, gt.end_sec, float(pred["start_sec"]), float(pred["end_sec"])))
        gt_mid = (gt.start_sec + gt.end_sec) / 2.0
        pred_mid = (float(pred["start_sec"]) + float(pred["end_sec"])) / 2.0
        time_errors.append(abs(gt_mid - pred_mid))
    return DetectionMetrics(
        n_gt=len(gt_events),
        n_pred=len(preds),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mean_iou=float(np.mean(ious)) if ious else 0.0,
        median_time_error_sec=float(np.median(time_errors)) if time_errors else None,
    )


def assign_predictions_to_ground_truth(
    file_id: str,
    gt_events: list["GroundTruthEvent"],
    pred_events: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each prediction to the ground-truth event it overlaps best."""
    if pred_events.empty:
        return pd.DataFrame(
            columns=[
                "file_id",
                "event_id",
                "start_sec",
                "end_sec",
                "cluster_id",
                "source_type",
                "is_mixed",
                "matched_gt_start",
                "matched_gt_end",
                "matched_gt_label",
                "match_iou",
                "match_overlap_ratio",
            ]
        )

    rows: list[dict[str, object]] = []
    for _, pred in pred_events.iterrows():
        best_gt: GroundTruthEvent | None = None
        best_iou = 0.0
        best_overlap = 0.0
        for gt in gt_events:
            iou = interval_iou(gt.start_sec, gt.end_sec, float(pred["start_sec"]), float(pred["end_sec"]))
            overlap = interval_overlap_ratio(gt.start_sec, gt.end_sec, float(pred["start_sec"]), float(pred["end_sec"]))
            if max(iou, overlap) > max(best_iou, best_overlap):
                best_gt = gt
                best_iou = iou
                best_overlap = overlap
        rows.append(
            {
                "file_id": file_id,
                "event_id": pred.get("event_id"),
                "start_sec": pred.get("start_sec"),
                "end_sec": pred.get("end_sec"),
                "cluster_id": pred.get("cluster_id"),
                "source_type": pred.get("source_type", "original"),
                "is_mixed": bool(pred.get("is_mixed", False)),
                "matched_gt_start": best_gt.start_sec if best_gt else np.nan,
                "matched_gt_end": best_gt.end_sec if best_gt else np.nan,
                "matched_gt_label": best_gt.label if best_gt else "",
                "match_iou": best_iou,
                "match_overlap_ratio": best_overlap,
            }
        )
    return pd.DataFrame(rows)


def compute_clustering_metrics(matched_predictions: pd.DataFrame) -> ClusteringMetrics:
    """Compute clustering usefulness metrics for matched non-noise predictions."""
    if matched_predictions.empty:
        return ClusteringMetrics(0, 0, 0, 0.0, 0.0, None, None, 0.0, 0.0, 0.0)

    df = matched_predictions.copy()
    df = df[~df["source_type"].fillna("original").isin({"mixed", "low_confidence_noise", "short_review"})]
    df = df[df["matched_gt_label"].fillna("") != ""]
    cluster_ids = pd.to_numeric(df["cluster_id"], errors="coerce")
    n_noise = int(cluster_ids.isna().sum() + (cluster_ids == -1).sum())
    clustered = df[cluster_ids.notna() & (cluster_ids != -1)].copy()
    if clustered.empty:
        return ClusteringMetrics(len(df), 0, n_noise, 0.0, 0.0, None, None, 0.0, 0.0, 0.0)

    clustered["cluster_id"] = pd.to_numeric(clustered["cluster_id"], errors="coerce").astype(int)
    purities: list[float] = []
    weights: list[int] = []
    for _, group in clustered.groupby("cluster_id"):
        counts = group["matched_gt_label"].value_counts()
        purity = float(counts.iloc[0] / len(group)) if len(group) else 0.0
        purities.append(purity)
        weights.append(len(group))
    labels = clustered["matched_gt_label"].astype(str).to_numpy()
    clusters = clustered["cluster_id"].astype(str).to_numpy()
    nmi = None
    ari = None
    if len(set(labels)) >= 2 and len(set(clusters)) >= 2:
        nmi = float(normalized_mutual_info_score(labels, clusters))
        ari = float(adjusted_rand_score(labels, clusters))
    n_clusters = int(clustered["cluster_id"].nunique())
    cluster_sizes = clustered.groupby("cluster_id").size().to_numpy()
    compression = float(len(clustered) / n_clusters) if n_clusters else 0.0
    return ClusteringMetrics(
        n_matched_events=int(len(df)),
        n_clusters=n_clusters,
        n_noise=n_noise,
        cluster_purity=float(np.mean(purities)) if purities else 0.0,
        weighted_cluster_purity=float(np.average(purities, weights=weights)) if purities else 0.0,
        normalized_mutual_info=nmi,
        adjusted_rand_index=ari,
        annotation_compression_ratio=compression,
        mean_cluster_size=float(np.mean(cluster_sizes)) if cluster_sizes.size else 0.0,
        median_cluster_size=float(np.median(cluster_sizes)) if cluster_sizes.size else 0.0,
    )


def compute_polyphony_metrics(events_df: pd.DataFrame) -> PolyphonyMetrics:
    """Compute summary metrics for mixed/component event handling."""
    required = {"is_mixed", "source_type", "is_component", "parent_event_id", "purity_score", "polyphony_score"}
    if events_df.empty or not required.issubset(events_df.columns):
        return PolyphonyMetrics(False, 0, 0, 0, 0.0, 0.0, None)
    n_total = len(events_df)
    source = events_df["source_type"].fillna("original")
    n_mixed = int((source == "mixed").sum())
    n_components = int((source == "component").sum())
    n_original = int((source == "original").sum())
    purity = pd.to_numeric(events_df["purity_score"], errors="coerce").dropna()
    return PolyphonyMetrics(
        enabled=True,
        n_mixed_events=n_mixed,
        n_component_events=n_components,
        n_original_events=n_original,
        mixed_rate=float(n_mixed / n_total) if n_total else 0.0,
        component_rate=float(n_components / n_total) if n_total else 0.0,
        mean_purity_score=float(purity.mean()) if not purity.empty else None,
    )


def compute_global_score(
    detection: DetectionMetrics,
    clustering: ClusteringMetrics,
    polyphony: PolyphonyMetrics,
) -> EvaluationScore:
    """Compute the BioSound evaluation score on a 0-100 scale."""
    detection_f1 = float(np.clip(detection.f1, 0.0, 1.0))
    weighted_purity = float(np.clip(clustering.weighted_cluster_purity, 0.0, 1.0))
    compression = min(log1p(max(0.0, clustering.annotation_compression_ratio)) / log1p(20.0), 1.0)
    temporal = float(np.clip(detection.mean_iou, 0.0, 1.0))
    polyphony_quality = 0.5 if not polyphony.enabled or polyphony.mean_purity_score is None else float(
        np.clip(polyphony.mean_purity_score, 0.0, 1.0)
    )
    final = (
        0.40 * detection_f1
        + 0.25 * weighted_purity
        + 0.15 * compression
        + 0.10 * temporal
        + 0.10 * polyphony_quality
    )
    return EvaluationScore(
        detection_f1=detection_f1,
        weighted_cluster_purity=weighted_purity,
        annotation_compression_score=float(compression),
        temporal_quality_score=temporal,
        polyphony_quality_score=polyphony_quality,
        final_score_100=float(final * 100.0),
    )


def aggregate_detection_metrics(metrics: list[DetectionMetrics]) -> DetectionMetrics:
    """Aggregate detection metrics by summing counts and recomputing rates."""
    if not metrics:
        return DetectionMetrics(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, None)
    tp = sum(item.true_positives for item in metrics)
    fp = sum(item.false_positives for item in metrics)
    fn = sum(item.false_negatives for item in metrics)
    n_gt = sum(item.n_gt for item in metrics)
    n_pred = sum(item.n_pred for item in metrics)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou_values = [item.mean_iou for item in metrics if item.true_positives > 0]
    iou_weights = [item.true_positives for item in metrics if item.true_positives > 0]
    errors = [item.median_time_error_sec for item in metrics if item.median_time_error_sec is not None]
    return DetectionMetrics(
        n_gt=n_gt,
        n_pred=n_pred,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mean_iou=float(np.average(iou_values, weights=iou_weights)) if iou_values else 0.0,
        median_time_error_sec=float(np.median(errors)) if errors else None,
    )
