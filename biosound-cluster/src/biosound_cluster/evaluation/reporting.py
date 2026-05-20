"""Reporting utilities for evaluation results."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from biosound_cluster.evaluation.metrics import (
    ClusteringMetrics,
    DetectionMetrics,
    EvaluationScore,
    PolyphonyMetrics,
)


def write_evaluation_report(
    path: Path,
    summary: dict,
    detection: DetectionMetrics,
    clustering: ClusteringMetrics,
    polyphony: PolyphonyMetrics,
    score: EvaluationScore,
) -> None:
    """Write a human-readable Markdown evaluation report."""
    dataset = summary.get("dataset", {})
    noise_filtering = summary.get("noise_filtering", {})
    routing = summary.get("routing", {})
    short_event_review = summary.get("short_event_review", {})
    assistant_metrics = summary.get("assistant_metrics", {})
    stability = assistant_metrics.get("cluster_stability_summary", {}) or {}
    lines = [
        "# BioSound DCASE 2024 Evaluation Report",
        "",
        "## Summary",
        "",
        f"Final score (clusterable): {summary.get('final_score_100_clusterable', score.final_score_100):.1f} / 100",
        f"Raw score: {summary.get('final_score_100_raw', 0.0):.1f} / 100",
        f"Parent-aware score: {summary.get('final_score_100_parent_aware', 0.0):.1f} / 100",
        "",
        "## Dataset",
        "",
        f"Files evaluated: {dataset.get('files_evaluated', 0)}",
        f"Split: {dataset.get('split', '')}",
        f"Subset: {dataset.get('subset') or 'all'}",
        f"Total ground truth events: {detection.n_gt}",
        f"Total predicted events: {detection.n_pred}",
        "",
        "## Event detection",
        "",
        f"Precision: {detection.precision:.3f}",
        f"Recall: {detection.recall:.3f}",
        f"F1: {detection.f1:.3f}",
        f"Mean IoU: {detection.mean_iou:.3f}",
        "",
        "Raw score penalizes every separated component as an independent prediction.",
        "Parent-aware score evaluates temporal detection before component splitting.",
        "Clusterable score estimates what the researcher sees in normal clusters.",
        "",
        "## Routing",
        "",
        f"Total events: {routing.get('n_total_events', 0)}",
        f"Clusterable events: {routing.get('n_clusterable_events', 0)}",
        f"Component review events: {routing.get('n_component_review', 0)}",
        f"Mixed events: {routing.get('n_mixed', 0)}",
        f"Low detection confidence: {routing.get('n_low_detection_confidence', 0)}",
        f"Component clusterable ratio: {routing.get('component_clusterable_ratio', 0.0):.3f}",
        "",
        "## Clustering",
        "",
        f"Clusters: {clustering.n_clusters}",
        f"Noise events: {clustering.n_noise}",
        f"Weighted purity: {clustering.weighted_cluster_purity:.3f}",
        f"NMI: {_fmt_optional(clustering.normalized_mutual_info)}",
        f"ARI: {_fmt_optional(clustering.adjusted_rand_index)}",
        f"Annotation compression ratio: {clustering.annotation_compression_ratio:.2f}",
        "",
        "## Annotation assistant metrics",
        "",
        f"Normal cluster precision: {_fmt_optional(assistant_metrics.get('normal_cluster_precision'))}",
        f"Global recall any folder: {_fmt_optional(assistant_metrics.get('global_recall_any_folder'))}",
        f"Representative precision@{assistant_metrics.get('representative_k', 5)}: {_fmt_optional(assistant_metrics.get('representative_precision_at_k'))}",
        f"Cluster stability mean: {_fmt_optional(stability.get('mean'))}",
        f"Cluster stability median: {_fmt_optional(stability.get('median'))}",
        "",
        "## Polyphony handling",
        "",
        f"Enabled: {str(polyphony.enabled).lower()}",
        f"Mixed events: {polyphony.n_mixed_events}",
        f"Component events: {polyphony.n_component_events}",
        f"Mean purity score: {_fmt_optional(polyphony.mean_purity_score)}",
        "",
        "## Noise filtering",
        "",
        f"Enabled: {str(noise_filtering.get('enabled', False)).lower()}",
        f"Low-confidence noise events: {noise_filtering.get('n_low_confidence_noise', 0)}",
        f"Ambiguous review events: {noise_filtering.get('n_ambiguous_review', 0)}",
        f"Low-confidence rate: {noise_filtering.get('low_confidence_rate', 0.0):.3f}",
        f"Mean quality score: {_fmt_optional(noise_filtering.get('mean_quality_score'))}",
        "",
        "## Short event review",
        "",
        f"Enabled: {str(short_event_review.get('enabled', False)).lower()}",
        f"Short review events: {short_event_review.get('n_short_review_events', 0)}",
        f"Short review rate: {short_event_review.get('short_review_rate', 0.0):.3f}",
        "",
        "## Interpretation",
        "",
        "- The score measures usefulness as a human-in-the-loop annotation assistant, not species classification accuracy.",
        "- Strong detection F1 means candidate events align well with annotated time intervals.",
        "- Strong weighted purity means events with the same annotation label tend to land in the same acoustic-family folders.",
        "- A high compression ratio means each cluster summarizes many events, reducing expert listening effort.",
        "- Clusters are acoustic groups, not species labels.",
        "",
        "## Recommended next actions",
        "",
        "- Adjust `threshold_db` if recall is low.",
        "- Adjust `min_cluster_size` if there are too many or too few clusters.",
        "- Inspect `mixed_overlapping` folders if polyphony score is weak.",
        "- Listen to representatives before assigning any biological label.",
        "",
    ]
    failed = dataset.get("failed_files") or []
    if failed:
        lines.extend(["## Failed files", ""])
        for item in failed:
            lines.append(f"- `{item.get('file_id', '')}`: {item.get('error', '')}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def print_evaluation_summary(summary: dict, console: Console | None = None) -> None:
    """Print a compact Rich evaluation summary."""
    console = console or Console()
    detection = summary.get("detection", {})
    clustering = summary.get("clustering", {})
    polyphony = summary.get("polyphony", {})
    noise_filtering = summary.get("noise_filtering", {})
    short_event_review = summary.get("short_event_review", {})
    assistant_metrics = summary.get("assistant_metrics", {})
    table = Table(title="BioSound DCASE Evaluation")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Final score (clusterable)", f"{summary.get('final_score_100_clusterable', summary.get('final_score_100', 0.0)):.1f} / 100")
    table.add_row("Score raw", f"{summary.get('final_score_100_raw', 0.0):.1f} / 100")
    table.add_row("Score parent-aware", f"{summary.get('final_score_100_parent_aware', 0.0):.1f} / 100")
    table.add_row("Detection F1", f"{detection.get('f1', 0.0):.3f}")
    table.add_row("Precision", f"{detection.get('precision', 0.0):.3f}")
    table.add_row("Recall", f"{detection.get('recall', 0.0):.3f}")
    table.add_row("Mean IoU", f"{detection.get('mean_iou', 0.0):.3f}")
    table.add_row("Weighted purity", f"{clustering.get('weighted_cluster_purity', 0.0):.3f}")
    table.add_row("Normal cluster precision", _fmt_optional(assistant_metrics.get("normal_cluster_precision")))
    table.add_row("Recall any folder", _fmt_optional(assistant_metrics.get("global_recall_any_folder")))
    table.add_row(
        f"Representative precision@{assistant_metrics.get('representative_k', 5)}",
        _fmt_optional(assistant_metrics.get("representative_precision_at_k")),
    )
    table.add_row("Compression", f"{clustering.get('annotation_compression_ratio', 0.0):.2f} events/cluster")
    table.add_row("Clusters", str(clustering.get("n_clusters", 0)))
    table.add_row("Mixed events", str(polyphony.get("n_mixed_events", 0)))
    table.add_row("Component events", str(polyphony.get("n_component_events", 0)))
    table.add_row("Low-confidence noise", str(noise_filtering.get("n_low_confidence_noise", 0)))
    table.add_row("Ambiguous review", str(noise_filtering.get("n_ambiguous_review", 0)))
    table.add_row("Short review events", str(short_event_review.get("n_short_review_events", 0)))
    console.print(table)


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
