"""Lightweight DCASE parameter tuning for the unsupervised pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from biosound_cluster.config import BioSoundConfig, config_fingerprint, config_to_dict
from biosound_cluster.evaluation.dcase2024 import evaluate_dcase2024

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TuningCandidate:
    """A named, human-readable parameter candidate."""

    name: str
    config: BioSoundConfig


def build_tuning_grid(base_config: BioSoundConfig | None = None, search_size: str = "quick") -> list[TuningCandidate]:
    """
    Build a small deterministic tuning grid.

    This is parameter selection for an unsupervised annotation assistant, not supervised
    species classification. Labels are used only to score detection and cluster usefulness.
    """
    base = base_config or BioSoundConfig()
    base_values = asdict(base)
    if search_size not in {"quick", "balanced"}:
        raise ValueError("search_size must be quick or balanced")

    variants: list[tuple[str, dict[str, object]]] = [
        (
            "baseline_current",
            {},
        ),
        (
            "tight_temporal_precision",
            {
                "threshold_db": max(base.threshold_db, 8.0),
                "flux_percentile": 94.0,
                "flux_mad_multiplier": 5.0,
                "flux_min_snr_db": 2.0,
                "merge_gap": min(base.merge_gap, 0.15),
                "padding": min(base.padding, 0.05),
                "max_event_duration": min(base.max_event_duration, 4.0),
                "noise_mode": "balanced",
                "min_quality_for_clustering": max(base.min_quality_for_clustering, 0.48),
                "min_eventness_for_clustering": max(base.min_eventness_for_clustering, 0.30),
                "max_components_per_parent": min(base.max_components_per_parent, 2),
            },
        ),
        (
            "energy_only_precision",
            {
                "enable_flux_detection": False,
                "threshold_db": max(base.threshold_db, 8.0),
                "merge_gap": min(base.merge_gap, 0.15),
                "padding": min(base.padding, 0.05),
                "max_event_duration": min(base.max_event_duration, 4.0),
                "noise_mode": "balanced",
                "min_quality_for_clustering": max(base.min_quality_for_clustering, 0.48),
                "min_eventness_for_clustering": max(base.min_eventness_for_clustering, 0.32),
                "max_components_per_parent": 1,
            },
        ),
        (
            "precision_balanced",
            {
                "threshold_db": max(base.threshold_db, 9.0),
                "flux_percentile": 92.0,
                "flux_mad_multiplier": 4.5,
                "flux_min_snr_db": 2.0,
                "noise_mode": "balanced",
                "min_quality_for_clustering": max(base.min_quality_for_clustering, 0.50),
                "min_eventness_for_clustering": max(base.min_eventness_for_clustering, 0.34),
                "max_components_per_parent": min(base.max_components_per_parent, 2),
            },
        ),
        (
            "rare_event_guard",
            {
                "threshold_db": min(base.threshold_db, 7.0),
                "flux_percentile": 94.0,
                "flux_mad_multiplier": 5.0,
                "flux_min_snr_db": 1.5,
                "noise_mode": "exploratory",
                "min_quality_for_clustering": min(base.min_quality_for_clustering, 0.40),
                "min_eventness_for_clustering": min(base.min_eventness_for_clustering, 0.24),
                "max_components_per_parent": max(base.max_components_per_parent, 3),
            },
        ),
        (
            "clean_clusters",
            {
                "threshold_db": max(base.threshold_db, 10.0),
                "flux_percentile": 94.0,
                "flux_mad_multiplier": 5.0,
                "flux_min_snr_db": 2.5,
                "noise_mode": "conservative",
                "min_quality_for_clustering": max(base.min_quality_for_clustering, 0.55),
                "min_eventness_for_clustering": max(base.min_eventness_for_clustering, 0.38),
                "max_components_per_parent": 1,
                "umap_neighbors": max(10, min(base.umap_neighbors, 20)),
            },
        ),
    ]
    if search_size == "balanced":
        variants.extend(
            [
                (
                    "high_recall_structured",
                    {
                        "threshold_db": 6.0,
                        "flux_percentile": 90.0,
                        "flux_mad_multiplier": 4.0,
                        "flux_min_snr_db": 1.0,
                        "noise_mode": "exploratory",
                        "min_quality_for_clustering": 0.38,
                    },
                ),
                (
                    "strict_flux",
                    {
                        "threshold_db": 8.0,
                        "flux_percentile": 96.0,
                        "flux_mad_multiplier": 5.5,
                        "flux_min_snr_db": 2.0,
                        "noise_mode": "balanced",
                        "min_quality_for_clustering": 0.48,
                    },
                ),
                (
                    "stable_medium_clusters",
                    {
                        "threshold_db": 9.0,
                        "flux_percentile": 93.0,
                        "flux_mad_multiplier": 4.5,
                        "flux_min_snr_db": 2.0,
                        "noise_mode": "balanced",
                        "min_quality_for_clustering": 0.50,
                        "umap_neighbors": 15,
                        "umap_min_dist": 0.0,
                    },
                ),
            ]
        )

    candidates: list[TuningCandidate] = []
    seen: set[str] = set()
    for name, updates in variants:
        config = BioSoundConfig(**{**base_values, **updates})
        fingerprint = config_fingerprint(config)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidates.append(TuningCandidate(name=name, config=config))
    return candidates


def tune_dcase2024(
    dataset_dir: Path,
    output_dir: Path,
    annotations_dir: Path | None = None,
    split: str = "validation",
    subset: str | None = None,
    max_files: int | None = None,
    base_config: BioSoundConfig | None = None,
    iou_threshold: float = 0.3,
    overlap_threshold: float = 0.3,
    search_size: str = "quick",
    max_trials: int | None = None,
    force: bool = False,
) -> dict:
    """Run a deterministic parameter sweep and return the best DCASE evaluation result."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_tuning_grid(base_config, search_size=search_size)
    if max_trials is not None:
        candidates = candidates[:max_trials]

    rows: list[dict[str, object]] = []
    summaries: dict[str, dict] = {}
    for index, candidate in enumerate(candidates):
        trial_name = f"trial_{index:03d}_{candidate.name}"
        LOGGER.info("Running DCASE tuning candidate %s", trial_name)
        summary = evaluate_dcase2024(
            dataset_dir=dataset_dir,
            annotations_dir=annotations_dir,
            output_dir=output_dir / trial_name,
            split=split,
            subset=subset,
            max_files=max_files,
            config=candidate.config,
            iou_threshold=iou_threshold,
            overlap_threshold=overlap_threshold,
            force=force,
        )
        summaries[trial_name] = summary
        detection = summary.get("detection", {})
        clustering = summary.get("clustering", {})
        noise_filtering = summary.get("noise_filtering", {})
        rows.append(
            {
                "trial": trial_name,
                "config_hash": config_fingerprint(candidate.config),
                "final_score_100": summary.get("final_score_100", 0.0),
                "precision": detection.get("precision", 0.0),
                "recall": detection.get("recall", 0.0),
                "f1": detection.get("f1", 0.0),
                "mean_iou": detection.get("mean_iou", 0.0),
                "weighted_cluster_purity": clustering.get("weighted_cluster_purity", 0.0),
                "annotation_compression_ratio": clustering.get("annotation_compression_ratio", 0.0),
                "n_clusters": clustering.get("n_clusters", 0),
                "n_noise": clustering.get("n_noise", 0),
                "n_low_confidence_noise": noise_filtering.get("n_low_confidence_noise", 0),
                "config": json.dumps(config_to_dict(candidate.config), sort_keys=True),
            }
        )

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(
            ["final_score_100", "f1", "weighted_cluster_purity"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    best_row = results.iloc[0].to_dict() if not results.empty else {}
    best_trial = str(best_row.get("trial", "")) if best_row else ""
    best_config = {}
    if best_trial:
        candidate_index = int(best_trial.split("_")[1])
        best_config = config_to_dict(candidates[candidate_index].config)

    results_csv = output_dir / "tuning_results.csv"
    summary_json = output_dir / "tuning_summary.json"
    best_config_json = output_dir / "best_config.json"
    results.to_csv(results_csv, index=False)
    best_config_json.write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    summary = {
        "best_trial": best_trial,
        "best_score_100": best_row.get("final_score_100", 0.0) if best_row else 0.0,
        "best_config": best_config,
        "results_csv": str(results_csv),
        "best_config_json": str(best_config_json),
        "summaries": summaries,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
