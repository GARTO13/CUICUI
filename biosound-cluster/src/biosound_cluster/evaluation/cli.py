"""CLI for DCASE 2024 Task 5 evaluation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.evaluation.dcase2024 import (
    ZENODO_RECORD_URL,
    compare_dcase2024_noise_modes,
    download_dcase2024_annotations,
    evaluate_dcase2024,
)
from biosound_cluster.evaluation.calibration import calibrate_routing_thresholds_from_dcase
from biosound_cluster.evaluation.reporting import print_evaluation_summary
from biosound_cluster.evaluation.tuning import tune_dcase2024
from biosound_cluster.logging_utils import configure_logging

app = typer.Typer(help="Evaluate biosound-cluster on DCASE 2024 Task 5.")
console = Console()


@app.command()
def run(
    dataset_dir: Path = typer.Option(..., "--dataset-dir", help="Path to DCASE Development_Set."),
    annotations_dir: Optional[Path] = typer.Option(None, "--annotations-dir", help="Optional separate annotations directory."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Evaluation output directory."),
    split: str = typer.Option("validation", "--split", help="train, validation, or all."),
    max_files: Optional[int] = typer.Option(None, "--max-files", help="Maximum number of audio files to evaluate."),
    subset: Optional[str] = typer.Option(None, "--subset", help="Optional subset, e.g. BV, JD, MT, WMW, ME, PB."),
    iou_threshold: float = typer.Option(0.3, "--iou-threshold", help="IoU threshold for event matching."),
    overlap_threshold: float = typer.Option(0.3, "--overlap-threshold", help="GT overlap threshold for event matching."),
    threshold_db: float = typer.Option(8.0, "--threshold-db", help="Pipeline energy threshold above rolling noise floor."),
    flux_percentile: float = typer.Option(90.0, "--flux-percentile", help="Pipeline robust spectral-flux percentile threshold."),
    flux_min_snr_db: float = typer.Option(1.5, "--flux-min-snr-db", help="Minimum energy above noise floor for flux-only detections."),
    disable_flux_detection: bool = typer.Option(False, "--disable-flux-detection", help="Use energy-only event detection."),
    disable_multiband_segmentation: bool = typer.Option(False, "--disable-multiband-segmentation", help="Disable conservative multi-band event detection."),
    multiband_min_snr_db: float = typer.Option(6.0, "--multiband-min-snr-db", help="Minimum local per-band SNR for multi-band detection."),
    multiband_min_flux_z: float = typer.Option(2.0, "--multiband-min-flux-z", help="Minimum robust per-band flux z-score for multi-band detection."),
    disable_segmentation_refinement: bool = typer.Option(False, "--disable-segmentation-refinement", help="Disable boundary refinement and temporal duplicate removal."),
    min_cluster_size: int = typer.Option(10, "--min-cluster-size", help="Pipeline HDBSCAN minimum cluster size."),
    disable_polyphony_handling: bool = typer.Option(False, "--disable-polyphony-handling", help="Disable pipeline polyphony handling."),
    legacy_polyphony_routing: bool = typer.Option(False, "--legacy-polyphony-routing", help="Use permissive pre-v2 component routing."),
    disable_noise_filtering: bool = typer.Option(False, "--disable-noise-filtering", help="Disable low-confidence noise routing."),
    noise_mode: str = typer.Option("balanced", "--noise-mode", help="Noise filtering mode: exploratory, balanced, or conservative."),
    min_quality_for_clustering: float = typer.Option(0.55, "--min-quality-for-clustering", help="Minimum acoustic quality score sent to clustering."),
    disable_eventness_filtering: bool = typer.Option(False, "--disable-eventness-filtering", help="Disable low-confidence eventness routing."),
    min_eventness_for_clustering: float = typer.Option(0.28, "--min-eventness-for-clustering", help="Minimum temporal salience score sent to clustering."),
    disable_candidate_selection: bool = typer.Option(False, "--disable-candidate-selection", help="Disable component limiting and temporal NMS candidate pruning."),
    max_components_per_parent: int = typer.Option(3, "--max-components-per-parent", help="Maximum separated components kept per original event parent."),
    disable_short_event_review: bool = typer.Option(False, "--disable-short-event-review", help="Keep very short events in normal clustering."),
    min_review_event_duration: float = typer.Option(0.20, "--min-review-event-duration", help="Route events shorter than this many seconds to secondary review."),
    disable_clusterability_filtering: bool = typer.Option(False, "--disable-clusterability-filtering", help="Disable clusterability scoring and ambiguous review routing."),
    min_clusterability_for_clustering: float = typer.Option(0.55, "--min-clusterability-for-clustering", help="Minimum clusterability score sent to normal clustering."),
    min_clusterability_for_review: float = typer.Option(0.30, "--min-clusterability-for-review", help="Below this clusterability score is very low priority/noise review."),
    disable_embedding_stability: bool = typer.Option(False, "--disable-embedding-stability", help="Disable multi-view embedding stability scoring."),
    disable_acoustic_prefamilies: bool = typer.Option(False, "--disable-acoustic-prefamilies", help="Disable acoustic pre-family routing before clustering."),
    prefamily_min_events: int = typer.Option(15, "--prefamily-min-events", help="Minimum events required to cluster an acoustic pre-family separately."),
    cluster_ensemble_runs: int = typer.Option(1, "--cluster-ensemble-runs", help="Optional extra clustering runs for stability scoring; 1 disables ensemble overhead."),
    no_spectrograms: bool = typer.Option(False, "--no-spectrograms", help="Disable spectrogram PNG export during pipeline runs."),
    no_clips: bool = typer.Option(False, "--no-clips", help="Disable WAV/PNG media export during pipeline runs."),
    sensor_id: Optional[str] = typer.Option(None, "--sensor-id", help="Optional sensor/deployment identifier stored in metadata."),
    sensor_latitude: Optional[float] = typer.Option(None, "--sensor-latitude", help="Sensor latitude in decimal degrees."),
    sensor_longitude: Optional[float] = typer.Option(None, "--sensor-longitude", help="Sensor longitude in decimal degrees."),
    sensor_elevation_m: Optional[float] = typer.Option(None, "--sensor-elevation-m", help="Sensor elevation in meters."),
    environment_type: Optional[str] = typer.Option(None, "--environment-type", help="Environment type stored in run/event metadata."),
    recording_start_time: Optional[str] = typer.Option(None, "--recording-start-time", help="Recording start time as ISO-8601."),
    recording_timezone: Optional[str] = typer.Option(None, "--recording-timezone", help="Optional timezone name stored in metadata."),
    enable_auto_profile: bool = typer.Option(False, "--enable-auto-profile", help="Profile each audio file and adapt detection/filtering parameters to its acoustic regime."),
    enable_semantic_tagging: bool = typer.Option(False, "--enable-semantic-tagging", help="Use optional PANNs AudioSet tagging as an adaptive-profile prior."),
    enable_denoiser: bool = typer.Option(False, "--enable-denoiser", help="Pre-process audio with an optional registered denoiser."),
    denoiser_name: str = typer.Option("biodenoising", "--denoiser-name", help="Denoiser backend name used with --enable-denoiser."),
    compare_noise_modes: bool = typer.Option(False, "--compare-noise-modes", help="Run baseline/exploratory/balanced/conservative noise-mode comparison."),
    tune: bool = typer.Option(False, "--tune", help="Run a deterministic DCASE parameter tuning sweep."),
    tuning_search: str = typer.Option("quick", "--tuning-search", help="Tuning grid size: quick or balanced."),
    max_trials: Optional[int] = typer.Option(None, "--max-trials", help="Maximum tuning trials to run."),
    force: bool = typer.Option(False, "--force", help="Rerun pipeline even if outputs already exist."),
    download: bool = typer.Option(False, "--download", help="Explicitly download DCASE annotations helper archive from Zenodo."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    """Evaluate the pipeline on local DCASE files."""
    configure_logging(verbose)
    if download:
        console.print(
            "[bold yellow]Download requested.[/bold yellow] The full DCASE audio archive is ~21.9 GB and is not downloaded silently."
        )
        console.print(f"Dataset record: {ZENODO_RECORD_URL}")
        try:
            downloaded_annotations = download_dcase2024_annotations(output_dir / "downloads")
            console.print(f"Downloaded annotations to: {downloaded_annotations}")
            if annotations_dir is None:
                annotations_dir = downloaded_annotations
        except Exception as exc:
            logging.getLogger(__name__).warning("Could not download annotations: %s", exc)
            console.print("Could not download automatically. Visit the Zenodo record above and download manually.")

    config = BioSoundConfig(
        threshold_db=threshold_db,
        flux_percentile=flux_percentile,
        flux_min_snr_db=flux_min_snr_db,
        enable_flux_detection=not disable_flux_detection,
        enable_multiband_segmentation=not disable_multiband_segmentation,
        multiband_min_snr_db=multiband_min_snr_db,
        multiband_min_flux_z=multiband_min_flux_z,
        enable_segmentation_refinement=not disable_segmentation_refinement,
        min_cluster_size=min_cluster_size,
        enable_polyphony_handling=not disable_polyphony_handling,
        enable_noise_filtering=not disable_noise_filtering,
        noise_mode=noise_mode,
        min_quality_for_clustering=min_quality_for_clustering,
        enable_eventness_filtering=not disable_eventness_filtering,
        min_eventness_for_clustering=min_eventness_for_clustering,
        enable_candidate_selection=not disable_candidate_selection,
        max_components_per_parent=max_components_per_parent,
        enable_short_event_review=not disable_short_event_review,
        min_review_event_duration=min_review_event_duration,
        enable_clusterability_filtering=not disable_clusterability_filtering,
        min_clusterability_for_clustering=min_clusterability_for_clustering,
        min_clusterability_for_review=min_clusterability_for_review,
        enable_embedding_stability=not disable_embedding_stability,
        enable_acoustic_prefamilies=not disable_acoustic_prefamilies,
        prefamily_min_events=prefamily_min_events,
        cluster_ensemble_runs=cluster_ensemble_runs,
        generate_spectrograms=False if no_clips else not no_spectrograms,
        export_clips=not no_clips,
        sensor_id=sensor_id,
        sensor_latitude=sensor_latitude,
        sensor_longitude=sensor_longitude,
        sensor_elevation_m=sensor_elevation_m,
        environment_type=environment_type,
        recording_start_time=recording_start_time,
        recording_timezone=recording_timezone,
        enable_auto_profile=enable_auto_profile,
        enable_semantic_tagging=enable_semantic_tagging,
        enable_denoiser=enable_denoiser,
        denoiser_name=denoiser_name,
    )
    if legacy_polyphony_routing:
        config.enable_component_explosion_control = False
        config.polyphony_split_requires_low_overlap = False
        config.polyphony_split_requires_compact_masks = False
        config.max_components_for_clustering_per_parent = config.max_components_per_event
        config.min_component_purity_for_clustering_strict = config.min_purity_for_clustering
        config.min_component_quality_for_clustering = 0.0
        config.min_component_snr_db_for_clustering = 0.0
    if compare_noise_modes:
        comparison = compare_dcase2024_noise_modes(
            dataset_dir=dataset_dir,
            annotations_dir=annotations_dir,
            output_dir=output_dir,
            split=split,
            subset=subset,
            max_files=max_files,
            base_config=config,
            iou_threshold=iou_threshold,
            overlap_threshold=overlap_threshold,
            force=force,
        )
        _print_comparison(comparison)
        return
    if tune:
        tuning = tune_dcase2024(
            dataset_dir=dataset_dir,
            annotations_dir=annotations_dir,
            output_dir=output_dir,
            split=split,
            subset=subset,
            max_files=max_files,
            base_config=config,
            iou_threshold=iou_threshold,
            overlap_threshold=overlap_threshold,
            search_size=tuning_search,
            max_trials=max_trials,
            force=force,
        )
        _print_tuning(tuning)
        return

    summary = evaluate_dcase2024(
        dataset_dir=dataset_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        split=split,
        subset=subset,
        max_files=max_files,
        config=config,
        iou_threshold=iou_threshold,
        overlap_threshold=overlap_threshold,
        force=force,
    )
    print_evaluation_summary(summary, console=console)


def _print_comparison(comparison: dict) -> None:
    table = Table(title="BioSound noise-mode comparison")
    table.add_column("Config", style="bold")
    table.add_column("Score")
    table.add_column("Precision")
    table.add_column("Recall")
    table.add_column("F1")
    table.add_column("Purity")
    table.add_column("Low-noise")
    for row in comparison.get("configs", []):
        table.add_row(
            str(row.get("config", "")),
            f"{row.get('final_score_100', 0.0):.1f}",
            f"{row.get('precision', 0.0):.3f}",
            f"{row.get('recall', 0.0):.3f}",
            f"{row.get('f1', 0.0):.3f}",
            f"{row.get('weighted_cluster_purity', 0.0):.3f}",
            str(row.get("n_low_confidence_noise", 0)),
        )
    console.print(table)
    console.print(f"Comparison CSV: {comparison.get('comparison_csv')}")


def _print_tuning(tuning: dict) -> None:
    console.print("[bold]BioSound tuning complete[/bold]")
    console.print(f"Best trial: {tuning.get('best_trial')}")
    console.print(f"Best score: {float(tuning.get('best_score_100', 0.0)):.1f} / 100")
    console.print(f"Results CSV: {tuning.get('results_csv')}")
    console.print(f"Best config: {tuning.get('best_config_json')}")


def calibrate(
    eval_dir: Path = typer.Option(..., "--eval-dir", help="Existing evaluation directory."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Calibration output directory."),
    min_recall_retention: float = typer.Option(0.85, "--min-recall-retention", help="Minimum retained true-positive proxy."),
) -> None:
    result = calibrate_routing_thresholds_from_dcase(eval_dir, output_dir, min_recall_retention)
    console.print("[bold]Routing calibration complete[/bold]")
    console.print(f"Results CSV: {result.get('results_csv')}")
    console.print(f"Recommended thresholds: {result.get('recommended_thresholds_json')}")


def calibrate_main() -> None:
    """Entry point for routing-threshold calibration."""
    typer.run(calibrate)


def main() -> None:
    """Entry point."""
    app()


def tune_main() -> None:
    """Entry point that defaults to DCASE tuning."""
    if "--tune" not in sys.argv:
        sys.argv.append("--tune")
    app()


if __name__ == "__main__":
    main()
