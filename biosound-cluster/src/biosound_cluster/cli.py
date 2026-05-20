"""Command-line interface for biosound-cluster."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.logging_utils import configure_logging

app = typer.Typer(help="Unsupervised acoustic event clustering for bioacoustic review.")
console = Console()


@app.command()
def run(
    input_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True, help="Input audio file."),
    output: Path = typer.Option(..., "--output", "-o", help="Output run directory."),
    sample_rate: int = typer.Option(32000, "--sample-rate", help="Target audio sample rate."),
    min_cluster_size: int = typer.Option(10, "--min-cluster-size", help="Minimum HDBSCAN cluster size."),
    min_event_duration: float = typer.Option(0.25, "--min-event-duration", help="Minimum event duration in seconds."),
    max_event_duration: float = typer.Option(8.0, "--max-event-duration", help="Maximum event duration before splitting."),
    merge_gap: float = typer.Option(0.4, "--merge-gap", help="Merge events separated by this many seconds."),
    padding: float = typer.Option(0.15, "--padding", help="Seconds of padding around each event."),
    threshold_db: float = typer.Option(8.0, "--threshold-db", help="Energy threshold above rolling noise floor."),
    flux_percentile: float = typer.Option(90.0, "--flux-percentile", help="Robust spectral-flux percentile threshold."),
    flux_min_snr_db: float = typer.Option(1.5, "--flux-min-snr-db", help="Minimum energy above noise floor for flux-only detections."),
    disable_flux_detection: bool = typer.Option(False, "--disable-flux-detection", help="Use energy-only event detection."),
    disable_multiband_segmentation: bool = typer.Option(False, "--disable-multiband-segmentation", help="Disable conservative multi-band event detection."),
    multiband_min_snr_db: float = typer.Option(6.0, "--multiband-min-snr-db", help="Minimum local per-band SNR for multi-band detection."),
    multiband_min_flux_z: float = typer.Option(2.0, "--multiband-min-flux-z", help="Minimum robust per-band flux z-score for multi-band detection."),
    disable_segmentation_refinement: bool = typer.Option(False, "--disable-segmentation-refinement", help="Disable boundary refinement and temporal duplicate removal."),
    max_events: Optional[int] = typer.Option(None, "--max-events", help="Optional event limit for debugging."),
    no_spectrograms: bool = typer.Option(False, "--no-spectrograms", help="Disable spectrogram PNG export."),
    no_clips: bool = typer.Option(False, "--no-clips", help="Disable WAV/PNG media export; write metadata only."),
    sensor_id: Optional[str] = typer.Option(None, "--sensor-id", help="Optional sensor/deployment identifier stored in metadata."),
    sensor_latitude: Optional[float] = typer.Option(None, "--sensor-latitude", help="Sensor latitude in decimal degrees."),
    sensor_longitude: Optional[float] = typer.Option(None, "--sensor-longitude", help="Sensor longitude in decimal degrees."),
    sensor_elevation_m: Optional[float] = typer.Option(None, "--sensor-elevation-m", help="Sensor elevation in meters."),
    environment_type: Optional[str] = typer.Option(None, "--environment-type", help="Environment type, e.g. tropical_forest, wetland, urban_edge."),
    recording_start_time: Optional[str] = typer.Option(None, "--recording-start-time", help="Recording start time as ISO-8601, e.g. 2026-05-20T06:30:00+02:00."),
    recording_timezone: Optional[str] = typer.Option(None, "--recording-timezone", help="Optional timezone name stored in metadata, e.g. Europe/Paris."),
    disable_polyphony_handling: bool = typer.Option(False, "--disable-polyphony-handling", help="Disable mixed/overlapping sound handling."),
    legacy_polyphony_routing: bool = typer.Option(False, "--legacy-polyphony-routing", help="Use permissive pre-v2 component routing."),
    component_snr_db: float = typer.Option(10.0, "--component-snr-db", help="SNR threshold for time-frequency components."),
    min_purity_for_clustering: float = typer.Option(0.55, "--min-purity-for-clustering", help="Minimum component purity score sent to clustering."),
    max_components_per_event: int = typer.Option(4, "--max-components-per-event", help="Maximum separated components per detected event."),
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
    enable_auto_profile: bool = typer.Option(False, "--enable-auto-profile", help="Profile the audio and adapt detection/filtering parameters to its acoustic regime."),
    enable_semantic_tagging: bool = typer.Option(False, "--enable-semantic-tagging", help="Use optional PANNs AudioSet tagging as an adaptive-profile prior."),
    enable_denoiser: bool = typer.Option(False, "--enable-denoiser", help="Pre-process audio with an optional registered denoiser."),
    denoiser_name: str = typer.Option("biodenoising", "--denoiser-name", help="Denoiser backend name used with --enable-denoiser."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Cluster candidate acoustic events from one local audio recording."""
    configure_logging(verbose)
    config = BioSoundConfig(
        sample_rate=sample_rate,
        min_cluster_size=min_cluster_size,
        min_event_duration=min_event_duration,
        max_event_duration=max_event_duration,
        merge_gap=merge_gap,
        padding=padding,
        threshold_db=threshold_db,
        flux_percentile=flux_percentile,
        flux_min_snr_db=flux_min_snr_db,
        enable_flux_detection=not disable_flux_detection,
        enable_multiband_segmentation=not disable_multiband_segmentation,
        multiband_min_snr_db=multiband_min_snr_db,
        multiband_min_flux_z=multiband_min_flux_z,
        enable_segmentation_refinement=not disable_segmentation_refinement,
        max_events=max_events,
        generate_spectrograms=False if no_clips else not no_spectrograms,
        export_clips=not no_clips,
        sensor_id=sensor_id,
        sensor_latitude=sensor_latitude,
        sensor_longitude=sensor_longitude,
        sensor_elevation_m=sensor_elevation_m,
        environment_type=environment_type,
        recording_start_time=recording_start_time,
        recording_timezone=recording_timezone,
        enable_polyphony_handling=not disable_polyphony_handling,
        component_snr_db=component_snr_db,
        min_purity_for_clustering=min_purity_for_clustering,
        max_components_per_event=max_components_per_event,
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
    from biosound_cluster.pipeline import process_audio_file

    result = process_audio_file(input_path, output, config)

    table = Table(title="biosound-cluster summary")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("input", result.input_path)
    table.add_row("output", result.output_dir)
    table.add_row("duration", f"{result.duration_sec:.2f}s")
    table.add_row("events", str(result.n_events))
    table.add_row("clusters", str(result.n_clusters))
    table.add_row("noise/unknown", str(result.n_noise))
    table.add_row("report", result.report_md)
    table.add_row("html index", result.index_html)
    console.print(table)


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
