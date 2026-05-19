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
    disable_segmentation_refinement: bool = typer.Option(False, "--disable-segmentation-refinement", help="Disable boundary refinement and temporal duplicate removal."),
    max_events: Optional[int] = typer.Option(None, "--max-events", help="Optional event limit for debugging."),
    no_spectrograms: bool = typer.Option(False, "--no-spectrograms", help="Disable spectrogram PNG export."),
    no_clips: bool = typer.Option(False, "--no-clips", help="Disable WAV/PNG media export; write metadata only."),
    disable_polyphony_handling: bool = typer.Option(False, "--disable-polyphony-handling", help="Disable mixed/overlapping sound handling."),
    component_snr_db: float = typer.Option(10.0, "--component-snr-db", help="SNR threshold for time-frequency components."),
    min_purity_for_clustering: float = typer.Option(0.55, "--min-purity-for-clustering", help="Minimum component purity score sent to clustering."),
    max_components_per_event: int = typer.Option(4, "--max-components-per-event", help="Maximum separated components per detected event."),
    disable_noise_filtering: bool = typer.Option(False, "--disable-noise-filtering", help="Disable low-confidence noise routing."),
    noise_mode: str = typer.Option("balanced", "--noise-mode", help="Noise filtering mode: exploratory, balanced, or conservative."),
    min_quality_for_clustering: float = typer.Option(0.45, "--min-quality-for-clustering", help="Minimum acoustic quality score sent to clustering."),
    disable_eventness_filtering: bool = typer.Option(False, "--disable-eventness-filtering", help="Disable low-confidence eventness routing."),
    min_eventness_for_clustering: float = typer.Option(0.28, "--min-eventness-for-clustering", help="Minimum temporal salience score sent to clustering."),
    disable_candidate_selection: bool = typer.Option(False, "--disable-candidate-selection", help="Disable component limiting and temporal NMS candidate pruning."),
    max_components_per_parent: int = typer.Option(3, "--max-components-per-parent", help="Maximum separated components kept per original event parent."),
    disable_short_event_review: bool = typer.Option(False, "--disable-short-event-review", help="Keep very short events in normal clustering."),
    min_review_event_duration: float = typer.Option(0.20, "--min-review-event-duration", help="Route events shorter than this many seconds to secondary review."),
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
        enable_segmentation_refinement=not disable_segmentation_refinement,
        max_events=max_events,
        generate_spectrograms=False if no_clips else not no_spectrograms,
        export_clips=not no_clips,
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
    )
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
