"""Typer CLI for the SOTA pipeline."""

from __future__ import annotations

from pathlib import Path

import typer

from biosound_cluster.sota.config import SOTAConfig
from biosound_cluster.sota.pipeline import process_audio_file_sota


app = typer.Typer(add_completion=False, help="biosound-cluster SOTA pipeline.")


@app.command()
def run(
    input_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    output: Path = typer.Option(Path("outputs/sota_run"), "--output", "-o", help="Output directory."),
    encoder: str = typer.Option("perch", "--encoder", help="Encoder: perch, aves, birdnet, mock."),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, mps."),
    batch_size: int = typer.Option(32, "--batch-size"),
    hop_sec: float = typer.Option(1.0, "--hop-sec"),
    window_sec: float | None = typer.Option(None, "--window-sec", help="Override encoder native window."),
    silence_db: float = typer.Option(-55.0, "--silence-db"),
    knn: int = typer.Option(15, "--knn"),
    resolution: float = typer.Option(1.0, "--resolution"),
    min_cluster_size: int = typer.Option(5, "--min-cluster-size"),
    no_zero_shot: bool = typer.Option(False, "--no-zero-shot"),
    no_clips: bool = typer.Option(False, "--no-clips"),
    no_spectrograms: bool = typer.Option(False, "--no-spectrograms"),
    few_shot_labels: Path | None = typer.Option(
        None, "--few-shot-labels", help="JSON file: {event_id: label}."
    ),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
) -> None:
    """Run the SOTA pipeline on a single audio file."""
    config = SOTAConfig(
        encoder=encoder,  # type: ignore[arg-type]
        encoder_device=device,  # type: ignore[arg-type]
        encoder_batch_size=batch_size,
        encoder_cache_dir=str(cache_dir) if cache_dir else None,
        window_sec=window_sec,
        hop_sec=hop_sec,
        silence_rms_db=silence_db,
        knn_neighbors=knn,
        leiden_resolution=resolution,
        min_cluster_size=min_cluster_size,
        enable_zero_shot=not no_zero_shot,
        export_clips=not no_clips,
        export_spectrograms=not no_spectrograms,
        enable_few_shot=few_shot_labels is not None,
        few_shot_labels_path=str(few_shot_labels) if few_shot_labels else None,
    )
    result = process_audio_file_sota(input_path, output, config)
    typer.echo(
        f"\nDone. {result.n_events} events, {result.n_clusters} clusters "
        f"({result.n_noise_events} noise).\n"
        f"Output: {result.output_dir}\n"
        f"Report: {result.report_md}\n"
        f"Index : {result.index_html}\n"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
