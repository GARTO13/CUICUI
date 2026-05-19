"""End-to-end audio processing pipeline."""

from __future__ import annotations

from pathlib import Path

from biosound_cluster.audio_io import get_audio_duration, load_audio
from biosound_cluster.candidate_selection import select_clusterable_candidates
from biosound_cluster.clustering import cluster_embeddings
from biosound_cluster.config import BioSoundConfig
from biosound_cluster.embeddings import extract_embeddings
from biosound_cluster.eventness import analyze_and_route_eventness
from biosound_cluster.export import export_outputs
from biosound_cluster.logging_utils import get_logger
from biosound_cluster.noise import analyze_and_route_noise
from biosound_cluster.polyphony import analyze_and_split_events
from biosound_cluster.review_routing import route_short_review_events
from biosound_cluster.schemas import ProcessResult
from biosound_cluster.segmentation import detect_candidate_events
from biosound_cluster.segmentation_refinement import refine_candidate_events

LOGGER = get_logger(__name__)


def process_audio_file(
    input_path: str | Path,
    output_dir: str | Path,
    config: BioSoundConfig | None = None,
) -> ProcessResult:
    """Process one audio file into acoustic cluster folders and review metadata."""
    cfg = config or BioSoundConfig()
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    LOGGER.info("Loading audio: %s", input_path)
    audio, sr = load_audio(input_path, cfg.sample_rate)
    duration_sec = get_audio_duration(audio, sr)

    LOGGER.info("Detecting candidate acoustic events")
    events = detect_candidate_events(audio, sr, cfg)
    LOGGER.info("Detected %d candidate events", len(events))
    if events and cfg.enable_segmentation_refinement:
        before = len(events)
        LOGGER.info("Refining event boundaries and removing temporal duplicates")
        events = refine_candidate_events(audio, sr, events, cfg)
        LOGGER.info("Segmentation refinement: %d -> %d candidate events", before, len(events))

    mixed_events = []
    low_confidence_noise_events = []
    short_review_events = []
    clusterable_events = events
    if events and cfg.enable_polyphony_handling:
        LOGGER.info("Analyzing polyphony and splitting separable overlapping sounds")
        clusterable_events, mixed_events = analyze_and_split_events(audio, sr, events, cfg)
        LOGGER.info(
            "Polyphony routing: %d clusterable events, %d mixed/overlapping events",
            len(clusterable_events),
            len(mixed_events),
        )

    if clusterable_events and cfg.enable_noise_filtering:
        LOGGER.info("Scoring event noise confidence and routing low-quality events")
        clusterable_events, low_confidence_noise_events = analyze_and_route_noise(audio, sr, clusterable_events, cfg)
        LOGGER.info(
            "Noise routing: %d clusterable events, %d low-confidence noise events",
            len(clusterable_events),
            len(low_confidence_noise_events),
        )

    if clusterable_events and cfg.enable_eventness_filtering:
        LOGGER.info("Scoring eventness and routing weak candidates")
        clusterable_events, low_confidence_event_events = analyze_and_route_eventness(audio, sr, clusterable_events, cfg)
        low_confidence_noise_events.extend(low_confidence_event_events)
        LOGGER.info(
            "Eventness routing: %d clusterable events, %d low-confidence candidates",
            len(clusterable_events),
            len(low_confidence_event_events),
        )

    if clusterable_events and cfg.enable_candidate_selection:
        LOGGER.info("Selecting best candidates and pruning temporal duplicates")
        clusterable_events, pruned_candidates = select_clusterable_candidates(clusterable_events, cfg)
        low_confidence_noise_events.extend(pruned_candidates)
        LOGGER.info(
            "Candidate selection: %d clusterable events, %d pruned review candidates",
            len(clusterable_events),
            len(pruned_candidates),
        )

    if clusterable_events and cfg.enable_short_event_review:
        LOGGER.info("Routing short events to secondary review")
        clusterable_events, short_review_events = route_short_review_events(clusterable_events, cfg)
        LOGGER.info(
            "Short-event review routing: %d clusterable events, %d short review events",
            len(clusterable_events),
            len(short_review_events),
        )

    clusters = []
    if clusterable_events:
        LOGGER.info("Extracting handcrafted acoustic embeddings")
        embeddings = extract_embeddings(audio, sr, clusterable_events, cfg)
        LOGGER.info("Clustering events with UMAP + HDBSCAN")
        clusterable_events, clusters, _ = cluster_embeddings(embeddings, clusterable_events, cfg)
    else:
        LOGGER.info("No clusterable events detected; writing outputs")

    if cfg.export_clips:
        LOGGER.info("Exporting clips, metadata, report, and HTML index")
    else:
        LOGGER.info("Exporting metadata, report, and HTML index without WAV/PNG media")
    paths = export_outputs(
        audio,
        sr,
        input_path,
        output_dir,
        clusterable_events,
        clusters,
        cfg,
        duration_sec,
        mixed_events=mixed_events,
        low_confidence_noise_events=low_confidence_noise_events,
        short_review_events=short_review_events,
    )
    n_noise = sum(1 for event in clusterable_events if event.is_noise) + len(low_confidence_noise_events)

    result = ProcessResult(
        input_path=str(input_path),
        output_dir=str(output_dir),
        duration_sec=duration_sec,
        sample_rate=sr,
        n_events=len(clusterable_events) + len(mixed_events) + len(low_confidence_noise_events) + len(short_review_events),
        n_clusters=len(clusters),
        n_noise=n_noise,
        events_csv=str(paths["events_csv"]),
        clusters_csv=str(paths["clusters_csv"]),
        report_md=str(paths["report_md"]),
        index_html=str(paths["index_html"]),
    )
    LOGGER.info("Done: %s", output_dir)
    return result
