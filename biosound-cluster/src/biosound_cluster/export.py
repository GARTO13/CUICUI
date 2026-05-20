"""Export clips, manifests, reports, and local HTML review pages."""

from __future__ import annotations

import html
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from biosound_cluster.audio_io import safe_write_wav
from biosound_cluster.config import BioSoundConfig, config_fingerprint, config_to_dict
from biosound_cluster.embeddings import extract_event_clip
from biosound_cluster.schemas import AudioEvent, ClusterSummary
from biosound_cluster.visualization import save_spectrogram_png


EVENT_FIELDS = [
    "event_id",
    "start_sec",
    "end_sec",
    "duration_sec",
    "cluster_id",
    "is_noise",
    "cluster_probability",
    "rms_db",
    "peak_db",
    "spectral_centroid",
    "clip_path",
    "spectrogram_path",
    "parent_event_id",
    "component_id",
    "is_component",
    "is_overlapping",
    "is_mixed",
    "n_components",
    "polyphony_score",
    "purity_score",
    "source_type",
    "context_clip_path",
    "is_low_confidence_noise",
    "snr_db",
    "noise_floor_db",
    "spectral_flatness",
    "tonality_score",
    "bandwidth_hz",
    "peak_band_snr_db",
    "quality_score",
    "eventness_score",
    "temporal_contrast_db",
    "active_ratio",
    "is_low_confidence_event",
    "selection_score",
    "is_pruned_candidate",
    "candidate_route_reason",
    "is_short_review_event",
    "local_snr_score",
    "spectral_structure_score",
    "duration_confidence",
    "embedding_stability_score",
    "broadband_noise_penalty",
    "overlap_penalty",
    "edge_case_penalty",
    "clusterability_score",
    "is_ambiguous_review",
    "acoustic_prefamily",
    "cluster_stability_score",
    "representative_score",
    "detection_score",
    "event_snr_db",
    "active_band_fraction",
    "spectral_flux_score",
    "stationarity_score",
    "component_energy_ratio",
    "component_snr_db",
    "component_compactness",
    "component_quality_score",
    "is_component_review",
    "component_rank_in_parent",
    "clusterable",
    "routing_reason",
    "is_low_detection_confidence",
]

CLUSTER_FIELDS = [
    "cluster_id",
    "size",
    "folder_name",
    "mean_probability",
    "representative_event_ids",
    "mean_purity_score",
    "n_component_events",
    "n_original_events",
    "mean_clusterability_score",
    "mean_stability_score",
    "acoustic_prefamily",
]

TOP_LEVEL_OUTPUT_FILES = {
    "events.csv",
    "clusters.csv",
    "report.md",
    "run_metadata.json",
    "event_metadata.json",
    "index.html",
}


def cluster_folder_name(cluster_id: int | None, size: int, is_noise: bool = False) -> str:
    """Return the output folder name for a cluster or noise bucket."""
    if is_noise or cluster_id is None:
        return f"noise_unknown_size_{size:03d}"
    return f"cluster_{cluster_id:03d}_size_{size:03d}"


def event_file_stem(event: AudioEvent) -> str:
    """Return a stable file stem for an exported event."""
    if event.is_component and event.parent_event_id is not None and event.component_id is not None:
        return f"{event.parent_event_id}_component_{event.component_id}__{event.start_sec:.3f}-{event.end_sec:.3f}"
    return f"{event.event_id}__{event.start_sec:.3f}-{event.end_sec:.3f}"


def _event_row(event: AudioEvent) -> dict[str, object]:
    data = event.to_dict()
    return {field: data.get(field) for field in EVENT_FIELDS}


def _cluster_row(cluster: ClusterSummary) -> dict[str, object]:
    data = cluster.to_dict()
    return {field: data.get(field) for field in CLUSTER_FIELDS}


def _write_events_csv(path: Path, events: list[AudioEvent]) -> None:
    rows = [_event_row(event) for event in events]
    pd.DataFrame(rows, columns=EVENT_FIELDS).to_csv(path, index=False)


def _write_clusters_csv(path: Path, clusters: list[ClusterSummary]) -> None:
    rows = [_cluster_row(cluster) for cluster in clusters]
    pd.DataFrame(rows, columns=CLUSTER_FIELDS).to_csv(path, index=False)


def _score_summary(events: list[AudioEvent], field: str) -> str:
    values = [
        getattr(event, field)
        for event in events
        if getattr(event, field) is not None
    ]
    if not values:
        return "n/a"
    array = np.asarray(values, dtype=float)
    return (
        f"mean={np.mean(array):.3f}, median={np.median(array):.3f}, "
        f"p10={np.percentile(array, 10):.3f}, p90={np.percentile(array, 90):.3f}"
    )


def _recording_metadata(input_path: Path, duration_sec: float, sr: int, config: BioSoundConfig) -> dict[str, object]:
    return {
        "input_path": str(input_path),
        "sample_rate": sr,
        "duration_sec": duration_sec,
        "recording_start_time": config.recording_start_time,
        "recording_timezone": config.recording_timezone,
        "sensor": {
            "sensor_id": config.sensor_id,
            "latitude": config.sensor_latitude,
            "longitude": config.sensor_longitude,
            "elevation_m": config.sensor_elevation_m,
        },
        "environment_type": config.environment_type,
    }


def _parse_recording_start(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _time_at_offset(recording_start: datetime | None, offset_sec: float) -> str | None:
    if recording_start is None:
        return None
    return (recording_start + timedelta(seconds=float(offset_sec))).isoformat()


def _event_metadata_row(
    event: AudioEvent,
    input_path: Path,
    duration_sec: float,
    sr: int,
    config: BioSoundConfig,
) -> dict[str, object]:
    recording_start = _parse_recording_start(config.recording_start_time)
    return {
        "event_id": event.event_id,
        "source_type": event.source_type,
        "cluster_id": event.cluster_id,
        "is_noise": event.is_noise,
        "is_mixed": event.is_mixed,
        "is_component": event.is_component,
        "parent_event_id": event.parent_event_id,
        "component_id": event.component_id,
        "recording": _recording_metadata(input_path, duration_sec, sr, config),
        "clip_timing": {
            "start_sec": event.start_sec,
            "end_sec": event.end_sec,
            "duration_sec": event.duration_sec,
            "recording_start_time": config.recording_start_time,
            "clip_start_time": _time_at_offset(recording_start, event.start_sec),
            "clip_end_time": _time_at_offset(recording_start, event.end_sec),
        },
        "files": {
            "clip_path": event.clip_path,
            "spectrogram_path": event.spectrogram_path,
            "context_clip_path": event.context_clip_path,
        },
        "scores": {
            "clusterability_score": event.clusterability_score,
            "embedding_stability_score": event.embedding_stability_score,
            "eventness_score": event.eventness_score,
            "quality_score": event.quality_score,
            "representative_score": event.representative_score,
        },
        "acoustic_prefamily": event.acoustic_prefamily,
    }


def _write_event_sidecar_json(
    path: Path,
    event: AudioEvent,
    input_path: Path,
    duration_sec: float,
    sr: int,
    config: BioSoundConfig,
) -> None:
    metadata = _event_metadata_row(event, input_path, duration_sec, sr, config)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_context_sidecar_json(
    path: Path,
    event: AudioEvent,
    input_path: Path,
    duration_sec: float,
    sr: int,
    config: BioSoundConfig,
) -> None:
    metadata = _event_metadata_row(event, input_path, duration_sec, sr, config)
    metadata["audio_role"] = "original_context"
    metadata["files"]["separated_component_clip_path"] = event.clip_path
    metadata["files"]["clip_path"] = event.context_clip_path
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_event_metadata_json(
    path: Path,
    events: list[AudioEvent],
    input_path: Path,
    duration_sec: float,
    sr: int,
    config: BioSoundConfig,
) -> None:
    payload = {
        "recording": _recording_metadata(input_path, duration_sec, sr, config),
        "events": [
            _event_metadata_row(event, input_path, duration_sec, sr, config)
            for event in events
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_report(
    path: Path,
    input_path: Path,
    duration_sec: float,
    events: list[AudioEvent],
    clusters: list[ClusterSummary],
    n_noise: int,
    mixed_events: list[AudioEvent],
    low_confidence_noise_events: list[AudioEvent],
    short_review_events: list[AudioEvent],
    component_review_events: list[AudioEvent],
    config: BioSoundConfig,
) -> None:
    n_components = sum(1 for event in events if event.is_component)
    n_component_review = len(component_review_events)
    n_originals = sum(1 for event in events if event.source_type == "original")
    all_events = events + mixed_events + low_confidence_noise_events + short_review_events + component_review_events
    ambiguous_events = [event for event in low_confidence_noise_events if event.source_type == "ambiguous_review"]
    cluster_sizes = [cluster.size for cluster in clusters]
    mixed_folder = f"mixed_overlapping_size_{len(mixed_events):03d}" if mixed_events else "none"
    low_noise_folder = (
        f"low_confidence_noise_size_{len(low_confidence_noise_events):03d}"
        if low_confidence_noise_events
        else "none"
    )
    short_review_folder = (
        f"short_events_review_size_{len(short_review_events):03d}"
        if short_review_events
        else "none"
    )
    component_review_folder = (
        f"{config.component_review_folder_name}_size_{len(component_review_events):03d}"
        if component_review_events
        else "none"
    )
    lines = [
        "# biosound-cluster report",
        "",
        "This tool performs unsupervised acoustic grouping. Cluster IDs are not species names.",
        "HDBSCAN noise points are not necessarily useless; they may contain rare sounds, ambiguous sounds, or actual noise.",
        "Human validation is required. The purpose is to reduce expert listening time by grouping similar events.",
        "",
        "## Run summary",
        "",
        f"- Input file: `{input_path}`",
        f"- Audio duration: {duration_sec:.2f} seconds",
        f"- Exported events: {len(events) + len(mixed_events) + len(low_confidence_noise_events) + len(short_review_events)}",
        f"- Events sent to normal clustering: {len(events)}",
        f"- Events routed to review/low confidence: {len(mixed_events) + len(low_confidence_noise_events) + len(short_review_events)}",
        f"- Acoustic clusters: {len(clusters)}",
        f"- Noise/unknown events: {n_noise}",
        f"- Low-confidence noise events excluded from clustering: {len(low_confidence_noise_events)}",
        f"- Component review events excluded from clustering: {n_component_review}",
        f"- Ambiguous review events excluded from clustering: {len(ambiguous_events)}",
        f"- Short review events excluded from main clusters: {len(short_review_events)}",
        f"- Mean cluster size: {float(np.mean(cluster_sizes)):.2f}" if cluster_sizes else "- Mean cluster size: n/a",
        f"- Median cluster size: {float(np.median(cluster_sizes)):.2f}" if cluster_sizes else "- Median cluster size: n/a",
        "",
        "## Clusterability and stability",
        "",
        f"- Clusterability filtering enabled: {str(config.enable_clusterability_filtering).lower()}",
        f"- Minimum clusterability for clustering: {config.min_clusterability_for_clustering:.3f}",
        f"- Clusterability score distribution: {_score_summary(all_events, 'clusterability_score')}",
        f"- Embedding stability enabled: {str(config.enable_embedding_stability).lower()}",
        f"- Embedding stability distribution: {_score_summary(all_events, 'embedding_stability_score')}",
        f"- Acoustic pre-families enabled: {str(config.enable_acoustic_prefamilies).lower()}",
        "",
        "## Noise filtering",
        "",
        f"- Noise filtering enabled: {str(config.enable_noise_filtering).lower()}",
        f"- Noise mode: `{config.noise_mode}`",
        f"- Minimum quality for clustering: {config.min_quality_for_clustering:.3f}",
        f"- Low-confidence noise folder: `{low_noise_folder}`",
        f"- Short events review folder: `{short_review_folder}`",
        "",
        "Low-confidence noise events are not deleted. They are isolated for expert review because they are likely to be broadband, low-SNR, or weakly structured sounds that would reduce cluster purity.",
        "",
        "Short review events may contain real biological signals, but they are often too short to be reliable cluster representatives. They are exported separately for secondary review.",
        "",
        "## Polyphony handling",
        "",
        f"- Polyphony handling enabled: {str(config.enable_polyphony_handling).lower()}",
        f"- Mixed overlapping events excluded from clustering: {len(mixed_events)}",
        f"- Separated component events sent to clustering: {n_components}",
        f"- Separated component events routed to review: {n_component_review}",
        f"- Original clean events sent to clustering: {n_originals}",
        f"- Mixed folder: `{mixed_folder}`",
        "",
        "Mixed events are not necessarily useless. They may contain real biological signals, but they require expert review because several acoustic sources overlap.",
        "",
        "## Component review",
        "",
        f"- Component events sent to clustering: {n_components}",
        f"- Component events routed to review: {n_component_review}",
        f"- Parents marked mixed instead of split: {len(mixed_events)}",
        f"- Average component quality score: {_score_summary([event for event in events + component_review_events if event.is_component], 'component_quality_score')}",
        f"- Component review folder: `{component_review_folder}`",
        "",
    ]
    if not events and not mixed_events and not low_confidence_noise_events and not short_review_events:
        lines.extend(
            [
                "## No Events Detected",
                "",
                "No candidate acoustic events were detected with the current settings.",
                "Try lowering `threshold_db`, lowering `min_event_duration`, or inspecting the input audio.",
                "",
            ]
        )
    else:
        lines.extend(["## Clusters", ""])
        for cluster in clusters:
            reps = ", ".join(cluster.representative_event_ids) or "none"
            lines.append(
                f"- `{cluster.folder_name}`: {cluster.size} events; "
                f"prefamily={cluster.acoustic_prefamily or 'unknown'}; representatives: {reps}"
            )
        if n_noise:
            lines.append(f"- `noise_unknown_size_{n_noise:03d}`: {n_noise} events")
        if mixed_events:
            lines.append(f"- `mixed_overlapping_size_{len(mixed_events):03d}`: {len(mixed_events)} mixed/overlapping events excluded from clustering")
        if low_confidence_noise_events:
            lines.append(f"- `low_confidence_noise_size_{len(low_confidence_noise_events):03d}`: {len(low_confidence_noise_events)} low-confidence noise events excluded from clustering")
        if component_review_events:
            lines.append(f"- `{component_review_folder}`: {len(component_review_events)} separated components excluded from normal clustering")
        if short_review_events:
            lines.append(f"- `short_events_review_size_{len(short_review_events):03d}`: {len(short_review_events)} short events excluded from main cluster representatives")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _representative_html(event: AudioEvent, output_dir: Path) -> str:
    clip = html.escape(str(Path(event.clip_path).as_posix())) if event.clip_path else ""
    image = html.escape(str(Path(event.spectrogram_path or "").as_posix())) if event.spectrogram_path else ""
    title = html.escape(f"{event.event_id} {event.start_sec:.3f}-{event.end_sec:.3f}s")
    source = html.escape(event.source_type)
    score_bits = []
    if event.representative_score is not None:
        score_bits.append(f"rep={event.representative_score:.3f}")
    if event.clusterability_score is not None:
        score_bits.append(f"clusterability={event.clusterability_score:.3f}")
    if event.embedding_stability_score is not None:
        score_bits.append(f"stability={event.embedding_stability_score:.3f}")
    if event.acoustic_prefamily:
        score_bits.append(f"family={event.acoustic_prefamily}")
    score_html = f'<p class="source">{" | ".join(html.escape(bit) for bit in score_bits)}</p>' if score_bits else ""
    context = ""
    if event.context_clip_path:
        context_path = html.escape(str(Path(event.context_clip_path).as_posix()))
        context = f'<p><a href="{context_path}">original context</a></p>'
    image_html = f'<img src="{image}" alt="Spectrogram for {title}">' if image else ""
    audio_html = f'<audio controls src="{clip}"></audio>' if clip else '<p class="source">clip export disabled</p>'
    return (
        '<div class="rep">'
        f"<h3>{title}</h3>"
        f'<p class="source">source: {source}</p>'
        f"{score_html}"
        f"{image_html}"
        f"{audio_html}"
        f"{context}"
        "</div>"
    )


def _write_index_html(
    path: Path,
    input_path: Path,
    duration_sec: float,
    events: list[AudioEvent],
    clusters: list[ClusterSummary],
    n_noise: int,
    mixed_events: list[AudioEvent],
    low_confidence_noise_events: list[AudioEvent],
    short_review_events: list[AudioEvent],
    component_review_events: list[AudioEvent],
) -> None:
    events_by_id = {event.event_id: event for event in events}
    cards: list[str] = []
    for cluster in clusters:
        reps = [events_by_id[event_id] for event_id in cluster.representative_event_ids if event_id in events_by_id]
        reps_html = "\n".join(_representative_html(event, path.parent) for event in reps)
        probability_text = (
            f"{cluster.size} events | mean membership probability: {cluster.mean_probability:.3f}"
            if cluster.mean_probability is not None
            else f"{cluster.size} events"
        )
        cards.append(
            '<section class="cluster">'
            f"<h2>{html.escape(cluster.folder_name)}</h2>"
            f"<p>{probability_text}</p>"
            f'<p><a href="{html.escape(cluster.folder_name)}/_cluster_manifest.csv">manifest</a></p>'
            f'<div class="reps">{reps_html}</div>'
            "</section>"
        )
    if n_noise:
        noise_folder = f"noise_unknown_size_{n_noise:03d}"
        cards.append(
            '<section class="cluster">'
            f"<h2>{noise_folder}</h2>"
            f"<p>{n_noise} noise/unknown events. These may include rare sounds, ambiguous sounds, or actual noise.</p>"
            f'<p><a href="{noise_folder}/_cluster_manifest.csv">manifest</a></p>'
            "</section>"
        )
    if mixed_events:
        mixed_folder = f"mixed_overlapping_size_{len(mixed_events):03d}"
        reps_html = "\n".join(_representative_html(event, path.parent) for event in mixed_events[:5])
        cards.append(
            '<section class="cluster mixed">'
            "<h2>Mixed / overlapping sounds</h2>"
            f"<p>{len(mixed_events)} events excluded from normal clustering because several acoustic sources appear to overlap.</p>"
            f'<p><a href="{mixed_folder}/_cluster_manifest.csv">manifest</a></p>'
            f'<div class="reps">{reps_html}</div>'
            "</section>"
        )
    low_detection_events = [event for event in low_confidence_noise_events if event.source_type == "low_detection_confidence"]
    low_confidence_review_events = [event for event in low_confidence_noise_events if event.source_type != "low_detection_confidence"]
    if low_confidence_review_events:
        low_noise_folder = f"low_confidence_noise_size_{len(low_confidence_review_events):03d}"
        ambiguous_count = sum(1 for event in low_confidence_review_events if event.source_type == "ambiguous_review")
        reps_html = "\n".join(_representative_html(event, path.parent) for event in low_confidence_review_events[:5])
        cards.append(
            '<section class="cluster low-noise">'
            "<h2>Low-confidence / ambiguous review</h2>"
            f"<p>{len(low_confidence_review_events)} events excluded from normal clustering; {ambiguous_count} were marked ambiguous by clusterability scoring.</p>"
            f'<p><a href="{low_noise_folder}/_cluster_manifest.csv">manifest</a></p>'
            f'<div class="reps">{reps_html}</div>'
            "</section>"
        )
    if low_detection_events:
        low_detection_folder = f"low_detection_confidence_size_{len(low_detection_events):03d}"
        reps_html = "\n".join(_representative_html(event, path.parent) for event in low_detection_events[:5])
        cards.append(
            '<section class="cluster low-detection">'
            "<h2>Low detection confidence</h2>"
            f"<p>{len(low_detection_events)} events excluded from normal clustering by detection-score routing.</p>"
            f'<p><a href="{low_detection_folder}/_cluster_manifest.csv">manifest</a></p>'
            f'<div class="reps">{reps_html}</div>'
            "</section>"
        )
    if short_review_events:
        short_folder = f"short_events_review_size_{len(short_review_events):03d}"
        reps_html = "\n".join(_representative_html(event, path.parent) for event in short_review_events[:5])
        cards.append(
            '<section class="cluster short-review">'
            "<h2>Short events review</h2>"
            f"<p>{len(short_review_events)} short events excluded from main clusters because they may be poor representatives for quick human review.</p>"
            f'<p><a href="{short_folder}/_cluster_manifest.csv">manifest</a></p>'
            f'<div class="reps">{reps_html}</div>'
            "</section>"
        )
    if component_review_events:
        component_folder = f"component_review_size_{len(component_review_events):03d}"
        reps_html = "\n".join(_representative_html(event, path.parent) for event in component_review_events[:5])
        cards.append(
            '<section class="cluster component-review">'
            "<h2>Component review</h2>"
            f"<p>{len(component_review_events)} separated components excluded from normal clustering by conservative routing.</p>"
            f'<p><a href="{component_folder}/_cluster_manifest.csv">manifest</a></p>'
            f'<div class="reps">{reps_html}</div>'
            "</section>"
        )
    if not cards:
        cards.append("<section class=\"cluster\"><h2>No candidate events detected</h2></section>")

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>biosound-cluster review</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f7f7f4; color: #202124; }}
    header {{ background: #193b3f; color: white; padding: 28px; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    a {{ color: #0b5f6a; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ background: white; border: 1px solid #d8ddd9; border-radius: 8px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 1.45rem; }}
    .cluster {{ background: white; border: 1px solid #d8ddd9; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    .reps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    .rep {{ border-top: 1px solid #e5e7e4; padding-top: 12px; }}
    .rep h3 {{ font-size: 0.95rem; font-weight: 650; margin: 0 0 8px; }}
    img {{ width: 100%; height: auto; border: 1px solid #e1e2df; border-radius: 6px; }}
    audio {{ width: 100%; margin-top: 8px; }}
    code {{ background: #edf0ed; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>biosound-cluster review</h1>
    <p>Unsupervised acoustic grouping. Cluster IDs are not species labels; human validation is required.</p>
  </header>
  <main>
    <p><strong>Input:</strong> <code>{html.escape(str(input_path))}</code></p>
    <div class="summary">
      <div class="metric"><span>Duration</span><strong>{duration_sec:.2f}s</strong></div>
      <div class="metric"><span>Events</span><strong>{len(events) + len(mixed_events) + len(low_confidence_noise_events) + len(short_review_events) + len(component_review_events)}</strong></div>
      <div class="metric"><span>Clusters</span><strong>{len(clusters)}</strong></div>
      <div class="metric"><span>Noise/unknown</span><strong>{n_noise}</strong></div>
      <div class="metric"><span>Mixed/overlapping</span><strong>{len(mixed_events)}</strong></div>
      <div class="metric"><span>Low-confidence noise</span><strong>{len(low_confidence_noise_events)}</strong></div>
      <div class="metric"><span>Ambiguous review</span><strong>{sum(1 for event in low_confidence_noise_events if event.source_type == "ambiguous_review")}</strong></div>
      <div class="metric"><span>Short review</span><strong>{len(short_review_events)}</strong></div>
      <div class="metric"><span>Component review</span><strong>{len(component_review_events)}</strong></div>
    </div>
    <p><a href="events.csv">events.csv</a> | <a href="clusters.csv">clusters.csv</a> | <a href="report.md">report.md</a></p>
    {"".join(cards)}
  </main>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def _export_event_media(
    audio: np.ndarray,
    sr: int,
    event: AudioEvent,
    root: Path,
    folder: Path,
    config: BioSoundConfig,
    input_path: Path,
    duration_sec: float,
) -> None:
    if not config.export_clips:
        event.clip_path = None
        event.spectrogram_path = None
        return

    stem = event_file_stem(event)
    clip_path = folder / f"{stem}.wav"
    png_path = folder / f"{stem}.png"
    clip = extract_event_clip(audio, sr, event)
    safe_write_wav(clip_path, clip, sr)
    event.clip_path = str(clip_path.relative_to(root))
    if config.generate_spectrograms:
        save_spectrogram_png(clip, sr, png_path)
        event.spectrogram_path = str(png_path.relative_to(root))

    if event.is_component and config.export_original_context and event.context_audio is not None:
        context_path = folder / f"{stem}__context_original.wav"
        safe_write_wav(context_path, event.context_audio, sr)
        event.context_clip_path = str(context_path.relative_to(root))
        if config.generate_spectrograms:
            context_png = folder / f"{stem}__context_original.png"
            save_spectrogram_png(event.context_audio, sr, context_png)
        context_json = folder / f"{stem}__context_original.json"
        _write_context_sidecar_json(context_json, event, input_path, duration_sec, sr, config)

    json_path = folder / f"{stem}.json"
    _write_event_sidecar_json(json_path, event, input_path, duration_sec, sr, config)


def export_outputs(
    audio: np.ndarray,
    sr: int,
    input_path: str | Path,
    output_dir: str | Path,
    events: list[AudioEvent],
    clusters: list[ClusterSummary],
    config: BioSoundConfig,
    duration_sec: float,
    mixed_events: list[AudioEvent] | None = None,
    low_confidence_noise_events: list[AudioEvent] | None = None,
    short_review_events: list[AudioEvent] | None = None,
    component_review_events: list[AudioEvent] | None = None,
) -> dict[str, Path]:
    """Export event clips, spectrograms, metadata tables, report, and HTML index."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)
    _clean_previous_outputs(root)
    mixed_events = mixed_events or []
    low_confidence_noise_events = low_confidence_noise_events or []
    short_review_events = short_review_events or []
    component_review_events = component_review_events or []

    sizes: dict[int | None, int] = {}
    for event in events:
        key = None if event.is_noise else event.cluster_id
        sizes[key] = sizes.get(key, 0) + 1

    folder_by_key: dict[int | None, Path] = {
        cluster.cluster_id: root / cluster.folder_name for cluster in clusters
    }
    if sizes.get(None, 0):
        folder_by_key[None] = root / cluster_folder_name(None, sizes[None], is_noise=True)
    mixed_folder = root / f"mixed_overlapping_size_{len(mixed_events):03d}" if mixed_events else None
    if mixed_folder is not None and config.export_mixed_overlapping:
        mixed_folder.mkdir(parents=True, exist_ok=True)
    low_detection_events = [
        event for event in low_confidence_noise_events if event.source_type == "low_detection_confidence"
    ]
    low_confidence_review_events = [
        event for event in low_confidence_noise_events if event.source_type != "low_detection_confidence"
    ]
    low_noise_folder = (
        root / f"low_confidence_noise_size_{len(low_confidence_review_events):03d}"
        if low_confidence_review_events
        else None
    )
    if low_noise_folder is not None and config.export_low_confidence_noise:
        low_noise_folder.mkdir(parents=True, exist_ok=True)
    low_detection_folder = (
        root / f"low_detection_confidence_size_{len(low_detection_events):03d}"
        if low_detection_events
        else None
    )
    if low_detection_folder is not None:
        low_detection_folder.mkdir(parents=True, exist_ok=True)
    short_review_folder = (
        root / f"short_events_review_size_{len(short_review_events):03d}"
        if short_review_events
        else None
    )
    if short_review_folder is not None and config.export_short_events_review:
        short_review_folder.mkdir(parents=True, exist_ok=True)
    component_review_folder = (
        root / f"{config.component_review_folder_name}_size_{len(component_review_events):03d}"
        if component_review_events
        else None
    )
    if component_review_folder is not None:
        component_review_folder.mkdir(parents=True, exist_ok=True)

    for folder in folder_by_key.values():
        folder.mkdir(parents=True, exist_ok=True)

    for event in events:
        key = None if event.is_noise else event.cluster_id
        folder = folder_by_key[key]
        _export_event_media(audio, sr, event, root, folder, config, input_path, duration_sec)

    if mixed_folder is not None and config.export_mixed_overlapping:
        for event in mixed_events:
            _export_event_media(audio, sr, event, root, mixed_folder, config, input_path, duration_sec)
    if low_noise_folder is not None and config.export_low_confidence_noise:
        for event in low_confidence_review_events:
            _export_event_media(audio, sr, event, root, low_noise_folder, config, input_path, duration_sec)
    if low_detection_folder is not None:
        for event in low_detection_events:
            _export_event_media(audio, sr, event, root, low_detection_folder, config, input_path, duration_sec)
    if short_review_folder is not None and config.export_short_events_review:
        for event in short_review_events:
            _export_event_media(audio, sr, event, root, short_review_folder, config, input_path, duration_sec)
    if component_review_folder is not None:
        for event in component_review_events:
            _export_event_media(audio, sr, event, root, component_review_folder, config, input_path, duration_sec)

    events_by_id = {event.event_id: event for event in events}
    for cluster in clusters:
        folder = root / cluster.folder_name
        representatives_dir = folder / "_representatives"
        representatives_dir.mkdir(exist_ok=True)
        for event_id in cluster.representative_event_ids:
            event = events_by_id.get(event_id)
            if event is None or event.clip_path is None:
                continue
            source_wav = root / event.clip_path
            shutil.copy2(source_wav, representatives_dir / source_wav.name)
            if event.spectrogram_path:
                source_png = root / event.spectrogram_path
                shutil.copy2(source_png, representatives_dir / source_png.name)
            source_json = source_wav.with_suffix(".json")
            if source_json.exists():
                shutil.copy2(source_json, representatives_dir / source_json.name)

    for key, folder in folder_by_key.items():
        cluster_events = [
            event for event in events if (None if event.is_noise else event.cluster_id) == key
        ]
        pd.DataFrame([_event_row(event) for event in cluster_events], columns=EVENT_FIELDS).to_csv(
            folder / "_cluster_manifest.csv",
            index=False,
        )
    if mixed_folder is not None and config.export_mixed_overlapping:
        pd.DataFrame([_event_row(event) for event in mixed_events], columns=EVENT_FIELDS).to_csv(
            mixed_folder / "_cluster_manifest.csv",
            index=False,
        )
    if low_noise_folder is not None and config.export_low_confidence_noise:
        pd.DataFrame([_event_row(event) for event in low_confidence_review_events], columns=EVENT_FIELDS).to_csv(
            low_noise_folder / "_cluster_manifest.csv",
            index=False,
        )
    if low_detection_folder is not None:
        pd.DataFrame([_event_row(event) for event in low_detection_events], columns=EVENT_FIELDS).to_csv(
            low_detection_folder / "_cluster_manifest.csv",
            index=False,
        )
    if short_review_folder is not None and config.export_short_events_review:
        pd.DataFrame([_event_row(event) for event in short_review_events], columns=EVENT_FIELDS).to_csv(
            short_review_folder / "_cluster_manifest.csv",
            index=False,
        )
    if component_review_folder is not None:
        pd.DataFrame([_event_row(event) for event in component_review_events], columns=EVENT_FIELDS).to_csv(
            component_review_folder / "_cluster_manifest.csv",
            index=False,
        )

    events_csv = root / "events.csv"
    clusters_csv = root / "clusters.csv"
    report_md = root / "report.md"
    index_html = root / "index.html"
    metadata_json = root / "run_metadata.json"
    event_metadata_json = root / "event_metadata.json"
    n_noise = sum(1 for event in events if event.is_noise) + len(low_confidence_noise_events)
    all_events = events + mixed_events + low_confidence_noise_events + short_review_events + component_review_events

    _write_events_csv(events_csv, all_events)
    _write_clusters_csv(clusters_csv, clusters)
    _write_event_metadata_json(event_metadata_json, all_events, input_path, duration_sec, sr, config)
    metadata = {
        "input_path": str(input_path),
        "output_dir": str(root),
        "recording_metadata": _recording_metadata(input_path, duration_sec, sr, config),
        "duration_sec": duration_sec,
        "sample_rate": sr,
        "n_events": len(all_events),
        "n_clusters": len(clusters),
        "n_noise": n_noise,
        "n_mixed": len(mixed_events),
        "n_low_confidence_noise": len(low_confidence_noise_events),
        "n_ambiguous_review": sum(1 for event in low_confidence_noise_events if event.source_type == "ambiguous_review"),
        "n_short_review": len(short_review_events),
        "n_component_review": len(component_review_events),
        "n_component_events": sum(1 for event in all_events if event.is_component),
        "n_component_events_clusterable": sum(1 for event in events if event.is_component),
        "n_original_events": sum(1 for event in events if event.source_type == "original"),
        "clusterability_score_summary": _score_summary(all_events, "clusterability_score"),
        "embedding_stability_score_summary": _score_summary(all_events, "embedding_stability_score"),
        "config": config_to_dict(config),
        "config_hash": config_fingerprint(config),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_report(report_md, input_path, duration_sec, events, clusters, n_noise, mixed_events, low_confidence_noise_events, short_review_events, component_review_events, config)
    _write_index_html(index_html, input_path, duration_sec, events, clusters, n_noise, mixed_events, low_confidence_noise_events, short_review_events, component_review_events)

    return {
        "events_csv": events_csv,
        "clusters_csv": clusters_csv,
        "report_md": report_md,
        "index_html": index_html,
        "run_metadata_json": metadata_json,
        "event_metadata_json": event_metadata_json,
    }


def _clean_previous_outputs(root: Path) -> None:
    """Remove files and folders generated by a previous run in this output directory."""
    for file_name in TOP_LEVEL_OUTPUT_FILES:
        path = root / file_name
        if path.exists() and path.is_file():
            path.unlink()
    for child in root.iterdir():
        if child.is_dir() and (
            child.name.startswith("cluster_")
            or child.name.startswith("noise_unknown")
            or child.name.startswith("mixed_overlapping")
            or child.name.startswith("low_confidence_noise")
            or child.name.startswith("low_detection_confidence")
            or child.name.startswith("short_events_review")
            or child.name.startswith("component_review")
        ):
            shutil.rmtree(child)
