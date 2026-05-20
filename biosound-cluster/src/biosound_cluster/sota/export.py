"""Export SOTA pipeline results into review-ready folders."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from biosound_cluster.audio_io import safe_write_wav
from biosound_cluster.sota.config import SOTAConfig
from biosound_cluster.sota.schemas import SOTACluster, SOTAEvent
from biosound_cluster.visualization import save_spectrogram_png


def _slugify_label(label: str) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in label.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unlabeled"


def _cluster_folder_name(
    prefix: str,
    cluster_id: int,
    size: int,
    label: str | None,
) -> str:
    base = f"{prefix}_{cluster_id:03d}_size_{size:03d}"
    if label:
        return f"{base}_{_slugify_label(label)}"
    return base


def write_event_assets(
    audio: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    events: list[SOTAEvent],
    clusters: list[SOTACluster],
    config: SOTAConfig,
) -> None:
    """Write wav clips and spectrograms under per-cluster folders."""
    if not config.export_clips and not config.export_spectrograms:
        return

    folder_by_cluster = {cluster.cluster_id: cluster.folder_name for cluster in clusters}
    for event in events:
        if event.is_noise:
            folder = config.noise_folder_name
        elif event.is_low_stability:
            folder = config.low_stability_folder_name
        else:
            folder = folder_by_cluster.get(event.cluster_id, config.noise_folder_name)

        cluster_dir = output_dir / folder
        cluster_dir.mkdir(parents=True, exist_ok=True)

        start = max(0, int(round(event.start_sec * sample_rate)))
        end = min(audio.shape[0], int(round(event.end_sec * sample_rate)))
        clip = audio[start:end]
        if clip.size == 0:
            continue

        clip_path = cluster_dir / f"{event.event_id}.wav"
        if config.export_clips:
            safe_write_wav(clip_path, clip, sample_rate)
            event.clip_path = str(clip_path.relative_to(output_dir))
        if config.export_spectrograms:
            spec_path = cluster_dir / f"{event.event_id}.png"
            save_spectrogram_png(clip, sample_rate, spec_path)
            event.spectrogram_path = str(spec_path.relative_to(output_dir))


def write_csvs(
    output_dir: Path,
    events: list[SOTAEvent],
    clusters: list[SOTACluster],
) -> tuple[Path, Path]:
    """Write events.csv and clusters.csv."""
    events_df = pd.DataFrame([event.to_dict() for event in events])
    clusters_df = pd.DataFrame([cluster.to_dict() for cluster in clusters])
    events_csv = output_dir / "events.csv"
    clusters_csv = output_dir / "clusters.csv"
    events_df.to_csv(events_csv, index=False)
    clusters_df.to_csv(clusters_csv, index=False)
    return events_csv, clusters_csv


def write_embeddings(
    output_dir: Path,
    embeddings: np.ndarray,
    start_times: np.ndarray,
) -> Path:
    """Write window-level embeddings and start times as npz."""
    path = output_dir / "embeddings.npz"
    np.savez_compressed(path, embeddings=embeddings, start_times=start_times)
    return path


def write_report(
    output_dir: Path,
    encoder_name: str,
    embedding_dim: int,
    duration_sec: float,
    sample_rate: int,
    events: list[SOTAEvent],
    clusters: list[SOTACluster],
) -> Path:
    """Write a human-readable Markdown report."""
    lines: list[str] = []
    lines.append("# biosound-cluster SOTA report")
    lines.append("")
    lines.append(f"- Encoder: `{encoder_name}` ({embedding_dim}-d)")
    lines.append(f"- Audio duration: {duration_sec:.2f} s @ {sample_rate} Hz")
    lines.append(f"- Total events: {len(events)}")
    lines.append(f"- Clusters: {len(clusters)}")
    lines.append("")
    lines.append("## Clusters")
    lines.append("")
    lines.append("| ID | Size | Folder | Stability | Zero-shot label | Score |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for cluster in clusters:
        label = cluster.zero_shot_label or ""
        score = f"{cluster.zero_shot_score:.3f}" if cluster.zero_shot_score is not None else ""
        lines.append(
            f"| {cluster.cluster_id} | {cluster.size} | `{cluster.folder_name}` | "
            f"{cluster.mean_stability:.3f} | {label} | {score} |"
        )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def write_index_html(
    output_dir: Path,
    encoder_name: str,
    events: list[SOTAEvent],
    clusters: list[SOTACluster],
) -> Path:
    """Write a minimal index.html for browsing the output."""
    folder_by_cluster = {cluster.cluster_id: cluster.folder_name for cluster in clusters}
    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>biosound-cluster SOTA review</title>")
    parts.append("<style>")
    parts.append(
        "body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 12px;}"
        "h2{border-bottom:1px solid #ccc;padding-bottom:4px;}"
        "table{border-collapse:collapse;width:100%;margin-bottom:24px;}"
        "th,td{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px;}"
        "audio{height:32px;}img{max-height:120px;}"
    )
    parts.append("</style></head><body>")
    parts.append(f"<h1>biosound-cluster SOTA review</h1>")
    parts.append(f"<p>Encoder: <code>{escape(encoder_name)}</code></p>")
    for cluster in clusters:
        members = [e for e in events if e.cluster_id == cluster.cluster_id and not e.is_noise]
        if not members:
            continue
        members.sort(key=lambda e: (e.representative_rank if e.representative_rank is not None else 1e9))
        title = escape(cluster.folder_name)
        label = escape(cluster.zero_shot_label or "")
        parts.append(f"<h2>{title}</h2>")
        if label:
            parts.append(f"<p><strong>Zero-shot:</strong> {label} ({cluster.zero_shot_score:.3f})</p>")
        parts.append(f"<p>Size: {cluster.size} — mean stability: {cluster.mean_stability:.3f}</p>")
        parts.append("<table><tr><th>Event</th><th>Time (s)</th><th>Audio</th><th>Spectrogram</th></tr>")
        for event in members[:32]:
            audio_html = ""
            if event.clip_path:
                audio_html = f"<audio controls src='{escape(event.clip_path)}'></audio>"
            spec_html = ""
            if event.spectrogram_path:
                spec_html = f"<img src='{escape(event.spectrogram_path)}'>"
            time_str = f"{event.start_sec:.2f} – {event.end_sec:.2f}"
            parts.append(
                f"<tr><td>{escape(event.event_id)}</td><td>{time_str}</td>"
                f"<td>{audio_html}</td><td>{spec_html}</td></tr>"
            )
        parts.append("</table>")
    parts.append("</body></html>")
    path = output_dir / "index.html"
    path.write_text("".join(parts))
    return path


def write_run_metadata(
    output_dir: Path,
    metadata: dict[str, object],
) -> Path:
    path = output_dir / "run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, default=str))
    return path


def assign_cluster_folders(
    clusters: list[SOTACluster],
    prefix: str,
) -> None:
    """Mutate clusters in-place to fill ``folder_name`` from id/size/label."""
    for cluster in clusters:
        cluster.folder_name = _cluster_folder_name(
            prefix=prefix,
            cluster_id=cluster.cluster_id,
            size=cluster.size,
            label=cluster.zero_shot_label,
        )
