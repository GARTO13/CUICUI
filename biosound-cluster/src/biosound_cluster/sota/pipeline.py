"""End-to-end SOTA pipeline orchestrator.

Pipeline:
    1. Load audio at encoder-native sample rate.
    2. Slice into overlapping windows; mark silent windows.
    3. Embed loud windows with a pretrained bioacoustic encoder.
    4. Build k-NN graph + Leiden communities + per-window stability.
    5. Convert contiguous window runs into events.
    6. Temporal NMS + onset/offset refinement.
    7. Score and select representatives per cluster.
    8. (optional) zero-shot caption each cluster with BioLingual.
    9. (optional) few-shot prototype refinement from human labels.
    10. Export clips, spectrograms, CSVs, report, index.html.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from biosound_cluster.audio_io import get_audio_duration, load_audio
from biosound_cluster.sota.config import SOTAConfig
from biosound_cluster.sota.encoders import load_encoder
from biosound_cluster.sota.event_extraction import (
    WindowEvent,
    refine_boundaries,
    runs_to_events,
    temporal_nms,
)
from biosound_cluster.sota.export import (
    assign_cluster_folders,
    write_csvs,
    write_embeddings,
    write_event_assets,
    write_index_html,
    write_report,
    write_run_metadata,
)
from biosound_cluster.sota.few_shot import (
    compute_prototypes,
    load_labels_from_json,
    reassign,
)
from biosound_cluster.sota.graph_clustering import cluster_embeddings
from biosound_cluster.sota.schemas import SOTACluster, SOTAEvent, SOTAResult
from biosound_cluster.sota.windowing import extract_windows, silence_mask, window_rms_db
from biosound_cluster.sota.zero_shot import ZeroShotResult, caption_clusters


def process_audio_file_sota(
    input_path: str | Path,
    output_dir: str | Path,
    config: SOTAConfig | None = None,
) -> SOTAResult:
    """Run the SOTA pipeline on a single audio file."""
    if config is None:
        config = SOTAConfig()

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    encoder = load_encoder(config)
    encoder_info = encoder.info
    sample_rate = encoder_info.sample_rate
    window_sec = float(config.window_sec) if config.window_sec else encoder_info.window_sec

    audio, sr = load_audio(input_path, sample_rate)
    duration = get_audio_duration(audio, sr)

    grid = extract_windows(audio, sr, window_sec, config.hop_sec)
    rms_db = window_rms_db(grid.windows)
    loud_mask = silence_mask(grid.windows, config.silence_rms_db)
    loud_indices = np.flatnonzero(loud_mask)

    if loud_indices.size == 0:
        embeddings = np.zeros((0, encoder_info.embedding_dim), dtype=np.float32)
    else:
        embeddings = encoder.embed(grid.windows[loud_indices])

    cluster_result = cluster_embeddings(
        embeddings=embeddings,
        knn_neighbors=config.knn_neighbors,
        metric=config.knn_metric,
        resolution=config.leiden_resolution,
        n_iterations=config.leiden_n_iterations,
        seed=config.leiden_seed,
        min_cluster_size=config.min_cluster_size,
        stability_subsamples=config.stability_subsamples,
        stability_subsample_fraction=config.stability_subsample_fraction,
    )

    window_labels = np.full(grid.n_windows, -1, dtype=np.int64)
    window_stability = np.zeros(grid.n_windows, dtype=np.float32)
    if loud_indices.size:
        window_labels[loud_indices] = cluster_result.labels
        window_stability[loud_indices] = cluster_result.stability

    raw_events = runs_to_events(
        start_times=grid.start_times,
        window_sec=grid.window_sec,
        hop_sec=grid.hop_sec,
        labels=window_labels,
        stability=window_stability,
        rms_db=rms_db,
        min_event_duration=config.min_event_duration,
        max_event_duration=config.max_event_duration,
        max_event_gap=config.max_event_gap,
    )
    events_after_nms = temporal_nms(raw_events, iou_threshold=config.event_nms_iou)
    if config.refine_onset_offset:
        events_after_nms = refine_boundaries(
            audio=audio,
            sample_rate=sr,
            events=events_after_nms,
            activity_db=config.refinement_activity_db,
            padding_sec=config.refinement_padding,
            smoothing_sec=config.refinement_smoothing_sec,
        )

    sota_events, sota_clusters = _materialize_events_and_clusters(
        events=events_after_nms,
        embeddings=embeddings,
        loud_indices=loud_indices,
        grid_start_times=grid.start_times,
        window_sec=grid.window_sec,
        config=config,
        stability=cluster_result.stability,
    )

    zero_shot_results: list[ZeroShotResult] = []
    if config.enable_zero_shot and sota_clusters:
        try:
            zero_shot_results = _run_zero_shot(
                audio=audio,
                sample_rate=sr,
                events=sota_events,
                clusters=sota_clusters,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[sota] zero-shot captioning skipped: {exc}")
    label_by_cluster = {result.cluster_id: result for result in zero_shot_results}
    for cluster in sota_clusters:
        result = label_by_cluster.get(cluster.cluster_id)
        if result is not None:
            cluster.zero_shot_label = result.label
            cluster.zero_shot_score = result.score

    assign_cluster_folders(sota_clusters, prefix=config.cluster_folder_prefix)

    if config.enable_few_shot and config.few_shot_labels_path:
        _apply_few_shot(sota_events, sota_clusters, config)

    write_event_assets(
        audio=audio,
        sample_rate=sr,
        output_dir=output_dir_path,
        events=sota_events,
        clusters=sota_clusters,
        config=config,
    )
    events_csv, clusters_csv = write_csvs(
        output_dir=output_dir_path,
        events=sota_events,
        clusters=sota_clusters,
    )
    embeddings_npy: Path | None = None
    if config.export_embeddings:
        embeddings_npy = write_embeddings(
            output_dir=output_dir_path,
            embeddings=embeddings,
            start_times=grid.start_times[loud_indices] if loud_indices.size else np.array([]),
        )
    report_path = write_report(
        output_dir=output_dir_path,
        encoder_name=encoder_info.name,
        embedding_dim=encoder_info.embedding_dim,
        duration_sec=duration,
        sample_rate=sr,
        events=sota_events,
        clusters=sota_clusters,
    )
    index_path: Path | None = None
    if config.export_index_html:
        index_path = write_index_html(
            output_dir=output_dir_path,
            encoder_name=encoder_info.name,
            events=sota_events,
            clusters=sota_clusters,
        )
    write_run_metadata(
        output_dir=output_dir_path,
        metadata={
            "encoder": encoder_info.name,
            "embedding_dim": encoder_info.embedding_dim,
            "sample_rate": sr,
            "duration_sec": duration,
            "n_windows": grid.n_windows,
            "n_loud_windows": int(loud_indices.size),
            "n_events": len(sota_events),
            "n_clusters": len(sota_clusters),
            "config": {
                "window_sec": grid.window_sec,
                "hop_sec": grid.hop_sec,
                "silence_rms_db": config.silence_rms_db,
                "knn_neighbors": config.knn_neighbors,
                "leiden_resolution": config.leiden_resolution,
                "min_cluster_size": config.min_cluster_size,
                "stability_subsamples": config.stability_subsamples,
            },
        },
    )

    n_noise = sum(1 for e in sota_events if e.is_noise)
    return SOTAResult(
        input_path=str(input_path),
        output_dir=str(output_dir_path),
        encoder=encoder_info.name,
        embedding_dim=encoder_info.embedding_dim,
        duration_sec=duration,
        sample_rate=sr,
        n_windows=grid.n_windows,
        n_events=len(sota_events),
        n_clusters=len(sota_clusters),
        n_noise_events=n_noise,
        events_csv=str(events_csv),
        clusters_csv=str(clusters_csv),
        report_md=str(report_path),
        index_html=str(index_path) if index_path else None,
        embeddings_npy=str(embeddings_npy) if embeddings_npy else None,
    )


def _materialize_events_and_clusters(
    events: list[WindowEvent],
    embeddings: np.ndarray,
    loud_indices: np.ndarray,
    grid_start_times: np.ndarray,
    window_sec: float,
    config: SOTAConfig,
    stability: np.ndarray,
) -> tuple[list[SOTAEvent], list[SOTACluster]]:
    """Build SOTAEvent + SOTACluster objects, attach embeddings & representatives."""
    loud_to_idx_in_embeddings = {int(i): k for k, i in enumerate(loud_indices)}

    event_embeddings: list[np.ndarray] = []
    sota_events: list[SOTAEvent] = []
    for event_index, event in enumerate(events):
        emb_rows = []
        for window_idx in event.window_indices:
            mapped = loud_to_idx_in_embeddings.get(int(window_idx))
            if mapped is not None:
                emb_rows.append(embeddings[mapped])
        if emb_rows:
            mean_emb = np.mean(np.stack(emb_rows, axis=0), axis=0)
            mean_emb /= np.linalg.norm(mean_emb) + 1e-9
        else:
            mean_emb = np.zeros(embeddings.shape[1] if embeddings.size else 1, dtype=np.float32)

        event_embeddings.append(mean_emb.astype(np.float32))
        sota_events.append(
            SOTAEvent(
                event_id=f"event_{event_index:06d}",
                start_sec=float(event.start_sec),
                end_sec=float(event.end_sec),
                duration_sec=float(event.end_sec - event.start_sec),
                cluster_id=int(event.cluster_id),
                mean_window_score=float(event.mean_stability),
                n_windows=len(event.window_indices),
                rms_db=float(event.mean_rms_db),
                is_low_stability=bool(event.mean_stability < config.min_stability_for_keep),
                embedding=mean_emb.astype(np.float32),
            )
        )

    cluster_to_events: dict[int, list[int]] = defaultdict(list)
    for idx, sota in enumerate(sota_events):
        cluster_to_events[sota.cluster_id].append(idx)

    sota_clusters: list[SOTACluster] = []
    for cluster_id, idxs in sorted(cluster_to_events.items()):
        cluster_embs = np.stack([event_embeddings[i] for i in idxs], axis=0)
        centroid = cluster_embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        distances = 1.0 - cluster_embs @ centroid
        ranked = np.argsort(distances)
        for rank_position, local_idx in enumerate(ranked):
            event_idx = idxs[int(local_idx)]
            sota_events[event_idx].representative_rank = int(rank_position)
            sota_events[event_idx].centroid_distance = float(distances[int(local_idx)])
        representative_ids = [
            sota_events[idxs[int(local_idx)]].event_id
            for local_idx in ranked[: config.representatives_per_cluster]
        ]
        stab_values = [sota_events[i].mean_window_score for i in idxs]
        sota_clusters.append(
            SOTACluster(
                cluster_id=int(cluster_id),
                size=len(idxs),
                folder_name=f"{config.cluster_folder_prefix}_{cluster_id:03d}_size_{len(idxs):03d}",
                mean_stability=float(np.mean(stab_values)) if stab_values else 0.0,
                representative_event_ids=representative_ids,
            )
        )

    return sota_events, sota_clusters


def _run_zero_shot(
    audio: np.ndarray,
    sample_rate: int,
    events: list[SOTAEvent],
    clusters: list[SOTACluster],
    config: SOTAConfig,
) -> list[ZeroShotResult]:
    """Pick representative clips per cluster and caption them."""
    rep_clips_per_cluster: dict[int, list[np.ndarray]] = {}
    source_rates: dict[int, int] = {}
    for cluster in clusters:
        chosen_ids = cluster.representative_event_ids[: config.zero_shot_clips_per_cluster]
        clips: list[np.ndarray] = []
        for event_id in chosen_ids:
            event = next((e for e in events if e.event_id == event_id), None)
            if event is None:
                continue
            s = max(0, int(round(event.start_sec * sample_rate)))
            e = min(audio.shape[0], int(round(event.end_sec * sample_rate)))
            if e > s:
                clips.append(audio[s:e].astype(np.float32, copy=False))
        if clips:
            rep_clips_per_cluster[cluster.cluster_id] = clips
            source_rates[cluster.cluster_id] = sample_rate
    if not rep_clips_per_cluster:
        return []
    return caption_clusters(
        cluster_clips=rep_clips_per_cluster,
        cluster_source_rates=source_rates,
        prompts=list(config.zero_shot_prompts),
        device=config.encoder_device,
    )


def _apply_few_shot(
    events: list[SOTAEvent],
    clusters: list[SOTACluster],
    config: SOTAConfig,
) -> None:
    """Re-caption clusters from a few human labels, using prototype matching."""
    assert config.few_shot_labels_path is not None
    labels = load_labels_from_json(config.few_shot_labels_path)
    event_ids = [event.event_id for event in events]
    embeddings = np.stack(
        [event.embedding if event.embedding is not None else np.zeros(1, dtype=np.float32) for event in events],
        axis=0,
    )
    label_names, prototypes = compute_prototypes(embeddings, event_ids, labels)
    if not label_names:
        return
    cluster_centroids = []
    cluster_ids = []
    for cluster in clusters:
        cluster_events = [e for e in events if e.cluster_id == cluster.cluster_id]
        if not cluster_events:
            continue
        embs = np.stack([e.embedding for e in cluster_events if e.embedding is not None], axis=0)
        if embs.size == 0:
            continue
        centroid = embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        cluster_centroids.append(centroid)
        cluster_ids.append(cluster.cluster_id)

    if not cluster_centroids:
        return
    centroid_matrix = np.stack(cluster_centroids, axis=0)
    results = reassign(centroid_matrix, label_names, prototypes, config.few_shot_min_confidence)
    by_id = {cid: result for cid, result in zip(cluster_ids, results, strict=False)}
    for cluster in clusters:
        result = by_id.get(cluster.cluster_id)
        if result and result.new_label:
            cluster.zero_shot_label = result.new_label
            cluster.zero_shot_score = result.similarity
