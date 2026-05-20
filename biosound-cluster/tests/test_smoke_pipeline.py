from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import pandas as pd

from biosound_cluster import BioSoundConfig, process_audio_file
from biosound_cluster.schemas import AudioEvent
from biosound_cluster.segmentation_refinement import refine_candidate_events


def _tone_burst(sr: int, frequency: float, duration: float, amplitude: float) -> np.ndarray:
    n = int(round(sr * duration))
    t = np.arange(n, dtype=np.float32) / sr
    envelope = np.hanning(n).astype(np.float32)
    return (amplitude * np.sin(2 * np.pi * frequency * t) * envelope).astype(np.float32)


def _chirp(sr: int, start_freq: float, end_freq: float, duration: float, amplitude: float) -> np.ndarray:
    n = int(round(sr * duration))
    t = np.arange(n, dtype=np.float32) / sr
    phase = 2 * np.pi * (start_freq * t + (end_freq - start_freq) * t**2 / (2 * duration))
    envelope = np.hanning(n).astype(np.float32)
    return (amplitude * np.sin(phase) * envelope).astype(np.float32)


def _add_event(audio: np.ndarray, sr: int, start_sec: float, event: np.ndarray) -> None:
    start = int(round(start_sec * sr))
    end = min(len(audio), start + len(event))
    audio[start:end] += event[: end - start]


def test_smoke_pipeline(tmp_path: Path) -> None:
    sr = 16_000
    duration = 16.0
    rng = np.random.default_rng(42)
    audio = (0.008 * rng.standard_normal(int(sr * duration))).astype(np.float32)

    for start in [0.8, 4.8, 8.8, 12.8]:
        _add_event(audio, sr, start, _chirp(sr, 2200, 3100, 0.35, 0.35))

    for start in [2.1, 6.1, 10.1, 14.1]:
        burst = _tone_burst(sr, 500, 0.45, 0.32) + 0.6 * _tone_burst(sr, 760, 0.45, 0.18)
        _add_event(audio, sr, start, burst.astype(np.float32))

    for start in [3.4, 7.4, 11.4, 15.2]:
        insect = _tone_burst(sr, 6000, 0.28, 0.24)
        _add_event(audio, sr, start, insect)

    audio = np.clip(audio, -0.95, 0.95)
    input_path = Path("tests/generated_synthetic.wav")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(input_path, audio, sr)

    output_dir = tmp_path / "synthetic_output"
    result = process_audio_file(
        input_path,
        output_dir,
        BioSoundConfig(
            sample_rate=sr,
            threshold_db=5.0,
            min_cluster_size=3,
            min_event_duration=0.12,
            max_event_duration=2.0,
            merge_gap=0.25,
            padding=0.05,
            generate_spectrograms=True,
            umap_neighbors=5,
            umap_components=3,
        ),
    )

    assert output_dir.exists()
    assert Path(result.events_csv).exists()
    assert Path(result.clusters_csv).exists()
    assert Path(result.report_md).exists()
    assert Path(result.index_html).exists()
    assert result.n_events > 0
    assert any(path.is_dir() for path in output_dir.iterdir())
    assert list(output_dir.glob("**/*.wav"))
    events = pd.read_csv(output_dir / "events.csv")
    expected_columns = {
        "clusterability_score",
        "embedding_stability_score",
        "local_snr_score",
        "spectral_structure_score",
        "acoustic_prefamily",
        "representative_score",
    }
    assert expected_columns.issubset(events.columns)


def test_polyphony_outputs_components_or_mixed(tmp_path: Path) -> None:
    sr = 16_000
    duration = 7.0
    rng = np.random.default_rng(7)
    audio = (0.004 * rng.standard_normal(int(sr * duration))).astype(np.float32)

    croak = _tone_burst(sr, 500, 0.65, 0.35) + 0.5 * _tone_burst(sr, 760, 0.65, 0.18)
    chirp = _chirp(sr, 3000, 4100, 0.55, 0.28)
    _add_event(audio, sr, 0.8, croak.astype(np.float32))
    _add_event(audio, sr, 2.2, chirp)
    _add_event(audio, sr, 4.0, croak.astype(np.float32))
    _add_event(audio, sr, 4.05, chirp)

    input_path = tmp_path / "polyphony.wav"
    sf.write(input_path, np.clip(audio, -0.95, 0.95), sr)

    output_dir = tmp_path / "polyphony_output"
    process_audio_file(
        input_path,
        output_dir,
        BioSoundConfig(
            sample_rate=sr,
            threshold_db=4.0,
            min_cluster_size=2,
            min_event_duration=0.12,
            max_event_duration=2.0,
            merge_gap=0.2,
            padding=0.04,
            umap_neighbors=3,
            umap_components=2,
            component_snr_db=6.0,
            component_min_area=8,
            component_min_duration=0.05,
            component_min_bandwidth_hz=80.0,
            min_component_energy_ratio=0.03,
            min_purity_for_clustering=0.45,
        ),
    )

    events = pd.read_csv(output_dir / "events.csv")
    expected_columns = {
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
    }
    assert expected_columns.issubset(events.columns)
    mixed_rows = events[events["source_type"] == "mixed"]
    component_rows = events[events["source_type"] == "component"]
    assert not mixed_rows["cluster_id"].notna().any()
    if not component_rows.empty:
        assert component_rows["parent_event_id"].notna().all()
        assert component_rows["context_clip_path"].notna().all()
    assert not component_rows.empty or list(output_dir.glob("mixed_overlapping_size_*"))


def test_noise_filter_routes_broadband_events_out_of_clusters(tmp_path: Path) -> None:
    sr = 16_000
    duration = 9.0
    rng = np.random.default_rng(123)
    audio = (0.003 * rng.standard_normal(int(sr * duration))).astype(np.float32)

    for start in [0.8, 3.2, 6.4]:
        _add_event(audio, sr, start, _chirp(sr, 2600, 3300, 0.45, 0.30))

    for start in [2.0, 5.0]:
        noise = (0.23 * rng.standard_normal(int(sr * 0.45))).astype(np.float32)
        noise *= np.hanning(noise.size).astype(np.float32)
        _add_event(audio, sr, start, noise)

    input_path = tmp_path / "noisy_events.wav"
    sf.write(input_path, np.clip(audio, -0.95, 0.95), sr)

    output_dir = tmp_path / "noise_filtered_output"
    process_audio_file(
        input_path,
        output_dir,
        BioSoundConfig(
            sample_rate=sr,
            threshold_db=4.0,
            min_cluster_size=2,
            min_event_duration=0.10,
            max_event_duration=1.5,
            merge_gap=0.15,
            padding=0.03,
            umap_neighbors=3,
            umap_components=2,
            enable_polyphony_handling=False,
            noise_mode="conservative",
            min_quality_for_clustering=0.45,
            generate_spectrograms=False,
        ),
    )

    events = pd.read_csv(output_dir / "events.csv")
    expected_columns = {
        "is_low_confidence_noise",
        "snr_db",
        "noise_floor_db",
        "spectral_flatness",
        "tonality_score",
        "bandwidth_hz",
        "quality_score",
    }
    assert expected_columns.issubset(events.columns)
    low_noise = events[events["source_type"] == "low_confidence_noise"]
    assert not low_noise.empty
    assert not low_noise["cluster_id"].notna().any()
    assert list(output_dir.glob("low_confidence_noise_size_*"))


def test_short_events_are_exported_for_secondary_review(tmp_path: Path) -> None:
    sr = 16_000
    audio = (0.002 * np.random.default_rng(321).standard_normal(sr * 4)).astype(np.float32)
    _add_event(audio, sr, 0.7, _tone_burst(sr, 2500, 0.08, 0.45))
    _add_event(audio, sr, 2.0, _tone_burst(sr, 1200, 0.45, 0.35))

    input_path = tmp_path / "short_review.wav"
    sf.write(input_path, np.clip(audio, -0.95, 0.95), sr)

    output_dir = tmp_path / "short_review_output"
    process_audio_file(
        input_path,
        output_dir,
        BioSoundConfig(
            sample_rate=sr,
            threshold_db=3.0,
            min_cluster_size=2,
            min_event_duration=0.04,
            max_event_duration=1.0,
            merge_gap=0.05,
            padding=0.0,
            generate_spectrograms=False,
            enable_polyphony_handling=False,
            enable_noise_filtering=False,
            enable_eventness_filtering=False,
            enable_candidate_selection=False,
            enable_short_event_review=True,
            min_review_event_duration=0.20,
            umap_neighbors=3,
            umap_components=2,
        ),
    )

    events = pd.read_csv(output_dir / "events.csv")
    assert "is_short_review_event" in events.columns
    short_rows = events[events["source_type"] == "short_review"]
    assert not short_rows.empty
    assert not short_rows["cluster_id"].notna().any()
    assert list(output_dir.glob("short_events_review_size_*"))


def test_metadata_only_export_skips_media_files(tmp_path: Path) -> None:
    sr = 16_000
    audio = np.zeros(sr * 3, dtype=np.float32)
    _add_event(audio, sr, 0.8, _chirp(sr, 1800, 2600, 0.4, 0.3))

    input_path = tmp_path / "metadata_only.wav"
    sf.write(input_path, audio, sr)

    output_dir = tmp_path / "metadata_only_output"
    process_audio_file(
        input_path,
        output_dir,
        BioSoundConfig(
            sample_rate=sr,
            threshold_db=3.0,
            min_cluster_size=2,
            min_event_duration=0.1,
            max_event_duration=1.0,
            generate_spectrograms=False,
            export_clips=False,
            enable_polyphony_handling=False,
            umap_neighbors=3,
            umap_components=2,
        ),
    )

    assert (output_dir / "events.csv").exists()
    assert (output_dir / "run_metadata.json").exists()
    assert not list(output_dir.glob("**/*.wav"))
    assert not list(output_dir.glob("**/*.png"))


def test_recording_and_clip_metadata_json(tmp_path: Path) -> None:
    sr = 16_000
    audio = np.zeros(sr * 3, dtype=np.float32)
    _add_event(audio, sr, 1.0, _tone_burst(sr, 1800, 0.35, 0.35))

    input_path = tmp_path / "metadata_rich.wav"
    sf.write(input_path, audio, sr)

    output_dir = tmp_path / "metadata_rich_output"
    process_audio_file(
        input_path,
        output_dir,
        BioSoundConfig(
            sample_rate=sr,
            threshold_db=3.0,
            min_cluster_size=2,
            min_event_duration=0.1,
            max_event_duration=1.0,
            generate_spectrograms=False,
            enable_polyphony_handling=False,
            enable_noise_filtering=False,
            enable_eventness_filtering=False,
            enable_candidate_selection=False,
            enable_clusterability_filtering=False,
            sensor_id="sensor_A",
            sensor_latitude=4.9372,
            sensor_longitude=-52.3260,
            sensor_elevation_m=18.5,
            environment_type="tropical_forest",
            recording_start_time="2026-05-20T06:30:00+02:00",
            recording_timezone="Europe/Paris",
            umap_neighbors=3,
            umap_components=2,
        ),
    )

    run_metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert run_metadata["recording_metadata"]["sensor"]["latitude"] == 4.9372
    assert run_metadata["recording_metadata"]["environment_type"] == "tropical_forest"

    event_metadata_path = output_dir / "event_metadata.json"
    assert event_metadata_path.exists()
    event_metadata = json.loads(event_metadata_path.read_text())
    assert event_metadata["recording"]["recording_start_time"] == "2026-05-20T06:30:00+02:00"
    assert event_metadata["events"]
    first_event = event_metadata["events"][0]
    assert first_event["recording"]["sensor"]["sensor_id"] == "sensor_A"
    assert first_event["clip_timing"]["start_sec"] >= 0
    assert first_event["clip_timing"]["clip_start_time"] is not None

    sidecars = [
        path
        for path in output_dir.glob("**/*.json")
        if path.name not in {"run_metadata.json", "event_metadata.json"}
    ]
    assert sidecars
    sidecar = json.loads(sidecars[0].read_text())
    assert sidecar["recording"]["environment_type"] == "tropical_forest"
    assert "clip_timing" in sidecar


def test_segmentation_refinement_tightens_broad_event() -> None:
    sr = 16_000
    audio = np.zeros(sr * 3, dtype=np.float32)
    _add_event(audio, sr, 1.0, _tone_burst(sr, 2200, 0.35, 0.4))
    event = AudioEvent(
        event_id="event_000000",
        start_sec=0.45,
        end_sec=1.85,
        duration_sec=1.40,
    )

    refined = refine_candidate_events(
        audio,
        sr,
        [event],
        BioSoundConfig(
            sample_rate=sr,
            min_event_duration=0.1,
            refinement_activity_db=2.0,
            refinement_peak_drop_db=24.0,
            refinement_padding=0.02,
            refinement_min_trim_sec=0.01,
        ),
    )

    assert len(refined) == 1
    assert refined[0].duration_sec < event.duration_sec
    assert refined[0].start_sec > 0.7
    assert refined[0].end_sec < 1.6
