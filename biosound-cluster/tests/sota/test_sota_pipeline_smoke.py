"""End-to-end smoke test for the SOTA pipeline using the mock encoder.

The mock encoder avoids torch/TF/Perch downloads. The Leiden step requires
``leidenalg`` + ``python-igraph``; the test is skipped if they aren't present.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from biosound_cluster.sota import SOTAConfig, process_audio_file_sota

LEIDEN_AVAILABLE = (
    importlib.util.find_spec("leidenalg") is not None
    and importlib.util.find_spec("igraph") is not None
)


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


def _add(buffer: np.ndarray, sr: int, start_sec: float, clip: np.ndarray) -> None:
    s = int(round(start_sec * sr))
    e = min(buffer.shape[0], s + clip.shape[0])
    buffer[s:e] += clip[: e - s]


@pytest.mark.skipif(not LEIDEN_AVAILABLE, reason="Leiden deps (igraph/leidenalg) not installed")
def test_sota_pipeline_smoke(tmp_path: Path) -> None:
    sr = 32_000
    duration = 40.0
    rng = np.random.default_rng(11)
    audio = (0.005 * rng.standard_normal(int(sr * duration))).astype(np.float32)

    for start in (1.0, 6.0, 11.0, 16.0, 21.0, 26.0, 31.0):
        _add(audio, sr, start, _chirp(sr, 2200, 3100, 0.6, 0.4))
    for start in (3.0, 8.0, 13.0, 18.0, 23.0, 28.0, 33.0):
        burst = _tone_burst(sr, 600, 0.7, 0.35) + 0.5 * _tone_burst(sr, 900, 0.7, 0.2)
        _add(audio, sr, start, burst)

    audio = np.clip(audio, -0.95, 0.95)
    wav_path = tmp_path / "sota_synth.wav"
    sf.write(wav_path, audio, sr)

    output_dir = tmp_path / "sota_out"
    result = process_audio_file_sota(
        wav_path,
        output_dir,
        SOTAConfig(
            sample_rate=sr,
            encoder="mock",
            window_sec=2.0,
            hop_sec=0.5,
            silence_rms_db=-60.0,
            knn_neighbors=8,
            leiden_resolution=1.0,
            min_cluster_size=2,
            stability_subsamples=2,
            min_event_duration=0.2,
            max_event_gap=0.5,
            enable_zero_shot=False,
            export_clips=True,
            export_spectrograms=False,
            export_embeddings=True,
            representatives_per_cluster=4,
        ),
    )

    assert result.n_events > 0, "expected some events from synthetic audio"
    assert result.n_clusters > 0, "expected at least one cluster"
    events = pd.read_csv(result.events_csv)
    clusters = pd.read_csv(result.clusters_csv)
    assert "cluster_id" in events.columns
    assert "mean_stability" in clusters.columns
    assert (output_dir / "report.md").exists()
    assert (output_dir / "index.html").exists()
    assert (output_dir / "embeddings.npz").exists()
    assert (output_dir / "run_metadata.json").exists()
    assert list(output_dir.glob("**/*.wav")), "expected exported wav clips"
