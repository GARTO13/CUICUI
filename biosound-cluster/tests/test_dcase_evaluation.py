from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.evaluation.dcase2024 import evaluate_dcase2024
from biosound_cluster.evaluation.tuning import tune_dcase2024


def _burst(sr: int, frequency: float, duration: float, amplitude: float = 0.35) -> np.ndarray:
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr
    return (amplitude * np.sin(2 * np.pi * frequency * t) * np.hanning(n)).astype(np.float32)


def _add(audio: np.ndarray, sr: int, start: float, event: np.ndarray) -> None:
    i0 = int(round(start * sr))
    i1 = min(len(audio), i0 + len(event))
    audio[i0:i1] += event[: i1 - i0]


def test_evaluate_dcase2024_synthetic_dataset(tmp_path: Path) -> None:
    sr = 16_000
    dataset_dir = tmp_path / "Development_Set" / "validation" / "BV"
    dataset_dir.mkdir(parents=True)
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()

    for idx in range(2):
        audio = np.zeros(sr * 5, dtype=np.float32)
        audio += (0.003 * np.random.default_rng(idx).standard_normal(audio.size)).astype(np.float32)
        _add(audio, sr, 0.8, _burst(sr, 500, 0.45))
        _add(audio, sr, 2.4, _burst(sr, 3500, 0.35))
        file_id = f"BV_validation_{idx:02d}"
        sf.write(dataset_dir / f"{file_id}.wav", np.clip(audio, -0.95, 0.95), sr)
        pd.DataFrame(
            [
                {"Starttime": 0.75, "Endtime": 1.30, "Q": "POS"},
                {"Starttime": 2.35, "Endtime": 2.85, "Q": "POS"},
                {"Starttime": 3.50, "Endtime": 3.90, "Q": "NEG"},
            ]
        ).to_csv(annotations_dir / f"{file_id}.csv", index=False)

    output_dir = tmp_path / "eval"
    summary = evaluate_dcase2024(
        dataset_dir=tmp_path / "Development_Set",
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        split="validation",
        subset="BV",
        max_files=2,
        config=BioSoundConfig(
            sample_rate=sr,
            threshold_db=4.0,
            min_event_duration=0.1,
            max_event_duration=1.5,
            merge_gap=0.15,
            padding=0.04,
            min_cluster_size=2,
            generate_spectrograms=False,
            enable_polyphony_handling=False,
            umap_neighbors=3,
            umap_components=2,
        ),
        iou_threshold=0.2,
        overlap_threshold=0.2,
        force=True,
    )

    assert (output_dir / "evaluation_summary.json").exists()
    assert (output_dir / "evaluation_report.md").exists()
    assert (output_dir / "per_file_metrics.csv").exists()
    assert (output_dir / "matched_predictions.csv").exists()
    loaded = json.loads((output_dir / "evaluation_summary.json").read_text())
    assert 0 <= loaded["final_score_100"] <= 100
    assert "f1" in loaded["detection"]
    assert "weighted_cluster_purity" in loaded["clustering"]
    assert 0 <= summary["final_score_100"] <= 100


def test_tune_dcase2024_synthetic_dataset(tmp_path: Path) -> None:
    sr = 16_000
    dataset_dir = tmp_path / "Development_Set" / "validation" / "BV"
    dataset_dir.mkdir(parents=True)
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()

    audio = np.zeros(sr * 4, dtype=np.float32)
    audio += (0.003 * np.random.default_rng(9).standard_normal(audio.size)).astype(np.float32)
    _add(audio, sr, 0.8, _burst(sr, 500, 0.4))
    _add(audio, sr, 2.3, _burst(sr, 3500, 0.35))
    file_id = "BV_validation_tune"
    sf.write(dataset_dir / f"{file_id}.wav", np.clip(audio, -0.95, 0.95), sr)
    pd.DataFrame(
        [
            {"Starttime": 0.75, "Endtime": 1.25, "Q": "POS"},
            {"Starttime": 2.25, "Endtime": 2.75, "Q": "POS"},
        ]
    ).to_csv(annotations_dir / f"{file_id}.csv", index=False)

    output_dir = tmp_path / "tune"
    summary = tune_dcase2024(
        dataset_dir=tmp_path / "Development_Set",
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        split="validation",
        subset="BV",
        max_files=1,
        base_config=BioSoundConfig(
            sample_rate=sr,
            threshold_db=4.0,
            min_event_duration=0.1,
            max_event_duration=1.5,
            merge_gap=0.15,
            padding=0.04,
            min_cluster_size=2,
            generate_spectrograms=False,
            export_clips=False,
            enable_polyphony_handling=False,
            umap_neighbors=3,
            umap_components=2,
        ),
        search_size="quick",
        max_trials=2,
        iou_threshold=0.2,
        overlap_threshold=0.2,
        force=True,
    )

    assert (output_dir / "tuning_results.csv").exists()
    assert (output_dir / "tuning_summary.json").exists()
    assert (output_dir / "best_config.json").exists()
    assert 0 <= float(summary["best_score_100"]) <= 100
    assert summary["best_config"]
