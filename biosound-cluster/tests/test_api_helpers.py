from __future__ import annotations

import math
from pathlib import Path

from biosound_cluster.api import _build_config, _is_relative_to, _json_clean, _looks_like_wav, _sanitize_filename


def test_api_filename_and_wav_guards() -> None:
    assert _sanitize_filename("../Bruits de la foret Guyanaise.wav") == "Bruits_de_la_foret_Guyanaise.wav"
    assert _sanitize_filename("") == "input.wav"
    assert _looks_like_wav(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    assert not _looks_like_wav(b"not a wav")


def test_api_path_and_json_helpers(tmp_path: Path) -> None:
    root = tmp_path / "job" / "run"
    inside = root / "cluster_000_size_001" / "event.wav"
    outside = tmp_path / "other.wav"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"data")
    outside.write_bytes(b"data")

    assert _is_relative_to(inside.resolve(), root.resolve())
    assert not _is_relative_to(outside.resolve(), root.resolve())

    cleaned = _json_clean({"value": float("nan"), "items": [1, None, math.nan]})
    assert cleaned == {"value": None, "items": [1, None, None]}


def test_api_build_config_keeps_upload_metadata_out_of_config() -> None:
    config, metadata = _build_config(
        original_filename="recording.wav",
        saved_size_bytes=123,
        sensor_id="sensor",
        sensor_latitude="",
        sensor_longitude=None,
        sensor_elevation_m=None,
        environment_type="forest",
        recording_start_time=None,
        recording_timezone=None,
        sample_rate="32000",
        min_cluster_size="5",
        max_events="5",
        generate_spectrograms="true",
        enable_auto_profile="true",
        enable_polyphony_handling="true",
        enable_clusterability_filtering="true",
    )
    assert config.sensor_id == "sensor"
    assert config.sensor_latitude is None
    assert config.max_events == 5
    assert config.min_cluster_size == 5
    assert config.enable_auto_profile is True
    assert metadata["saved_size_bytes"] == 123
    assert metadata["original_filename"] == "recording.wav"
