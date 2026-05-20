from __future__ import annotations

import numpy as np

from biosound_cluster.sota.windowing import extract_windows, silence_mask, window_rms_db


def test_extract_windows_basic() -> None:
    sr = 1000
    audio = np.arange(2500, dtype=np.float32) / 2500.0
    grid = extract_windows(audio, sr, window_sec=1.0, hop_sec=0.5)
    assert grid.windows.shape[1] == 1000
    assert grid.n_windows >= 3
    assert grid.start_times[0] == 0.0
    assert grid.start_times[1] == 0.5


def test_extract_windows_short_audio_is_padded() -> None:
    sr = 1000
    audio = np.zeros(300, dtype=np.float32)
    grid = extract_windows(audio, sr, window_sec=1.0, hop_sec=0.5)
    assert grid.n_windows == 1
    assert grid.windows.shape[1] == 1000


def test_silence_mask_marks_quiet_windows() -> None:
    sr = 1000
    loud = 0.5 * np.sin(2 * np.pi * 50 * np.arange(sr) / sr).astype(np.float32)
    quiet = 1e-5 * np.ones(sr, dtype=np.float32)
    windows = np.stack([loud, quiet])
    mask = silence_mask(windows, threshold_db=-40.0)
    assert mask[0] is np.bool_(True) or bool(mask[0])
    assert not bool(mask[1])


def test_window_rms_db_monotonic_with_amplitude() -> None:
    sr = 1000
    a = 0.1 * np.ones(sr, dtype=np.float32)
    b = 0.5 * np.ones(sr, dtype=np.float32)
    db = window_rms_db(np.stack([a, b]))
    assert db[1] > db[0]
