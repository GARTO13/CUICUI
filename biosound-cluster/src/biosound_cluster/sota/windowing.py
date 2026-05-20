"""Sliding-window extraction and silence gating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class WindowGrid:
    """Result of slicing a long audio buffer into overlapping windows."""

    windows: np.ndarray  # (n_windows, n_samples)
    start_times: np.ndarray  # (n_windows,) — seconds
    sample_rate: int
    window_sec: float
    hop_sec: float

    @property
    def end_times(self) -> np.ndarray:
        return self.start_times + self.window_sec

    @property
    def n_windows(self) -> int:
        return int(self.windows.shape[0])


def extract_windows(
    audio: np.ndarray,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
) -> WindowGrid:
    """Slice ``audio`` into overlapping windows.

    The last window is zero-padded if the audio doesn't divide evenly.
    A pure-zero audio shorter than one window still produces one (padded)
    window so the caller can keep a uniform shape.
    """
    if audio.ndim != 1:
        raise ValueError("audio must be mono (1-D)")
    if window_sec <= 0 or hop_sec <= 0:
        raise ValueError("window_sec and hop_sec must be positive")

    n_window = int(round(window_sec * sample_rate))
    n_hop = int(round(hop_sec * sample_rate))
    n_window = max(1, n_window)
    n_hop = max(1, n_hop)

    n_total = int(audio.shape[0])
    if n_total <= n_window:
        padded = np.zeros(n_window, dtype=np.float32)
        padded[: n_total] = audio.astype(np.float32, copy=False)
        return WindowGrid(
            windows=padded[None, :],
            start_times=np.array([0.0], dtype=np.float64),
            sample_rate=int(sample_rate),
            window_sec=float(window_sec),
            hop_sec=float(hop_sec),
        )

    n_windows = 1 + (n_total - n_window) // n_hop
    last_end = (n_windows - 1) * n_hop + n_window
    pad = 0
    if last_end + n_hop <= n_total + n_window:
        n_windows += 1
        pad = max(0, (n_windows - 1) * n_hop + n_window - n_total)
    if pad > 0:
        audio = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])

    starts = np.arange(n_windows, dtype=np.int64) * n_hop
    windows = np.stack(
        [audio[s : s + n_window] for s in starts],
        axis=0,
    ).astype(np.float32, copy=False)
    start_times = starts.astype(np.float64) / float(sample_rate)
    return WindowGrid(
        windows=windows,
        start_times=start_times,
        sample_rate=int(sample_rate),
        window_sec=float(window_sec),
        hop_sec=float(hop_sec),
    )


def window_rms_db(windows: np.ndarray) -> np.ndarray:
    """Per-window RMS in dB (ref=1.0)."""
    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1) + 1e-20)
    return 20.0 * np.log10(rms + 1e-20)


def silence_mask(windows: np.ndarray, threshold_db: float) -> np.ndarray:
    """Boolean mask: True for windows louder than ``threshold_db``."""
    return window_rms_db(windows) > threshold_db
