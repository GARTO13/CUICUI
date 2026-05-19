"""Spectrogram visualization helpers."""

from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_spectrogram_png(clip: np.ndarray, sr: int, output_path: str | Path) -> None:
    """Save a compact log-mel spectrogram PNG for an event clip."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if clip.size == 0:
        clip = np.zeros(int(sr * 0.05), dtype=np.float32)

    n_fft = min(2048, max(256, 2 ** int(np.floor(np.log2(max(256, clip.size))))))
    hop_length = min(512, max(64, n_fft // 4))
    mel = librosa.feature.melspectrogram(
        y=np.asarray(clip, dtype=np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=96,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max if np.max(mel) > 0 else 1.0)
    extent = [0, len(clip) / sr, 0, sr / 2]

    fig, ax = plt.subplots(figsize=(5.2, 2.8), dpi=120)
    ax.imshow(log_mel, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Log-mel spectrogram")
    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)
