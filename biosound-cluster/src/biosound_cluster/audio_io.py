"""Audio loading and writing utilities."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str | Path, sample_rate: int) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at the requested sample rate.

    WAV and FLAC are preferred for reproducible research workflows. Other formats
    such as AIFF and MP3 are supported when the local librosa/audioread backend can
    decode them.
    """
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Input path is not a file: {audio_path}")

    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    elif 0.0 < peak < 1e-8:
        audio = np.zeros_like(audio)
    return audio.astype(np.float32, copy=False), sr


def get_audio_duration(audio: np.ndarray, sr: int) -> float:
    """Return duration in seconds for a waveform."""
    if sr <= 0:
        raise ValueError("sr must be positive")
    return float(len(audio) / sr)


def safe_write_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    """Write a clipped float32 WAV file, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, -1.0, 1.0)
    sf.write(output_path, clean, sr, subtype="PCM_16")
