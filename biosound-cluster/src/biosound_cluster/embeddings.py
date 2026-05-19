"""Handcrafted acoustic embeddings for unsupervised similarity grouping."""

from __future__ import annotations

import librosa
import numpy as np

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.schemas import AudioEvent


def extract_event_clip(audio: np.ndarray, sr: int, event: AudioEvent) -> np.ndarray:
    """Extract a waveform clip for an event."""
    if event.separated_audio is not None:
        return np.asarray(event.separated_audio, dtype=np.float32)
    start = max(0, int(round(event.start_sec * sr)))
    end = min(len(audio), int(round(event.end_sec * sr)))
    return np.asarray(audio[start:end], dtype=np.float32)


def _safe_stats(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0, 0.0]
    clean = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return [float(np.mean(clean)), float(np.std(clean))]


def _embedding_dim(n_mels: int, config: BioSoundConfig) -> int:
    mel_blocks = 6 if config.embedding_include_delta else 4
    return (n_mels * mel_blocks) + 14


def compute_logmel_embedding(clip: np.ndarray, sr: int, config: BioSoundConfig, n_mels: int = 96) -> np.ndarray:
    """Compute a fixed-length log-mel and global-feature embedding for one clip."""
    if clip.size == 0:
        return np.zeros(_embedding_dim(n_mels, config), dtype=np.float32)

    n_fft = min(config.frame_length, max(256, 2 ** int(np.floor(np.log2(max(256, clip.size))))))
    hop_length = min(config.hop_length, max(64, n_fft // 4))
    mel = librosa.feature.melspectrogram(
        y=clip,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max if np.max(mel) > 0 else 1.0)
    mel_parts = [
        np.mean(log_mel, axis=1),
        np.std(log_mel, axis=1),
        np.max(log_mel, axis=1),
        np.percentile(log_mel, 90, axis=1),
    ]
    if config.embedding_include_delta:
        if log_mel.shape[1] >= 3:
            width = min(9, log_mel.shape[1] if log_mel.shape[1] % 2 == 1 else log_mel.shape[1] - 1)
            delta = librosa.feature.delta(log_mel, width=max(3, width), mode="nearest")
        else:
            delta = np.zeros_like(log_mel)
        mel_parts.extend([np.mean(delta, axis=1), np.std(delta, axis=1)])
    mel_features = np.concatenate(mel_parts)

    rms = librosa.feature.rms(y=clip, frame_length=n_fft, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=clip, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=clip, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=clip, n_fft=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=clip, frame_length=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=clip, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    global_features = [
        float(np.mean(rms)) if rms.size else 0.0,
        float(np.std(rms)) if rms.size else 0.0,
        float(np.max(rms)) if rms.size else 0.0,
        *_safe_stats(centroid),
        *_safe_stats(bandwidth),
        *_safe_stats(flatness),
        *_safe_stats(zcr),
        *_safe_stats(rolloff),
    ]
    vector = np.concatenate([mel_features, np.asarray(global_features, dtype=np.float64)])
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_embeddings(
    audio: np.ndarray,
    sr: int,
    events: list[AudioEvent],
    config: BioSoundConfig,
) -> np.ndarray:
    """Extract one handcrafted acoustic embedding per event."""
    if not events:
        return np.empty((0, _embedding_dim(96, config)), dtype=np.float32)
    vectors = [
        compute_logmel_embedding(extract_event_clip(audio, sr, event), sr, config)
        for event in events
    ]
    return np.vstack(vectors).astype(np.float32)
