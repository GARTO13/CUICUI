"""Simple acoustic pre-family routing before unsupervised clustering."""

from __future__ import annotations

import librosa
import numpy as np

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.embeddings import extract_event_clip
from biosound_cluster.schemas import AudioEvent


PREFAMILIES = {
    "tonal_whistle",
    "harmonic_call",
    "pulse_train",
    "broadband_click",
    "noisy_burst",
    "insect_trill",
    "low_frequency_call",
    "unknown",
}


def assign_acoustic_prefamilies(
    audio: np.ndarray,
    sr: int,
    events: list[AudioEvent],
    config: BioSoundConfig,
) -> list[AudioEvent]:
    """Attach a coarse acoustic pre-family to each event without species labels."""
    if not events or not config.enable_acoustic_prefamilies:
        for event in events:
            event.acoustic_prefamily = event.acoustic_prefamily or "unknown"
        return events

    for event in events:
        clip = extract_event_clip(audio, sr, event)
        event.acoustic_prefamily = classify_prefamily(clip, sr, event, config)
    return events


def classify_prefamily(
    clip: np.ndarray,
    sr: int,
    event: AudioEvent,
    config: BioSoundConfig,
) -> str:
    """Classify an event into a broad acoustic morphology family."""
    duration = float(event.duration_sec)
    centroid = _value(event.spectral_centroid, 0.0)
    bandwidth = _value(event.bandwidth_hz, 0.0)
    flatness = _value(event.spectral_flatness, 0.5)
    tonality = _value(event.tonality_score, 0.3)
    zcr = _zero_crossing_rate(clip, config)
    modulation_rate = _modulation_rate(clip, sr, config)

    if duration < 0.12 and flatness > 0.45 and bandwidth > 1500:
        return "broadband_click"
    if centroid < 900 and duration >= 0.18 and tonality >= 0.12:
        return "low_frequency_call"
    if duration >= 0.45 and centroid >= 3000 and tonality >= 0.18 and modulation_rate >= 8.0:
        return "insect_trill"
    if flatness >= 0.62 and tonality < 0.18:
        return "noisy_burst"
    if tonality >= 0.48 and bandwidth < max(900.0, centroid * 0.55):
        return "tonal_whistle"
    if tonality >= 0.22 and bandwidth < max(1800.0, centroid * 1.25):
        return "harmonic_call"
    if modulation_rate >= 5.0 and zcr > 0.08:
        return "pulse_train"
    return "unknown"


def _zero_crossing_rate(clip: np.ndarray, config: BioSoundConfig) -> float:
    if clip.size == 0:
        return 0.0
    frame_length = min(config.frame_length, max(128, clip.size))
    hop_length = min(config.hop_length, max(64, frame_length // 4))
    zcr = librosa.feature.zero_crossing_rate(clip, frame_length=frame_length, hop_length=hop_length)[0]
    return float(np.nanmedian(np.nan_to_num(zcr, nan=0.0))) if zcr.size else 0.0


def _modulation_rate(clip: np.ndarray, sr: int, config: BioSoundConfig) -> float:
    if clip.size == 0:
        return 0.0
    frame_length = min(config.frame_length, max(128, clip.size))
    hop_length = min(config.hop_length, max(64, frame_length // 4))
    rms = librosa.feature.rms(y=clip, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size < 4:
        return 0.0
    envelope = rms - np.median(rms)
    peaks = np.flatnonzero((envelope[1:-1] > envelope[:-2]) & (envelope[1:-1] >= envelope[2:])) + 1
    duration = max(clip.size / sr, 1e-6)
    return float(len(peaks) / duration)


def _value(value: float | None, default: float) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(value)
