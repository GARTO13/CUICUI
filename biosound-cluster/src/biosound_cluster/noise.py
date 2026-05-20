"""Noise confidence scoring and routing for candidate acoustic events."""

from __future__ import annotations

from dataclasses import dataclass, replace

import librosa
import numpy as np

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.embeddings import extract_event_clip
from biosound_cluster.logging_utils import get_logger
from biosound_cluster.schemas import AudioEvent

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class NoiseFeatures:
    """Compact acoustic quality features for one event."""

    snr_db: float
    noise_floor_db: float
    spectral_flatness: float
    tonality_score: float
    bandwidth_hz: float
    peak_band_snr_db: float
    quality_score: float


def analyze_and_route_noise(
    audio: np.ndarray,
    sr: int,
    events: list[AudioEvent],
    config: BioSoundConfig,
) -> tuple[list[AudioEvent], list[AudioEvent]]:
    """
    Split events into clusterable events and low-confidence noise review events.

    This does not classify species or permanently discard audio. It only prevents
    broadband / low-SNR / weakly structured events from polluting acoustic clusters.
    """
    if not events:
        return [], []

    noise_floor_db = estimate_recording_noise_floor_db(audio, config)
    band_noise_floor_db = estimate_per_band_noise_floor_db(audio, sr, config)
    clusterable: list[AudioEvent] = []
    low_confidence: list[AudioEvent] = []

    for event in events:
        try:
            features = compute_noise_features(audio, sr, event, config, noise_floor_db, band_noise_floor_db)
            _attach_noise_features(event, features)
            if is_low_confidence_noise(features, config):
                low_confidence.append(_mark_low_confidence_noise(event, features))
            else:
                clusterable.append(event)
        except Exception as exc:  # pragma: no cover - defensive path for unusual audio
            LOGGER.warning("Noise scoring failed for %s; keeping event clusterable: %s", event.event_id, exc)
            clusterable.append(event)

    return clusterable, low_confidence


def estimate_recording_noise_floor_db(audio: np.ndarray, config: BioSoundConfig) -> float:
    """Estimate a robust broadband recording noise floor from frame RMS."""
    if audio.size == 0:
        return -120.0
    frame_length = min(config.frame_length, max(256, audio.size))
    hop_length = min(config.hop_length, max(64, frame_length // 4))
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0)
    return float(np.percentile(rms_db, 20)) if rms_db.size else -120.0


def estimate_per_band_noise_floor_db(
    audio: np.ndarray,
    sr: int,
    config: BioSoundConfig,
    n_mels: int = 32,
) -> np.ndarray:
    """Estimate a robust per-mel-band noise floor (dB) from the whole recording."""
    if audio.size == 0:
        return np.full(n_mels, -120.0, dtype=np.float32)
    n_fft = min(config.frame_length, max(256, audio.size))
    hop_length = min(config.hop_length, max(64, n_fft // 4))
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    if mel.size == 0:
        return np.full(n_mels, -120.0, dtype=np.float32)
    mel_db = librosa.power_to_db(np.maximum(mel, 1e-20), ref=1.0)
    return np.percentile(mel_db, 20, axis=1).astype(np.float32)


def compute_peak_band_snr_db(
    clip: np.ndarray,
    sr: int,
    band_noise_floor_db: np.ndarray,
    config: BioSoundConfig,
) -> float:
    """Concentration of the per-band SNR distribution: peak band SNR minus median band SNR.

    A narrowband call has one or two loud bands and the rest near the noise floor — large gap.
    Broadband noise lights every band roughly equally — small gap. So this distinguishes
    "concentrated biological event" from "broadband disturbance" even when both have high
    overall SNR.
    """
    if clip.size == 0 or band_noise_floor_db.size == 0:
        return 0.0
    n_mels = int(band_noise_floor_db.size)
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
    if mel.size == 0:
        return 0.0
    mel_db = librosa.power_to_db(np.maximum(mel, 1e-20), ref=1.0)
    band_event_levels = np.percentile(mel_db, 75, axis=1)
    band_snrs = band_event_levels - band_noise_floor_db
    if band_snrs.size == 0:
        return 0.0
    return float(np.max(band_snrs) - np.median(band_snrs))


def compute_noise_features(
    audio: np.ndarray,
    sr: int,
    event: AudioEvent,
    config: BioSoundConfig,
    recording_noise_floor_db: float,
    band_noise_floor_db: np.ndarray | None = None,
) -> NoiseFeatures:
    """Compute SNR, flatness, tonality, bandwidth, peak-band SNR, and an aggregate quality score."""
    clip = extract_event_clip(audio, sr, event)
    if clip.size == 0:
        return NoiseFeatures(0.0, recording_noise_floor_db, 1.0, 0.0, 0.0, 0.0, 0.0)

    n_fft = min(config.frame_length, max(256, 2 ** int(np.floor(np.log2(max(256, clip.size))))))
    hop_length = min(config.hop_length, max(64, n_fft // 4))
    rms = librosa.feature.rms(y=clip, frame_length=n_fft, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0)
    event_level_db = float(np.percentile(rms_db, 75)) if rms_db.size else -120.0
    snr_db = max(0.0, event_level_db - recording_noise_floor_db)

    stft_mag = np.abs(librosa.stft(clip, n_fft=n_fft, hop_length=hop_length))
    if stft_mag.size == 0:
        return NoiseFeatures(snr_db, recording_noise_floor_db, 1.0, 0.0, 0.0, 0.0, 0.0)

    flatness_values = librosa.feature.spectral_flatness(S=stft_mag)[0]
    flatness = float(np.nanmedian(np.nan_to_num(flatness_values, nan=1.0))) if flatness_values.size else 1.0

    centroid = librosa.feature.spectral_centroid(S=stft_mag, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=stft_mag, sr=sr, centroid=centroid)[0]
    bandwidth_hz = float(np.nanmedian(np.nan_to_num(bandwidth, nan=0.0))) if bandwidth.size else 0.0

    tonality = _spectral_tonality(stft_mag)
    if band_noise_floor_db is None:
        band_noise_floor_db = estimate_per_band_noise_floor_db(audio, sr, config)
    peak_band_snr_db = compute_peak_band_snr_db(clip, sr, band_noise_floor_db, config)

    quality = _quality_score(snr_db, flatness, tonality, bandwidth_hz, peak_band_snr_db, sr, config)
    return NoiseFeatures(
        snr_db=float(snr_db),
        noise_floor_db=float(recording_noise_floor_db),
        spectral_flatness=float(np.clip(flatness, 0.0, 1.0)),
        tonality_score=float(np.clip(tonality, 0.0, 1.0)),
        bandwidth_hz=bandwidth_hz,
        peak_band_snr_db=float(peak_band_snr_db),
        quality_score=float(np.clip(quality, 0.0, 1.0)),
    )


def is_low_confidence_noise(features: NoiseFeatures, config: BioSoundConfig) -> bool:
    """Return true when an event is likely to pollute normal acoustic clusters."""
    threshold = _mode_adjusted_quality_threshold(config)
    obvious_broadband_noise = (
        features.snr_db < config.noise_min_snr_db
        and features.spectral_flatness > config.noise_max_flatness
        and features.tonality_score < config.noise_min_tonality
    )
    weak_tonality = features.tonality_score < config.weak_tonality_floor
    return features.quality_score < threshold or obvious_broadband_noise or weak_tonality


def _spectral_tonality(stft_mag: np.ndarray) -> float:
    power = np.maximum(stft_mag.astype(np.float64) ** 2, 1e-20)
    column_sums = np.sum(power, axis=0, keepdims=True)
    probs = power / np.maximum(column_sums, 1e-20)
    entropy = -np.sum(probs * np.log(probs), axis=0) / np.log(max(2, power.shape[0]))
    tonality = 1.0 - entropy
    return float(np.nanmedian(np.nan_to_num(tonality, nan=0.0))) if tonality.size else 0.0


def _quality_score(
    snr_db: float,
    flatness: float,
    tonality: float,
    bandwidth_hz: float,
    peak_band_snr_db: float,
    sr: int,
    config: BioSoundConfig,
) -> float:
    snr_score = _smoothstep(config.noise_min_snr_db, config.noise_min_snr_db + 12.0, snr_db)
    flatness_score = 1.0 - _smoothstep(0.35, config.noise_max_flatness, flatness)
    tonality_score = _smoothstep(config.noise_min_tonality, 0.42, tonality)
    bandwidth_ratio = bandwidth_hz / max(1.0, sr / 2.0)
    bandwidth_score = 1.0 - _smoothstep(0.45, 0.95, bandwidth_ratio)
    peak_band_score = _smoothstep(config.noise_min_snr_db, config.noise_min_snr_db + 18.0, peak_band_snr_db)
    return (
        0.15 * snr_score
        + 0.30 * flatness_score
        + 0.20 * tonality_score
        + 0.05 * bandwidth_score
        + 0.30 * peak_band_score
    )


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    x = float(np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _mode_adjusted_quality_threshold(config: BioSoundConfig) -> float:
    adjustment = {
        "exploratory": -0.12,
        "balanced": 0.0,
        "conservative": 0.15,
    }[config.noise_mode]
    return float(np.clip(config.min_quality_for_clustering + adjustment, 0.0, 1.0))


def _attach_noise_features(event: AudioEvent, features: NoiseFeatures) -> None:
    event.snr_db = features.snr_db
    event.noise_floor_db = features.noise_floor_db
    event.spectral_flatness = features.spectral_flatness
    event.tonality_score = features.tonality_score
    event.bandwidth_hz = features.bandwidth_hz
    event.peak_band_snr_db = features.peak_band_snr_db
    event.quality_score = features.quality_score


def _mark_low_confidence_noise(event: AudioEvent, features: NoiseFeatures) -> AudioEvent:
    low = replace(event)
    _attach_noise_features(low, features)
    low.cluster_id = None
    low.cluster_probability = None
    low.is_noise = True
    low.is_low_confidence_noise = True
    low.source_type = "low_confidence_noise"
    return low
