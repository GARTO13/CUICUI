"""Global audio profiling — characterizes the recording so downstream stages can adapt.

This is the first stage of v4. The pipeline calls `profile_audio` on the loaded
recording, then `adaptive_config_from_profile` (in adaptive_config.py) maps the
profile to a config tuned for that file's acoustic regime.

Profile features are computed once per file from cheap NumPy/librosa primitives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np


@dataclass(slots=True)
class AudioProfile:
    """Global properties of a recording, used to pick adaptive parameters."""

    duration_sec: float
    sample_rate: int
    rms_p10_db: float
    rms_p50_db: float
    rms_p90_db: float
    dynamic_range_db: float
    active_fraction: float
    median_active_run_sec: float
    median_inactive_run_sec: float
    sustainedness: float
    transient_density_per_sec: float
    dominant_band_hz: float
    band_spread_hz: float
    spectral_stationarity: float
    median_tonality: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @property
    def regime(self) -> str:
        """Coarse acoustic regime, useful for logging and adaptive-config selection."""
        if self.sustainedness >= 1.5 and self.spectral_stationarity >= 0.85:
            return "sustained_narrowband"
        if self.transient_density_per_sec >= 6.0 and self.active_fraction < 0.40:
            return "transient_dense"
        if self.dynamic_range_db < 18.0:
            return "low_dynamic_range"
        if self.active_fraction >= 0.20 and self.spectral_stationarity < 0.70:
            return "dense_variable"
        return "default_sparse"


def profile_audio(
    audio: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> AudioProfile:
    """Profile a mono audio array. Returns an AudioProfile with global descriptors."""
    duration = float(len(audio) / sr) if sr else 0.0
    if len(audio) == 0 or duration <= 0:
        return _empty_profile(sr)

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0:
        return _empty_profile(sr, duration=duration)
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0)
    p10 = float(np.percentile(rms_db, 10))
    p50 = float(np.percentile(rms_db, 50))
    p90 = float(np.percentile(rms_db, 90))
    dynamic_range = p90 - p10

    active = rms_db > (p10 + 8.0)
    active_fraction = float(np.mean(active))
    med_active_run, med_inactive_run = _median_run_lengths(active)
    sec_per_frame = hop_length / sr
    med_active_sec = med_active_run * sec_per_frame
    med_inactive_sec = med_inactive_run * sec_per_frame
    sustainedness = float(med_active_sec / med_inactive_sec) if med_inactive_sec > 0 else float(med_active_sec)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        n_mels=64,
        power=2.0,
    )
    log_mel = librosa.power_to_db(np.maximum(mel, 1e-12), ref=1.0)
    if log_mel.shape[1] > 1:
        diff = np.diff(log_mel, axis=1)
        flux = np.sqrt(np.mean(np.square(np.maximum(diff, 0.0)), axis=0))
        flux_thr = float(np.percentile(flux, 95))
        transient_count = int(np.sum(flux > flux_thr))
        transient_density = float(transient_count / duration) if duration > 0 else 0.0
    else:
        transient_density = 0.0

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    dominant_band = float(np.nanmedian(centroid)) if centroid.size else 0.0
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    band_spread = float(np.nanmedian(bandwidth)) if bandwidth.size else 0.0

    stationarity = _spectral_stationarity(log_mel)
    median_tonality = _spectral_tonality(log_mel)

    return AudioProfile(
        duration_sec=duration,
        sample_rate=int(sr),
        rms_p10_db=p10,
        rms_p50_db=p50,
        rms_p90_db=p90,
        dynamic_range_db=float(dynamic_range),
        active_fraction=active_fraction,
        median_active_run_sec=float(med_active_sec),
        median_inactive_run_sec=float(med_inactive_sec),
        sustainedness=sustainedness,
        transient_density_per_sec=transient_density,
        dominant_band_hz=dominant_band,
        band_spread_hz=band_spread,
        spectral_stationarity=stationarity,
        median_tonality=median_tonality,
    )


def _empty_profile(sr: int, duration: float = 0.0) -> AudioProfile:
    return AudioProfile(
        duration_sec=float(duration),
        sample_rate=int(sr),
        rms_p10_db=-120.0,
        rms_p50_db=-120.0,
        rms_p90_db=-120.0,
        dynamic_range_db=0.0,
        active_fraction=0.0,
        median_active_run_sec=0.0,
        median_inactive_run_sec=0.0,
        sustainedness=0.0,
        transient_density_per_sec=0.0,
        dominant_band_hz=0.0,
        band_spread_hz=0.0,
        spectral_stationarity=1.0,
        median_tonality=0.0,
    )


def _median_run_lengths(active: np.ndarray) -> tuple[float, float]:
    """Return median active-run and median inactive-run lengths in frames."""
    if active.size == 0:
        return 0.0, 0.0
    changes = np.diff(active.astype(np.int8))
    edges = np.concatenate(([-1], np.flatnonzero(changes), [active.size - 1]))
    runs = np.diff(edges)
    if runs.size == 0:
        return 0.0, 0.0
    values = active[edges[:-1] + 1]
    active_runs = runs[values]
    inactive_runs = runs[~values]
    med_active = float(np.median(active_runs)) if active_runs.size else 0.0
    med_inactive = float(np.median(inactive_runs)) if inactive_runs.size else 0.0
    return med_active, med_inactive


def _spectral_stationarity(log_mel: np.ndarray) -> float:
    """How similar are consecutive log-mel frames? 1 = perfectly stationary, 0 = chaotic."""
    if log_mel.shape[1] < 2:
        return 1.0
    norm = np.linalg.norm(log_mel, axis=0, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    normalized = log_mel / norm
    cosines = np.sum(normalized[:, :-1] * normalized[:, 1:], axis=0)
    return float(np.nanmean(cosines))


def _spectral_tonality(log_mel: np.ndarray) -> float:
    """Per-frame negentropy of log-mel, median across frames. High = tonal/peaky, low = noisy."""
    if log_mel.size == 0:
        return 0.0
    power = np.exp(log_mel * np.log(10) / 10.0)
    power = np.maximum(power, 1e-20)
    col_sums = np.sum(power, axis=0, keepdims=True)
    probs = power / np.maximum(col_sums, 1e-20)
    entropy = -np.sum(probs * np.log(probs + 1e-20), axis=0) / np.log(max(2, power.shape[0]))
    tonality = 1.0 - entropy
    return float(np.nanmedian(np.nan_to_num(tonality, nan=0.0)))
