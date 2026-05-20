"""Broad unsupervised candidate event detection."""

from __future__ import annotations

from collections.abc import Sequence

import librosa
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, percentile_filter
from scipy.signal import butter, sosfiltfilt

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.schemas import AudioEvent


def compute_rms_db(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Compute frame-wise RMS energy in dB."""
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return librosa.amplitude_to_db(rms, ref=1.0, top_db=None)


def rolling_percentile(values: np.ndarray, percentile: float = 20.0, window_frames: int = 101) -> np.ndarray:
    """Estimate a rolling percentile noise floor."""
    if values.size == 0:
        return values
    size = max(3, int(window_frames))
    if size % 2 == 0:
        size += 1
    return percentile_filter(values, percentile=percentile, size=size, mode="nearest")


def merge_intervals(intervals: Sequence[tuple[float, float]], merge_gap: float) -> list[tuple[float, float]]:
    """Merge sorted intervals separated by no more than merge_gap seconds."""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda pair: pair[0])
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def split_long_intervals(
    intervals: Sequence[tuple[float, float]],
    max_duration: float,
    min_duration: float,
) -> list[tuple[float, float]]:
    """Split long intervals into overlapping chunks."""
    split: list[tuple[float, float]] = []
    hop = max_duration * 0.5
    for start, end in intervals:
        duration = end - start
        if duration <= max_duration:
            split.append((start, end))
            continue
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + max_duration, end)
            if chunk_end - cursor >= min_duration:
                split.append((cursor, chunk_end))
            if chunk_end >= end:
                break
            cursor += hop
    return split


def compute_event_stats(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> dict[str, float | None]:
    """Compute simple descriptive audio stats for one event."""
    start = max(0, int(round(start_sec * sr)))
    end = min(len(audio), int(round(end_sec * sr)))
    clip = audio[start:end]
    if clip.size == 0:
        return {"rms_db": None, "peak_db": None, "spectral_centroid": None}

    rms = float(np.sqrt(np.mean(np.square(clip)) + 1e-12))
    peak = float(np.max(np.abs(clip)) + 1e-12)
    centroid = librosa.feature.spectral_centroid(y=clip, sr=sr)
    return {
        "rms_db": float(librosa.amplitude_to_db(np.array([rms]), ref=1.0, top_db=None)[0]),
        "peak_db": float(librosa.amplitude_to_db(np.array([peak]), ref=1.0, top_db=None)[0]),
        "spectral_centroid": float(np.mean(centroid)) if centroid.size else None,
    }


def _spectral_flux(audio: np.ndarray, sr: int, config: BioSoundConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=config.frame_length,
        hop_length=config.hop_length,
        n_mels=64,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max if np.max(mel) > 0 else 1.0)
    diff = np.diff(log_mel, axis=1)
    flux = np.sqrt(np.mean(np.square(np.maximum(diff, 0.0)), axis=0))
    return np.pad(flux, (1, 0), mode="edge") if flux.size else np.array([], dtype=np.float32)


def _frame_spectral_flatness(audio: np.ndarray, config: BioSoundConfig) -> np.ndarray:
    """Compute per-frame spectral flatness; low values mean frequency-concentrated energy."""
    if audio.size == 0:
        return np.array([], dtype=np.float32)
    stft_mag = np.abs(
        librosa.stft(
            y=audio,
            n_fft=config.frame_length,
            hop_length=config.hop_length,
            center=True,
        )
    )
    if stft_mag.size == 0:
        return np.array([], dtype=np.float32)
    flatness = librosa.feature.spectral_flatness(S=stft_mag)[0]
    return np.nan_to_num(flatness, nan=1.0).astype(np.float32)


def _default_frequency_bands(sr: int) -> list[tuple[str, float, float]]:
    nyquist = sr / 2.0
    candidates = [
        ("low", 100.0, 800.0),
        ("mid", 800.0, 3000.0),
        ("high", 3000.0, 8000.0),
        ("ultra_high", 8000.0, nyquist),
    ]
    return [
        (name, low, min(high, nyquist * 0.98))
        for name, low, high in candidates
        if low < nyquist * 0.95 and min(high, nyquist * 0.98) > low
    ]


def _bandpass(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    nyquist = sr / 2.0
    low = max(1.0, low_hz) / nyquist
    high = min(high_hz, nyquist * 0.98) / nyquist
    if not 0 < low < high < 1:
        return audio.astype(np.float32)
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    if audio.size <= 3 * sos.shape[0] * 2:
        return audio.astype(np.float32)
    try:
        return sosfiltfilt(sos, audio).astype(np.float32)
    except ValueError:
        return audio.astype(np.float32)


def _robust_flux_z(values: np.ndarray) -> np.ndarray:
    diff = np.pad(np.maximum(np.diff(values), 0.0), (1, 0), mode="edge") if values.size > 1 else np.zeros_like(values)
    median = float(np.median(diff)) if diff.size else 0.0
    mad = float(np.median(np.abs(diff - median))) + 1e-9
    return (diff - median) / (1.4826 * mad)


def _multiband_activity(audio: np.ndarray, sr: int, config: BioSoundConfig, target_len: int) -> np.ndarray:
    """Detect structured local activity in coarse frequency bands."""
    if target_len <= 0 or audio.size == 0:
        return np.zeros(target_len, dtype=bool)

    combined = np.zeros(target_len, dtype=bool)
    window_frames = max(5, int(round(config.multiband_local_noise_window_sec * sr / config.hop_length)))
    for _, low_hz, high_hz in _default_frequency_bands(sr):
        band_audio = _bandpass(audio, sr, low_hz, high_hz)
        band_rms_db = compute_rms_db(band_audio, config.frame_length, config.hop_length)
        if band_rms_db.size == 0:
            continue
        band_rms_db = band_rms_db[:target_len]
        if band_rms_db.size < target_len:
            band_rms_db = np.pad(band_rms_db, (0, target_len - band_rms_db.size), mode="edge")
        noise_floor = rolling_percentile(band_rms_db, percentile=20.0, window_frames=window_frames)
        local_snr = band_rms_db - noise_floor
        flux_z = _robust_flux_z(band_rms_db)
        flatness = _frame_spectral_flatness(band_audio, config)[:target_len]
        if flatness.size == 0:
            flatness = np.ones(target_len, dtype=np.float32)
        elif flatness.size < target_len:
            flatness = np.pad(flatness, (0, target_len - flatness.size), mode="edge")
        structured = flatness < config.spectral_concentration_max_flatness
        band_active = (
            (local_snr >= config.multiband_min_snr_db)
            & ((flux_z >= config.multiband_min_flux_z) | structured)
        )
        combined |= band_active
    close_frames = max(1, int(round(config.multiband_merge_gap_sec * sr / config.hop_length)))
    combined = binary_closing(combined, structure=np.ones(close_frames, dtype=bool))
    return combined


def _active_frames_to_intervals(active: np.ndarray, sr: int, hop_length: int, duration_sec: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    in_event = False
    start_frame = 0
    for idx, value in enumerate(active):
        if value and not in_event:
            in_event = True
            start_frame = idx
        elif not value and in_event:
            in_event = False
            start_sec = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            end_sec = librosa.frames_to_time(idx, sr=sr, hop_length=hop_length)
            intervals.append((float(start_sec), float(end_sec)))
    if in_event:
        start_sec = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        intervals.append((float(start_sec), duration_sec))
    return intervals


def detect_candidate_events(audio: np.ndarray, sr: int, config: BioSoundConfig) -> list[AudioEvent]:
    """Detect broad candidate acoustic events from energy and spectral structure."""
    duration_sec = float(len(audio) / sr)
    if len(audio) == 0 or duration_sec < config.min_event_duration:
        return []

    rms_db = compute_rms_db(audio, config.frame_length, config.hop_length)
    if rms_db.size == 0:
        return []
    window_seconds = 10.0
    window_frames = max(11, int(round(window_seconds * sr / config.hop_length)))
    noise_floor = rolling_percentile(rms_db, percentile=20.0, window_frames=window_frames)
    energy_active = rms_db > (noise_floor + config.threshold_db)

    flux = _spectral_flux(audio, sr, config) if config.enable_flux_detection else np.zeros_like(energy_active, dtype=float)
    min_len = min(len(energy_active), len(flux))
    energy_active = energy_active[:min_len]
    flux = flux[:min_len]
    if flux.size:
        median = float(np.median(flux))
        mad = float(np.median(np.abs(flux - median))) + 1e-9
        flux_threshold = max(
            float(np.percentile(flux, config.flux_percentile)),
            median + config.flux_mad_multiplier * mad,
        )
        flux_active = (flux > flux_threshold) & (rms_db[:min_len] > (noise_floor[:min_len] + config.flux_min_snr_db))
    else:
        flux_active = np.zeros_like(energy_active, dtype=bool)

    if config.enable_spectral_concentration_gate:
        flatness = _frame_spectral_flatness(audio, config)[:min_len]
        if flatness.size < min_len:
            flatness = np.pad(flatness, (0, min_len - flatness.size), mode="edge")
        concentrated_active = flatness < config.spectral_concentration_max_flatness
    else:
        concentrated_active = np.ones_like(energy_active, dtype=bool)

    votes = energy_active.astype(np.int8) + flux_active.astype(np.int8) + concentrated_active.astype(np.int8)
    active = votes >= config.segmentation_min_active_votes
    if config.enable_multiband_segmentation:
        active = np.logical_or(active, _multiband_activity(audio, sr, config, min_len))
    close_frames = max(1, int(round(config.merge_gap * sr / config.hop_length)))
    open_frames = max(1, int(round(0.04 * sr / config.hop_length)))
    active = binary_opening(active, structure=np.ones(open_frames, dtype=bool))
    active = binary_closing(active, structure=np.ones(close_frames, dtype=bool))

    intervals = _active_frames_to_intervals(active, sr, config.hop_length, duration_sec)
    padded = [
        (max(0.0, start - config.padding), min(duration_sec, end + config.padding))
        for start, end in intervals
    ]
    merged = merge_intervals(padded, config.merge_gap)
    filtered = [(s, e) for s, e in merged if (e - s) >= config.min_event_duration]
    split = split_long_intervals(filtered, config.max_event_duration, config.min_event_duration)
    if config.max_events is not None:
        split = split[: config.max_events]

    events: list[AudioEvent] = []
    frame_times = librosa.frames_to_time(np.arange(min_len), sr=sr, hop_length=config.hop_length)
    snr_values = rms_db[:min_len] - noise_floor[:min_len]
    flux_norm = flux / (float(np.percentile(flux, 95)) + 1e-9) if flux.size else np.zeros(min_len, dtype=float)
    for idx, (start, end) in enumerate(split):
        stats = compute_event_stats(audio, sr, start, end)
        frame_mask = (frame_times >= start) & (frame_times <= end)
        if np.any(frame_mask):
            mean_snr = float(np.mean(snr_values[frame_mask]))
            mean_flux = float(np.mean(np.clip(flux_norm[frame_mask], 0.0, 1.0)))
            active_fraction = float(np.mean(active[frame_mask]))
            detection_score = float(np.clip(0.45 * (mean_snr / 20.0) + 0.35 * mean_flux + 0.20 * active_fraction, 0.0, 1.0))
            stationarity = float(1.0 - np.clip(np.std(snr_values[frame_mask]) / 12.0, 0.0, 1.0))
        else:
            mean_snr = 0.0
            mean_flux = 0.0
            active_fraction = 0.0
            detection_score = 0.0
            stationarity = 1.0
        events.append(
            AudioEvent(
                event_id=f"event_{idx:06d}",
                start_sec=float(start),
                end_sec=float(end),
                duration_sec=float(end - start),
                rms_db=stats["rms_db"],
                peak_db=stats["peak_db"],
                spectral_centroid=stats["spectral_centroid"],
                detection_score=detection_score,
                event_snr_db=mean_snr,
                active_band_fraction=active_fraction,
                spectral_flux_score=mean_flux,
                stationarity_score=stationarity,
            )
        )
    return events
