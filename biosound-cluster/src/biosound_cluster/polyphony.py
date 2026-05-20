"""Polyphony detection and lightweight time-frequency component separation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import librosa
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, label

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.schemas import AudioEvent

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TFComponent:
    """A connected active region in the event time-frequency plane."""

    component_id: int
    time_start_frame: int
    time_end_frame: int
    freq_start_bin: int
    freq_end_bin: int
    area: int
    energy: float
    energy_ratio: float
    duration_sec: float
    bandwidth_hz: float
    mask: np.ndarray
    purity_score: float
    compactness: float = 0.0
    snr_db: float = 0.0
    quality_score: float = 0.0


def _stft_params(clip: np.ndarray, config: BioSoundConfig) -> tuple[int, int]:
    n_fft = min(config.polyphony_n_fft, max(256, 2 ** int(np.floor(np.log2(max(256, clip.size))))))
    hop_length = min(config.polyphony_hop_length, max(64, n_fft // 4))
    return n_fft, hop_length


def extract_event_audio(audio: np.ndarray, sr: int, event: AudioEvent) -> np.ndarray:
    """Extract original waveform audio for an event."""
    start = max(0, int(round(event.start_sec * sr)))
    end = min(len(audio), int(round(event.end_sec * sr)))
    return np.asarray(audio[start:end], dtype=np.float32)


def compute_stft(clip: np.ndarray, config: BioSoundConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return a complex STFT and its magnitude spectrogram in dB."""
    if clip.size == 0:
        return np.empty((0, 0), dtype=np.complex64), np.empty((0, 0), dtype=np.float32)
    n_fft, hop_length = _stft_params(clip, config)
    stft_complex = librosa.stft(
        y=clip,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
    )
    magnitude = np.abs(stft_complex)
    reference = float(np.max(magnitude)) if magnitude.size and np.max(magnitude) > 0 else 1.0
    stft_mag_db = librosa.amplitude_to_db(magnitude, ref=reference, top_db=None)
    return stft_complex.astype(np.complex64, copy=False), stft_mag_db.astype(np.float32, copy=False)


def build_activity_mask(stft_mag_db: np.ndarray, config: BioSoundConfig) -> np.ndarray:
    """Build a denoised active time-frequency mask above a per-band noise floor."""
    if stft_mag_db.size == 0:
        return np.zeros_like(stft_mag_db, dtype=bool)
    noise_floor_db = np.percentile(stft_mag_db, 20, axis=1, keepdims=True)
    mask = (stft_mag_db - noise_floor_db) > config.component_snr_db
    structure = np.ones((3, 3), dtype=bool)
    mask = binary_opening(mask, structure=structure)
    mask = binary_closing(mask, structure=structure)
    return np.asarray(mask, dtype=bool)


def find_tf_components(
    mask: np.ndarray,
    stft_mag: np.ndarray,
    sr: int,
    config: BioSoundConfig,
) -> list[TFComponent]:
    """Find connected time-frequency components in an activity mask."""
    if mask.size == 0 or stft_mag.size == 0:
        return []
    labeled, n_labels = label(mask)
    total_active_energy = float(np.sum(np.square(stft_mag[mask])) + 1e-12)
    hz_per_bin = (sr / 2.0) / max(1, mask.shape[0] - 1)
    components: list[TFComponent] = []

    for label_id in range(1, n_labels + 1):
        component_mask = labeled == label_id
        area = int(np.count_nonzero(component_mask))
        if area == 0:
            continue
        freq_idx, time_idx = np.nonzero(component_mask)
        f0, f1 = int(freq_idx.min()), int(freq_idx.max())
        t0, t1 = int(time_idx.min()), int(time_idx.max())
        energy = float(np.sum(np.square(stft_mag[component_mask])))
        energy_ratio = float(energy / total_active_energy)
        duration_sec = float((t1 - t0 + 1) * config.polyphony_hop_length / sr)
        bandwidth_hz = float(max(1, f1 - f0 + 1) * hz_per_bin)
        bbox = stft_mag[f0 : f1 + 1, t0 : t1 + 1]
        bbox_energy = float(np.sum(np.square(bbox)) + 1e-12)
        isolation = float(energy / bbox_energy)
        bbox_area = max(1, (f1 - f0 + 1) * (t1 - t0 + 1))
        compactness = float(area / bbox_area)
        background = stft_mag[np.logical_not(component_mask)]
        noise = float(np.median(np.square(background)) + 1e-12) if background.size else 1e-12
        snr_db = float(10.0 * np.log10((energy / max(area, 1)) / noise))
        share_score = min(1.0, energy_ratio / max(config.min_component_energy_ratio, 1e-6))
        purity_score = float(np.clip(0.45 * isolation + 0.35 * share_score + 0.20 * compactness, 0.0, 1.0))
        component = TFComponent(
            component_id=len(components),
            time_start_frame=t0,
            time_end_frame=t1,
            freq_start_bin=f0,
            freq_end_bin=f1,
            area=area,
            energy=energy,
            energy_ratio=energy_ratio,
            duration_sec=duration_sec,
            bandwidth_hz=bandwidth_hz,
            mask=component_mask,
            purity_score=purity_score,
            compactness=compactness,
            snr_db=snr_db,
        )
        component.quality_score = compute_component_quality_score(component, config)
        components.append(component)
    return components


def compute_component_quality_score(component: TFComponent, config: BioSoundConfig) -> float:
    """
    Score between 0 and 1.
    Rewards energetic, compact, clean, low-overlap components.
    """
    purity_score = float(np.clip(component.purity_score, 0.0, 1.0))
    snr_score = float(np.clip(component.snr_db / max(config.min_component_snr_db_for_clustering * 2.0, 1e-6), 0.0, 1.0))
    energy_ratio_score = float(np.clip(component.energy_ratio / max(config.min_component_energy_ratio_for_clustering * 2.0, 1e-6), 0.0, 1.0))
    compactness_score = float(np.clip(component.compactness / max(config.min_component_compactness * 3.0, 1e-6), 0.0, 1.0))
    duration_score = float(np.clip(component.duration_sec / max(config.component_min_duration * 3.0, 1e-6), 0.0, 1.0))
    return float(
        np.clip(
            0.30 * purity_score
            + 0.25 * snr_score
            + 0.20 * energy_ratio_score
            + 0.15 * compactness_score
            + 0.10 * duration_score,
            0.0,
            1.0,
        )
    )


def should_component_enter_clustering(
    component: TFComponent,
    rank_in_parent: int,
    config: BioSoundConfig,
) -> tuple[bool, str]:
    """Return whether a separated component is reliable enough for normal clustering."""
    if not config.enable_component_explosion_control:
        return True, "legacy_polyphony_routing"
    if rank_in_parent >= config.max_components_for_clustering_per_parent:
        return False, "component_rank_exceeds_clustering_limit"
    if component.purity_score < config.min_component_purity_for_clustering_strict:
        return False, "component_purity_below_threshold"
    if component.energy_ratio < config.min_component_energy_ratio_for_clustering:
        return False, "component_energy_ratio_below_threshold"
    if component.snr_db < config.min_component_snr_db_for_clustering:
        return False, "component_snr_below_threshold"
    if component.compactness < config.min_component_compactness:
        return False, "component_compactness_below_threshold"
    if component.quality_score < config.min_component_quality_for_clustering:
        return False, "component_quality_below_threshold"
    return True, "clusterable_component"


def filter_components(components: list[TFComponent], config: BioSoundConfig) -> list[TFComponent]:
    """Filter tiny or weak time-frequency components."""
    strong_narrowband_ratio = max(0.02, config.min_component_energy_ratio * 2.0)
    filtered = [
        component
        for component in components
        if component.area >= config.component_min_area
        and component.duration_sec >= config.component_min_duration
        and (
            component.bandwidth_hz >= config.component_min_bandwidth_hz
            or component.energy_ratio >= strong_narrowband_ratio
        )
        and component.energy_ratio >= config.min_component_energy_ratio
    ]
    filtered.sort(key=lambda component: component.energy, reverse=True)
    max_export = config.max_components_for_export_per_parent if config.enable_component_explosion_control else config.max_components_per_event
    filtered = filtered[: max_export]
    for idx, component in enumerate(filtered):
        component.component_id = idx
    return filtered


def compute_overlap_ratio(components: list[TFComponent]) -> float:
    """Measure temporal overlap among components, from 0 to 1."""
    if len(components) < 2:
        return 0.0
    max_frame = max(component.time_end_frame for component in components)
    counts = np.zeros(max_frame + 1, dtype=np.int16)
    for component in components:
        counts[component.time_start_frame : component.time_end_frame + 1] += 1
    active = counts > 0
    if not np.any(active):
        return 0.0
    return float(np.count_nonzero(counts > 1) / np.count_nonzero(active))


def compute_polyphony_score(components: list[TFComponent], overlap_ratio: float) -> float:
    """Return a 0-1 score where higher means more likely polyphonic."""
    if len(components) <= 1:
        return 0.0
    energy_balance = 1.0 - max(component.energy_ratio for component in components)
    component_factor = min(1.0, (len(components) - 1) / 3.0)
    score = 0.45 * component_factor + 0.35 * energy_balance + 0.20 * overlap_ratio
    return float(np.clip(score, 0.0, 1.0))


def separate_component_audio(
    clip: np.ndarray,
    stft_complex: np.ndarray,
    component: TFComponent,
    config: BioSoundConfig,
) -> np.ndarray:
    """Reconstruct one component as mono audio using a smoothed STFT mask."""
    if clip.size == 0 or stft_complex.size == 0:
        return np.zeros(0, dtype=np.float32)
    soft_mask = gaussian_filter(component.mask.astype(np.float32), sigma=1.0)
    max_mask = float(np.max(soft_mask)) if soft_mask.size else 0.0
    if max_mask > 0:
        soft_mask = soft_mask / max_mask
    masked_stft = stft_complex * np.clip(soft_mask, 0.0, 1.0)
    _, hop_length = _stft_params(clip, config)
    reconstructed = librosa.istft(
        masked_stft,
        hop_length=hop_length,
        length=len(clip),
    )
    reconstructed = np.nan_to_num(reconstructed.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(reconstructed))) if reconstructed.size else 0.0
    if peak > 1.0:
        reconstructed = reconstructed / peak
    return reconstructed.astype(np.float32, copy=False)


def analyze_and_split_events(
    audio: np.ndarray,
    sr: int,
    events: list[AudioEvent],
    config: BioSoundConfig,
) -> tuple[list[AudioEvent], list[AudioEvent]]:
    """Analyze candidate events and route clean, separated, and mixed sounds."""
    clusterable_events: list[AudioEvent] = []
    mixed_events: list[AudioEvent] = []
    for event in events:
        try:
            clean_or_components, mixed = _analyze_one_event(audio, sr, event, config)
            clusterable_events.extend(clean_or_components)
            mixed_events.extend(mixed)
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("Polyphony analysis failed for %s; marking mixed: %s", event.event_id, exc)
            mixed_events.append(_mark_mixed(event, n_components=1, polyphony_score=1.0, purity_score=0.0))
    return clusterable_events, mixed_events


def _analyze_one_event(
    audio: np.ndarray,
    sr: int,
    event: AudioEvent,
    config: BioSoundConfig,
) -> tuple[list[AudioEvent], list[AudioEvent]]:
    clip = extract_event_audio(audio, sr, event)
    if clip.size == 0:
        return [event], []
    stft_complex, stft_mag_db = compute_stft(clip, config)
    stft_mag = np.abs(stft_complex)
    mask = build_activity_mask(stft_mag_db, config)
    components = filter_components(find_tf_components(mask, stft_mag, sr, config), config)
    overlap_ratio = compute_overlap_ratio(components)
    polyphony_score = compute_polyphony_score(components, overlap_ratio)

    if len(components) <= 1:
        purity = components[0].purity_score if components else 1.0
        event.source_type = "original"
        event.is_mixed = False
        event.is_overlapping = False
        event.n_components = max(1, len(components))
        event.polyphony_score = polyphony_score
        event.purity_score = max(purity, 0.85)
        return [event], []

    dominant_ratio = max(component.energy_ratio for component in components)
    if dominant_ratio >= 1.0 - config.min_component_energy_ratio:
        event.source_type = "original"
        event.is_mixed = False
        event.is_overlapping = False
        event.n_components = len(components)
        event.polyphony_score = polyphony_score
        event.purity_score = max(component.purity_score for component in components)
        return [event], []

    pure_components = [
        component for component in components if component.purity_score >= config.min_component_purity_for_separation
    ]
    mean_purity = float(np.mean([component.purity_score for component in components]))
    separable = (
        polyphony_score >= config.min_polyphony_score_for_separation
        and (
            overlap_ratio <= config.max_overlap_ratio_for_separation
            or mean_purity >= config.min_component_purity_for_separation
        )
    )
    if config.enable_component_explosion_control:
        if config.mark_parent_mixed_when_too_many_components and len(components) > config.max_significant_components_before_mixed:
            return [], [_mark_mixed(event, len(components), polyphony_score, mean_purity, "too_many_significant_components")]
        if config.polyphony_split_requires_low_overlap and overlap_ratio > config.strict_max_overlap_ratio_for_split:
            return [], [_mark_mixed(event, len(components), polyphony_score, mean_purity, "components_too_overlapping")]
        if config.polyphony_split_requires_compact_masks and any(
            component.compactness < config.min_component_compactness for component in pure_components
        ):
            return [], [_mark_mixed(event, len(components), polyphony_score, mean_purity, "component_masks_not_compact")]

    if separable and len(pure_components) >= 2:
        ordered = sorted(pure_components, key=lambda item: item.quality_score, reverse=True)
        component_events = [
            _component_event(audio, sr, clip, stft_complex, event, component, len(components), polyphony_score, config, rank)
            for rank, component in enumerate(ordered)
        ]
        component_events = [component_event for component_event in component_events if component_event is not None]
        if component_events:
            return component_events, []

    return [], [_mark_mixed(event, len(components), polyphony_score, mean_purity, "not_separable")]


def _component_event(
    audio: np.ndarray,
    sr: int,
    clip: np.ndarray,
    stft_complex: np.ndarray,
    parent: AudioEvent,
    component: TFComponent,
    n_components: int,
    polyphony_score: float,
    config: BioSoundConfig,
    rank_in_parent: int = 0,
) -> AudioEvent | None:
    separated = separate_component_audio(clip, stft_complex, component, config)
    _, hop_length = _stft_params(clip, config)
    offset_start = component.time_start_frame * hop_length / sr
    offset_end = min(len(clip) / sr, (component.time_end_frame + 1) * hop_length / sr)
    start_sample = max(0, int(round(offset_start * sr)))
    end_sample = min(len(separated), int(round(offset_end * sr)))
    if end_sample <= start_sample:
        return None
    separated_clip = separated[start_sample:end_sample]
    if separated_clip.size < int(round(config.component_min_duration * sr)):
        return None
    start_sec = parent.start_sec + offset_start
    end_sec = parent.start_sec + offset_end
    stats = _audio_stats(separated_clip, sr)
    clusterable, reason = should_component_enter_clustering(component, rank_in_parent, config)
    event = AudioEvent(
        event_id=f"{parent.event_id}_component_{component.component_id}",
        parent_event_id=parent.event_id,
        component_id=component.component_id,
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        duration_sec=float(end_sec - start_sec),
        rms_db=stats["rms_db"],
        peak_db=stats["peak_db"],
        spectral_centroid=stats["spectral_centroid"],
        is_component=True,
        is_overlapping=True,
        is_mixed=False,
        n_components=n_components,
        polyphony_score=polyphony_score,
        purity_score=component.purity_score,
        source_type="component" if clusterable else "component_review",
        component_energy_ratio=component.energy_ratio,
        component_snr_db=component.snr_db,
        component_compactness=component.compactness,
        component_quality_score=component.quality_score,
        is_component_review=not clusterable,
        component_rank_in_parent=rank_in_parent,
        clusterable=clusterable,
        routing_reason=reason,
        separated_audio=separated_clip,
        context_audio=extract_event_audio(audio, sr, parent),
    )
    return event


def _mark_mixed(
    event: AudioEvent,
    n_components: int,
    polyphony_score: float,
    purity_score: float,
    routing_reason: str | None = None,
) -> AudioEvent:
    mixed = replace(event)
    mixed.cluster_id = None
    mixed.cluster_probability = None
    mixed.is_noise = False
    mixed.is_component = False
    mixed.is_overlapping = True
    mixed.is_mixed = True
    mixed.n_components = max(1, n_components)
    mixed.polyphony_score = polyphony_score
    mixed.purity_score = purity_score
    mixed.source_type = "mixed"
    mixed.clusterable = False
    mixed.routing_reason = routing_reason or "mixed_overlapping"
    return mixed


def _audio_stats(clip: np.ndarray, sr: int) -> dict[str, float | None]:
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
