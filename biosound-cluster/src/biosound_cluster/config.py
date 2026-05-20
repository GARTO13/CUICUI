"""Configuration for biosound-cluster."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class BioSoundConfig:
    """Runtime settings for event detection, embeddings, clustering, and export."""

    sample_rate: int = 32000
    frame_length: int = 2048
    hop_length: int = 512
    threshold_db: float = 8.0
    flux_percentile: float = 90.0
    flux_mad_multiplier: float = 4.0
    flux_min_snr_db: float = 1.5
    enable_flux_detection: bool = True
    enable_spectral_concentration_gate: bool = True
    spectral_concentration_max_flatness: float = 0.40
    segmentation_min_active_votes: int = 2
    enable_multiband_segmentation: bool = True
    multiband_local_noise_window_sec: float = 8.0
    multiband_min_snr_db: float = 6.0
    multiband_min_flux_z: float = 2.0
    multiband_merge_gap_sec: float = 0.08
    enable_segmentation_refinement: bool = True
    refinement_activity_db: float = 3.0
    refinement_peak_drop_db: float = 18.0
    refinement_padding: float = 0.03
    refinement_min_trim_sec: float = 0.04
    refinement_nms_iou: float = 0.85
    min_event_duration: float = 0.25
    max_event_duration: float = 8.0
    merge_gap: float = 0.4
    padding: float = 0.15
    min_cluster_size: int = 10
    min_samples: int | None = None
    umap_components: int = 10
    umap_neighbors: int = 30
    umap_metric: str = "cosine"
    umap_min_dist: float = 0.05
    max_events: int | None = None
    embedding_include_delta: bool = True
    generate_spectrograms: bool = True
    export_clips: bool = True
    random_state: int = 42
    sensor_id: str | None = None
    sensor_latitude: float | None = None
    sensor_longitude: float | None = None
    sensor_elevation_m: float | None = None
    environment_type: str | None = None
    recording_start_time: str | None = None
    recording_timezone: str | None = None
    enable_polyphony_handling: bool = True
    polyphony_n_fft: int = 2048
    polyphony_hop_length: int = 512
    component_snr_db: float = 10.0
    component_min_area: int = 20
    component_min_duration: float = 0.08
    component_min_bandwidth_hz: float = 150.0
    max_components_per_event: int = 4
    min_component_energy_ratio: float = 0.12
    min_polyphony_score_for_separation: float = 0.25
    min_component_purity_for_separation: float = 0.55
    min_purity_for_clustering: float = 0.55
    max_overlap_ratio_for_separation: float = 0.65
    export_original_context: bool = True
    export_mixed_overlapping: bool = True
    enable_eventness_filtering: bool = True
    min_eventness_for_clustering: float = 0.28
    min_component_eventness_for_clustering: float = 0.50
    eventness_min_contrast_db: float = 4.0
    eventness_max_active_ratio: float = 0.92
    enable_candidate_selection: bool = True
    max_components_per_parent: int = 3
    candidate_nms_iou: float = 0.98
    enable_short_event_review: bool = True
    min_review_event_duration: float = 0.20
    export_short_events_review: bool = True
    enable_clusterability_filtering: bool = True
    min_clusterability_for_clustering: float = 0.55
    min_clusterability_for_review: float = 0.30
    enable_embedding_stability: bool = True
    embedding_stability_min_for_clustering: float = 0.60
    embedding_stability_views: tuple[str, ...] = ("original", "center_crop", "band_limited")
    enable_acoustic_prefamilies: bool = True
    prefamily_min_events: int = 15
    cluster_ensemble_runs: int = 1
    min_cluster_stability: float = 0.65
    enable_auto_profile: bool = False
    enable_semantic_tagging: bool = False
    enable_semantic_gate: bool = False
    semantic_bio_threshold: float = 0.20
    semantic_noise_threshold: float = 0.60
    enable_denoiser: bool = False
    denoiser_name: str = "biodenoising"
    enable_noise_filtering: bool = True
    noise_mode: str = "balanced"
    noise_min_snr_db: float = 3.0
    noise_max_flatness: float = 0.72
    noise_min_tonality: float = 0.08
    weak_tonality_floor: float = 0.20
    min_quality_for_clustering: float = 0.55
    export_low_confidence_noise: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_length <= 0:
            raise ValueError("frame_length must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.threshold_db < 0:
            raise ValueError("threshold_db must be non-negative")
        if not 0 < self.flux_percentile < 100:
            raise ValueError("flux_percentile must be in (0, 100)")
        if self.flux_mad_multiplier < 0:
            raise ValueError("flux_mad_multiplier must be non-negative")
        if self.flux_min_snr_db < 0:
            raise ValueError("flux_min_snr_db must be non-negative")
        if not 0 < self.spectral_concentration_max_flatness <= 1:
            raise ValueError("spectral_concentration_max_flatness must be in (0, 1]")
        if self.segmentation_min_active_votes not in (1, 2, 3):
            raise ValueError("segmentation_min_active_votes must be 1, 2, or 3")
        if self.multiband_local_noise_window_sec <= 0:
            raise ValueError("multiband_local_noise_window_sec must be positive")
        if self.multiband_min_snr_db < 0:
            raise ValueError("multiband_min_snr_db must be non-negative")
        if self.multiband_min_flux_z < 0:
            raise ValueError("multiband_min_flux_z must be non-negative")
        if self.multiband_merge_gap_sec < 0:
            raise ValueError("multiband_merge_gap_sec must be non-negative")
        if self.refinement_activity_db < 0:
            raise ValueError("refinement_activity_db must be non-negative")
        if self.refinement_peak_drop_db < 0:
            raise ValueError("refinement_peak_drop_db must be non-negative")
        if self.refinement_padding < 0:
            raise ValueError("refinement_padding must be non-negative")
        if self.refinement_min_trim_sec < 0:
            raise ValueError("refinement_min_trim_sec must be non-negative")
        if not 0 <= self.refinement_nms_iou <= 1:
            raise ValueError("refinement_nms_iou must be in [0, 1]")
        if self.min_event_duration <= 0:
            raise ValueError("min_event_duration must be positive")
        if self.max_event_duration <= 0:
            raise ValueError("max_event_duration must be positive")
        if self.min_event_duration > self.max_event_duration:
            raise ValueError("min_event_duration must be <= max_event_duration")
        if self.merge_gap < 0:
            raise ValueError("merge_gap must be non-negative")
        if self.padding < 0:
            raise ValueError("padding must be non-negative")
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        if self.min_samples is not None and self.min_samples < 1:
            raise ValueError("min_samples must be None or positive")
        if self.umap_components < 2:
            raise ValueError("umap_components must be at least 2")
        if self.umap_neighbors < 2:
            raise ValueError("umap_neighbors must be at least 2")
        if self.umap_min_dist < 0:
            raise ValueError("umap_min_dist must be non-negative")
        if self.max_events is not None and self.max_events < 1:
            raise ValueError("max_events must be None or positive")
        if not self.export_clips and self.generate_spectrograms:
            raise ValueError("generate_spectrograms requires export_clips")
        if self.sensor_latitude is not None and not -90 <= self.sensor_latitude <= 90:
            raise ValueError("sensor_latitude must be between -90 and 90")
        if self.sensor_longitude is not None and not -180 <= self.sensor_longitude <= 180:
            raise ValueError("sensor_longitude must be between -180 and 180")
        if self.polyphony_n_fft <= 0:
            raise ValueError("polyphony_n_fft must be positive")
        if self.polyphony_hop_length <= 0:
            raise ValueError("polyphony_hop_length must be positive")
        if self.component_snr_db < 0:
            raise ValueError("component_snr_db must be non-negative")
        if self.component_min_area < 1:
            raise ValueError("component_min_area must be positive")
        if self.component_min_duration <= 0:
            raise ValueError("component_min_duration must be positive")
        if self.component_min_bandwidth_hz <= 0:
            raise ValueError("component_min_bandwidth_hz must be positive")
        if self.max_components_per_event < 1:
            raise ValueError("max_components_per_event must be at least 1")
        if not 0 < self.min_component_energy_ratio <= 1:
            raise ValueError("min_component_energy_ratio must be in (0, 1]")
        if not 0 <= self.min_polyphony_score_for_separation <= 1:
            raise ValueError("min_polyphony_score_for_separation must be in [0, 1]")
        if not 0 <= self.min_component_purity_for_separation <= 1:
            raise ValueError("min_component_purity_for_separation must be in [0, 1]")
        if not 0 <= self.min_purity_for_clustering <= 1:
            raise ValueError("min_purity_for_clustering must be in [0, 1]")
        if not 0 <= self.max_overlap_ratio_for_separation <= 1:
            raise ValueError("max_overlap_ratio_for_separation must be in [0, 1]")
        if not 0 <= self.min_eventness_for_clustering <= 1:
            raise ValueError("min_eventness_for_clustering must be in [0, 1]")
        if not 0 <= self.min_component_eventness_for_clustering <= 1:
            raise ValueError("min_component_eventness_for_clustering must be in [0, 1]")
        if self.eventness_min_contrast_db < 0:
            raise ValueError("eventness_min_contrast_db must be non-negative")
        if not 0 < self.eventness_max_active_ratio <= 1:
            raise ValueError("eventness_max_active_ratio must be in (0, 1]")
        if self.max_components_per_parent < 1:
            raise ValueError("max_components_per_parent must be at least 1")
        if not 0 <= self.candidate_nms_iou <= 1:
            raise ValueError("candidate_nms_iou must be in [0, 1]")
        if self.min_review_event_duration < 0:
            raise ValueError("min_review_event_duration must be non-negative")
        if not 0 <= self.min_clusterability_for_clustering <= 1:
            raise ValueError("min_clusterability_for_clustering must be in [0, 1]")
        if not 0 <= self.min_clusterability_for_review <= 1:
            raise ValueError("min_clusterability_for_review must be in [0, 1]")
        if self.min_clusterability_for_review > self.min_clusterability_for_clustering:
            raise ValueError("min_clusterability_for_review must be <= min_clusterability_for_clustering")
        if not 0 <= self.embedding_stability_min_for_clustering <= 1:
            raise ValueError("embedding_stability_min_for_clustering must be in [0, 1]")
        if not self.embedding_stability_views:
            raise ValueError("embedding_stability_views must not be empty")
        if self.prefamily_min_events < 2:
            raise ValueError("prefamily_min_events must be at least 2")
        if self.cluster_ensemble_runs < 1:
            raise ValueError("cluster_ensemble_runs must be at least 1")
        if not 0 <= self.min_cluster_stability <= 1:
            raise ValueError("min_cluster_stability must be in [0, 1]")
        if self.noise_mode not in {"exploratory", "balanced", "conservative"}:
            raise ValueError("noise_mode must be exploratory, balanced, or conservative")
        if self.noise_min_snr_db < 0:
            raise ValueError("noise_min_snr_db must be non-negative")
        if not 0 <= self.noise_max_flatness <= 1:
            raise ValueError("noise_max_flatness must be in [0, 1]")
        if not 0 <= self.noise_min_tonality <= 1:
            raise ValueError("noise_min_tonality must be in [0, 1]")
        if not 0 <= self.weak_tonality_floor <= 1:
            raise ValueError("weak_tonality_floor must be in [0, 1]")
        if not 0 <= self.min_quality_for_clustering <= 1:
            raise ValueError("min_quality_for_clustering must be in [0, 1]")
        if not 0 <= self.semantic_bio_threshold <= 1:
            raise ValueError("semantic_bio_threshold must be in [0, 1]")
        if not 0 <= self.semantic_noise_threshold <= 1:
            raise ValueError("semantic_noise_threshold must be in [0, 1]")


def config_to_dict(config: BioSoundConfig) -> dict[str, object]:
    """Return a JSON-friendly representation of a runtime config."""
    return asdict(config)


def config_fingerprint(config: BioSoundConfig) -> str:
    """Return a stable fingerprint for cache reuse decisions."""
    payload = json.dumps(config_to_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
