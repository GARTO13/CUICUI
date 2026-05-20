"""Dataclasses used across the biosound-cluster pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np


@dataclass(slots=True)
class AudioEvent:
    """A candidate acoustic event detected in a longer recording."""

    event_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    rms_db: float | None = None
    peak_db: float | None = None
    spectral_centroid: float | None = None
    cluster_id: int | None = None
    cluster_probability: float | None = None
    is_noise: bool = False
    clip_path: str | None = None
    spectrogram_path: str | None = None
    parent_event_id: str | None = None
    component_id: int | None = None
    is_component: bool = False
    is_overlapping: bool = False
    is_mixed: bool = False
    n_components: int = 1
    polyphony_score: float | None = None
    purity_score: float | None = None
    source_type: str = "original"
    context_clip_path: str | None = None
    is_low_confidence_noise: bool = False
    snr_db: float | None = None
    noise_floor_db: float | None = None
    spectral_flatness: float | None = None
    tonality_score: float | None = None
    bandwidth_hz: float | None = None
    peak_band_snr_db: float | None = None
    quality_score: float | None = None
    eventness_score: float | None = None
    temporal_contrast_db: float | None = None
    active_ratio: float | None = None
    is_low_confidence_event: bool = False
    selection_score: float | None = None
    is_pruned_candidate: bool = False
    candidate_route_reason: str | None = None
    is_short_review_event: bool = False
    local_snr_score: float | None = None
    spectral_structure_score: float | None = None
    duration_confidence: float | None = None
    embedding_stability_score: float | None = None
    broadband_noise_penalty: float | None = None
    overlap_penalty: float | None = None
    edge_case_penalty: float | None = None
    clusterability_score: float | None = None
    is_ambiguous_review: bool = False
    acoustic_prefamily: str | None = None
    cluster_stability_score: float | None = None
    representative_score: float | None = None
    separated_audio: np.ndarray | None = field(default=None, repr=False, compare=False)
    context_audio: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/CSV-friendly dictionary."""
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in {"separated_audio", "context_audio"}
        }


@dataclass(slots=True)
class ClusterSummary:
    """Summary metadata for one acoustic cluster."""

    cluster_id: int
    size: int
    folder_name: str
    mean_probability: float | None
    representative_event_ids: list[str]
    mean_purity_score: float | None = None
    n_component_events: int = 0
    n_original_events: int = 0
    mean_clusterability_score: float | None = None
    mean_stability_score: float | None = None
    acoustic_prefamily: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/CSV-friendly dictionary."""
        data = asdict(self)
        data["representative_event_ids"] = ",".join(self.representative_event_ids)
        return data


@dataclass(slots=True)
class ProcessResult:
    """Structured return value from process_audio_file."""

    input_path: str
    output_dir: str
    duration_sec: float
    sample_rate: int
    n_events: int
    n_clusters: int
    n_noise: int
    events_csv: str
    clusters_csv: str
    report_md: str
    index_html: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary."""
        return asdict(self)
