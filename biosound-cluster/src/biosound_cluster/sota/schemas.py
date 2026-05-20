"""Dataclasses used across the SOTA pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np


@dataclass(slots=True)
class SOTAEvent:
    """An acoustic event derived from contiguous-window clustering."""

    event_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    cluster_id: int
    mean_window_score: float
    n_windows: int
    rms_db: float
    is_noise: bool = False
    is_low_stability: bool = False
    clip_path: str | None = None
    spectrogram_path: str | None = None
    representative_rank: int | None = None
    centroid_distance: float | None = None
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "embedding"
        }


@dataclass(slots=True)
class SOTACluster:
    """Summary metadata for one SOTA cluster."""

    cluster_id: int
    size: int
    folder_name: str
    mean_stability: float
    representative_event_ids: list[str]
    zero_shot_label: str | None = None
    zero_shot_score: float | None = None
    is_noise: bool = False

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["representative_event_ids"] = ",".join(self.representative_event_ids)
        return data


@dataclass(slots=True)
class SOTAResult:
    """Structured return value from process_audio_file_sota."""

    input_path: str
    output_dir: str
    encoder: str
    embedding_dim: int
    duration_sec: float
    sample_rate: int
    n_windows: int
    n_events: int
    n_clusters: int
    n_noise_events: int
    events_csv: str
    clusters_csv: str
    report_md: str
    index_html: str | None
    embeddings_npy: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
