"""Configuration for the SOTA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


DEFAULT_ZERO_SHOT_PROMPTS: tuple[str, ...] = (
    "a bird call",
    "a bird song",
    "a frog or amphibian call",
    "an insect call or stridulation",
    "a mammal vocalization",
    "a bat echolocation call",
    "wind or weather noise",
    "rain",
    "human speech",
    "engine or vehicle noise",
    "water flowing",
    "silence or background hiss",
    "anthropogenic noise",
)


@dataclass(slots=True)
class SOTAConfig:
    """All knobs for the SOTA pipeline.

    Defaults target the highest quality regardless of compute budget.
    """

    # --- Audio ---
    sample_rate: int = 32_000

    # --- Encoder ---
    encoder: Literal["perch", "aves", "birdnet", "mock"] = "perch"
    encoder_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    encoder_batch_size: int = 32
    encoder_cache_dir: str | None = None

    # --- Windowing ---
    window_sec: float | None = None
    hop_sec: float = 1.0
    silence_rms_db: float = -55.0

    # --- Clustering (k-NN graph + Leiden) ---
    knn_neighbors: int = 15
    knn_metric: Literal["cosine", "euclidean"] = "cosine"
    leiden_resolution: float = 1.0
    leiden_n_iterations: int = 10
    leiden_seed: int = 42
    min_cluster_size: int = 5
    stability_subsamples: int = 5
    stability_subsample_fraction: float = 0.8
    min_stability_for_keep: float = 0.0

    # --- Event extraction ---
    min_event_duration: float = 0.10
    max_event_duration: float = 30.0
    max_event_gap: float = 1.0
    event_nms_iou: float = 0.5
    refine_onset_offset: bool = True
    refinement_activity_db: float = 8.0
    refinement_padding: float = 0.05
    refinement_smoothing_sec: float = 0.02

    # --- Zero-shot captioning ---
    enable_zero_shot: bool = True
    zero_shot_model: Literal["biolingual", "naturelm"] = "biolingual"
    zero_shot_prompts: tuple[str, ...] = field(default_factory=lambda: DEFAULT_ZERO_SHOT_PROMPTS)
    zero_shot_clips_per_cluster: int = 4

    # --- Few-shot refinement ---
    enable_few_shot: bool = False
    few_shot_labels_path: str | None = None
    few_shot_min_confidence: float = 0.45

    # --- Export ---
    export_clips: bool = True
    export_spectrograms: bool = True
    export_embeddings: bool = True
    export_index_html: bool = True
    representatives_per_cluster: int = 16
    cluster_folder_prefix: str = "cluster"
    noise_folder_name: str = "noise_unknown"
    low_stability_folder_name: str = "low_stability_review"
