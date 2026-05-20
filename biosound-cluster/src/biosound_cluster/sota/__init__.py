"""SOTA pipeline for bioacoustic event discovery and clustering.

This subpackage implements a state-of-the-art pipeline based on:

- pretrained bioacoustic encoders (Perch 2.0, AVES, BirdNET),
- sliding-window embedding,
- k-NN graph + Leiden community detection,
- event extraction from contiguous cluster runs,
- zero-shot acoustic-family captioning via BioLingual,
- few-shot prototype refinement from a small set of human labels.

The original `biosound_cluster.pipeline` module is preserved unchanged.
"""

from biosound_cluster.sota.config import SOTAConfig
from biosound_cluster.sota.schemas import SOTAEvent, SOTACluster, SOTAResult

__all__ = [
    "SOTAConfig",
    "SOTAEvent",
    "SOTACluster",
    "SOTAResult",
    "process_audio_file_sota",
]


def __getattr__(name: str):
    if name == "process_audio_file_sota":
        from biosound_cluster.sota.pipeline import process_audio_file_sota

        return process_audio_file_sota
    raise AttributeError(f"module 'biosound_cluster.sota' has no attribute {name!r}")
