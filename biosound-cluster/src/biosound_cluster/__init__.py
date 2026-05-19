"""Unsupervised acoustic clustering for human-in-the-loop bioacoustic review."""

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.schemas import AudioEvent, ClusterSummary, ProcessResult

__all__ = [
    "AudioEvent",
    "BioSoundConfig",
    "ClusterSummary",
    "ProcessResult",
    "process_audio_file",
]


def __getattr__(name: str):
    if name == "process_audio_file":
        from biosound_cluster.pipeline import process_audio_file

        return process_audio_file
    raise AttributeError(f"module 'biosound_cluster' has no attribute {name!r}")
