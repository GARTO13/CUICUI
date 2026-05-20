"""Resolve an encoder name to a concrete backend instance."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biosound_cluster.sota.config import SOTAConfig
    from biosound_cluster.sota.encoders.base import AudioEncoder


def load_encoder(config: "SOTAConfig") -> "AudioEncoder":
    """Lazily import and instantiate the requested encoder backend."""
    name = config.encoder.lower()
    if name == "perch":
        from biosound_cluster.sota.encoders.perch import PerchEncoder

        return PerchEncoder(
            device=config.encoder_device,
            batch_size=config.encoder_batch_size,
            cache_dir=config.encoder_cache_dir,
        )
    if name == "aves":
        from biosound_cluster.sota.encoders.aves import AVESEncoder

        return AVESEncoder(
            device=config.encoder_device,
            batch_size=config.encoder_batch_size,
            cache_dir=config.encoder_cache_dir,
        )
    if name == "birdnet":
        from biosound_cluster.sota.encoders.birdnet import BirdNETEncoder

        return BirdNETEncoder(
            device=config.encoder_device,
            batch_size=config.encoder_batch_size,
            cache_dir=config.encoder_cache_dir,
        )
    if name == "mock":
        from biosound_cluster.sota.encoders.mock import MockEncoder

        return MockEncoder(
            sample_rate=config.sample_rate,
            window_sec=config.window_sec or 5.0,
        )
    raise ValueError(
        f"Unknown encoder '{config.encoder}'. "
        "Available: 'perch', 'aves', 'birdnet', 'mock'."
    )
