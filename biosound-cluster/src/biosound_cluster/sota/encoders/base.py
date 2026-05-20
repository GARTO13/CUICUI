"""Encoder abstraction shared by all backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(slots=True, frozen=True)
class EncoderInfo:
    """Static description of an encoder backend."""

    name: str
    sample_rate: int
    window_sec: float
    embedding_dim: int
    requires_gpu: bool = False


class AudioEncoder(Protocol):
    """Protocol every encoder backend implements."""

    @property
    def info(self) -> EncoderInfo: ...

    def embed(self, windows: np.ndarray) -> np.ndarray:
        """Embed a batch of mono windows.

        Args:
            windows: float32 array of shape ``(batch, n_samples)`` where
                ``n_samples == int(info.window_sec * info.sample_rate)``.

        Returns:
            float32 array of shape ``(batch, info.embedding_dim)``.
        """
        ...
