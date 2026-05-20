"""BirdNET encoder backend (Kahl et al., Cornell Lab).

Uses the `birdnet` PyPI package which wraps the TFLite model and exposes the
embedding head.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from biosound_cluster.sota.encoders.base import EncoderInfo

BIRDNET_SAMPLE_RATE = 48_000
BIRDNET_WINDOW_SEC = 3.0
BIRDNET_EMBEDDING_DIM = 1024


class BirdNETEncoder:
    """BirdNET embedding head (1024-d, 3 s @ 48 kHz)."""

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 16,
        cache_dir: str | None = None,
    ) -> None:
        self._device = device
        self._batch_size = int(batch_size)
        self._cache_dir = cache_dir
        self._analyzer: Any | None = None
        self._info = EncoderInfo(
            name="birdnet",
            sample_rate=BIRDNET_SAMPLE_RATE,
            window_sec=BIRDNET_WINDOW_SEC,
            embedding_dim=BIRDNET_EMBEDDING_DIM,
            requires_gpu=False,
        )

    @property
    def info(self) -> EncoderInfo:
        return self._info

    def _load(self) -> None:
        if self._analyzer is not None:
            return
        try:
            from birdnet_analyzer.embeddings.core import embeddings as _  # noqa: F401
            from birdnet_analyzer.model import get_embeddings
        except ImportError as exc:
            raise ImportError(
                "The BirdNET encoder requires the 'birdnet-analyzer' package. "
                "Install with: pip install 'biosound-cluster[sota]'"
            ) from exc

        self._embed_fn = get_embeddings

    def embed(self, windows: np.ndarray) -> np.ndarray:
        self._load()
        if windows.ndim != 2:
            raise ValueError("windows must have shape (batch, n_samples)")
        expected = int(BIRDNET_SAMPLE_RATE * BIRDNET_WINDOW_SEC)
        if windows.shape[1] != expected:
            raise ValueError(
                f"BirdNET expects windows of length {expected} samples "
                f"(3 s @ 48 kHz), got {windows.shape[1]}"
            )

        out = []
        for start in range(0, windows.shape[0], self._batch_size):
            batch = windows[start : start + self._batch_size]
            embeddings = self._embed_fn(batch.astype(np.float32))
            arr = np.asarray(embeddings, dtype=np.float32)
            out.append(arr.reshape(arr.shape[0], -1))
        embeddings = np.concatenate(out, axis=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        return (embeddings / norms).astype(np.float32)
