"""Perch 2.0 encoder backend (Google bird-vocalization-classifier).

Perch is a TensorFlow SavedModel distributed on Kaggle Hub. It takes 5-second
mono windows at 32 kHz and returns 1280-d embeddings that are SOTA for
bioacoustic transfer.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from biosound_cluster.sota.encoders.base import EncoderInfo

PERCH_SAMPLE_RATE = 32_000
PERCH_WINDOW_SEC = 5.0
PERCH_EMBEDDING_DIM = 1280
PERCH_KAGGLE_HANDLE = "google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier"


class PerchEncoder:
    """Perch 2.0 audio encoder (TensorFlow SavedModel)."""

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 32,
        cache_dir: str | None = None,
    ) -> None:
        self._device = device
        self._batch_size = int(batch_size)
        self._cache_dir = cache_dir
        self._model: Any | None = None
        self._tf: Any | None = None
        self._info = EncoderInfo(
            name="perch",
            sample_rate=PERCH_SAMPLE_RATE,
            window_sec=PERCH_WINDOW_SEC,
            embedding_dim=PERCH_EMBEDDING_DIM,
            requires_gpu=False,
        )

    @property
    def info(self) -> EncoderInfo:
        return self._info

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "The Perch encoder requires tensorflow. "
                "Install with: pip install 'biosound-cluster[sota]'"
            ) from exc
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError(
                "The Perch encoder requires kagglehub. "
                "Install with: pip install 'biosound-cluster[sota]'"
            ) from exc

        if self._device == "cpu":
            tf.config.set_visible_devices([], "GPU")
        if self._cache_dir:
            os.environ.setdefault("KAGGLEHUB_CACHE", self._cache_dir)

        model_path = kagglehub.model_download(PERCH_KAGGLE_HANDLE)
        self._tf = tf
        self._model = tf.saved_model.load(model_path)

    def embed(self, windows: np.ndarray) -> np.ndarray:
        self._load()
        assert self._model is not None and self._tf is not None
        if windows.ndim != 2:
            raise ValueError("windows must have shape (batch, n_samples)")

        expected = int(PERCH_SAMPLE_RATE * PERCH_WINDOW_SEC)
        if windows.shape[1] != expected:
            raise ValueError(
                f"Perch expects windows of length {expected} samples "
                f"(5 s @ 32 kHz), got {windows.shape[1]}"
            )

        out_embeddings: list[np.ndarray] = []
        for start in range(0, windows.shape[0], self._batch_size):
            batch = windows[start : start + self._batch_size]
            tensor = self._tf.convert_to_tensor(batch.astype(np.float32))
            outputs = self._model.infer_tf(tensor)
            if isinstance(outputs, dict):
                if "embedding" in outputs:
                    emb = outputs["embedding"]
                elif "embeddings" in outputs:
                    emb = outputs["embeddings"]
                else:
                    emb = next(iter(outputs.values()))
            elif isinstance(outputs, (list, tuple)):
                emb = outputs[-1]
            else:
                emb = outputs
            emb_np = np.asarray(emb).astype(np.float32)
            if emb_np.ndim == 3:
                emb_np = emb_np.mean(axis=1)
            out_embeddings.append(emb_np)

        embeddings = np.concatenate(out_embeddings, axis=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        return (embeddings / norms).astype(np.float32)
