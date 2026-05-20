"""Mock encoder used for tests and dependency-free smoke runs.

This encoder produces semantically meaningful (but cheap) embeddings from a
log-mel spectrogram so similar acoustic patterns end up nearby in feature
space. It does *not* require torch/TF/Perch weights, so it lets us unit-test
the rest of the pipeline.
"""

from __future__ import annotations

import numpy as np

from biosound_cluster.sota.encoders.base import EncoderInfo

_N_MELS = 64
_N_FFT = 1024
_HOP_LENGTH = 256
_EMBEDDING_DIM = 256


class MockEncoder:
    """Cheap mel-statistics encoder useful for tests and smoke runs."""

    def __init__(self, sample_rate: int = 32_000, window_sec: float = 5.0) -> None:
        self._info = EncoderInfo(
            name="mock",
            sample_rate=int(sample_rate),
            window_sec=float(window_sec),
            embedding_dim=_EMBEDDING_DIM,
            requires_gpu=False,
        )
        self._projection = self._build_projection()

    @property
    def info(self) -> EncoderInfo:
        return self._info

    def _build_projection(self) -> np.ndarray:
        rng = np.random.default_rng(seed=1234)
        feat_dim = _N_MELS * 4
        proj = rng.standard_normal(size=(feat_dim, _EMBEDDING_DIM)).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-9
        return proj

    def embed(self, windows: np.ndarray) -> np.ndarray:
        import librosa

        if windows.ndim != 2:
            raise ValueError("windows must have shape (batch, n_samples)")
        sr = self._info.sample_rate
        features = np.empty((windows.shape[0], _N_MELS * 4), dtype=np.float32)
        for i, window in enumerate(windows):
            mel = librosa.feature.melspectrogram(
                y=window.astype(np.float32),
                sr=sr,
                n_fft=_N_FFT,
                hop_length=_HOP_LENGTH,
                n_mels=_N_MELS,
                power=2.0,
            )
            log_mel = librosa.power_to_db(mel + 1e-10, ref=1.0).astype(np.float32)
            features[i, :_N_MELS] = log_mel.mean(axis=1)
            features[i, _N_MELS : 2 * _N_MELS] = log_mel.std(axis=1)
            features[i, 2 * _N_MELS : 3 * _N_MELS] = log_mel.max(axis=1)
            features[i, 3 * _N_MELS : 4 * _N_MELS] = log_mel.min(axis=1)
        features -= features.mean(axis=1, keepdims=True)
        scale = features.std(axis=1, keepdims=True) + 1e-6
        features /= scale
        embeddings = features @ self._projection
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        return (embeddings / norms).astype(np.float32)
