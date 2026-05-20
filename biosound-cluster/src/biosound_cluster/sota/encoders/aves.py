"""AVES encoder backend (Hagiwara 2023, self-supervised HuBERT for animal sounds)."""

from __future__ import annotations

from typing import Any

import numpy as np

from biosound_cluster.sota.encoders.base import EncoderInfo

AVES_SAMPLE_RATE = 16_000
AVES_WINDOW_SEC = 5.0
AVES_EMBEDDING_DIM = 768


class AVESEncoder:
    """AVES base encoder (~95M params). Loaded via torch.hub."""

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 16,
        cache_dir: str | None = None,
    ) -> None:
        self._device_pref = device
        self._batch_size = int(batch_size)
        self._cache_dir = cache_dir
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._info = EncoderInfo(
            name="aves",
            sample_rate=AVES_SAMPLE_RATE,
            window_sec=AVES_WINDOW_SEC,
            embedding_dim=AVES_EMBEDDING_DIM,
            requires_gpu=False,
        )

    @property
    def info(self) -> EncoderInfo:
        return self._info

    def _pick_device(self, torch: Any) -> Any:
        if self._device_pref == "cpu":
            return torch.device("cpu")
        if self._device_pref == "cuda":
            return torch.device("cuda")
        if self._device_pref == "mps":
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "The AVES encoder requires torch. "
                "Install with: pip install 'biosound-cluster[sota]'"
            ) from exc

        self._torch = torch
        self._device = self._pick_device(torch)
        if self._cache_dir:
            torch.hub.set_dir(self._cache_dir)

        model = torch.hub.load(
            "earthspecies/aves",
            "aves_base_all",
            trust_repo=True,
        )
        model.to(self._device)
        model.eval()
        self._model = model

    def embed(self, windows: np.ndarray) -> np.ndarray:
        self._load()
        assert self._model is not None and self._torch is not None
        torch = self._torch
        if windows.ndim != 2:
            raise ValueError("windows must have shape (batch, n_samples)")

        out_embeddings: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, windows.shape[0], self._batch_size):
                batch = windows[start : start + self._batch_size]
                tensor = torch.from_numpy(batch.astype(np.float32)).to(self._device)
                features = self._model.extract_features(tensor)
                if isinstance(features, (list, tuple)):
                    hidden = features[0]
                elif isinstance(features, dict):
                    hidden = features.get("last_hidden_state", next(iter(features.values())))
                else:
                    hidden = features
                if hidden.dim() == 3:
                    pooled = hidden.mean(dim=1)
                else:
                    pooled = hidden
                out_embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))

        embeddings = np.concatenate(out_embeddings, axis=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        return (embeddings / norms).astype(np.float32)
