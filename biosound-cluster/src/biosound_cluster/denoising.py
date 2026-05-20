"""Optional bioacoustic denoising hook.

Default: identity (no denoising).
When `--enable-denoiser` is set, calls a registered backend (currently the
EarthSpecies biodenoising diffusion model). The backend interface is a callable
that takes `(audio: np.ndarray, sr: int) -> np.ndarray` and returns the
denoised audio at the SAME sample rate.

The backend imports its model dependency lazily so the rest of biosound-cluster
keeps working when the denoiser isn't installed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

DenoiserCallable = Callable[[np.ndarray, int], np.ndarray]


def identity_denoiser(audio: np.ndarray, sr: int) -> np.ndarray:  # noqa: ARG001 - sr unused
    """No-op denoiser. Returned when denoising is disabled."""
    return audio


class BiodenoisingAdapter:
    """Lazy-loading wrapper around the EarthSpecies biodenoising model.

    Loads the diffusion model on first call, then caches it. The model runs on
    GPU if available, CPU otherwise (CPU is significantly slower).
    """

    def __init__(self, model_name: str = "earthspecies/biodenoising", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._model_sr: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from biodenoising import load_pretrained_denoiser  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "biodenoising not installed. Install with: "
                "`pip install biodenoising` (and torch)."
            ) from exc

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = load_pretrained_denoiser(self.model_name, device=device)
        self._model_sr = getattr(self._model, "sample_rate", 16000)
        self.device = device

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self._load()
        import torch
        import librosa

        target_sr = int(self._model_sr or 16000)
        if sr != target_sr:
            audio_in = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        else:
            audio_in = audio.astype(np.float32)

        with torch.no_grad():
            tensor = torch.from_numpy(audio_in).unsqueeze(0).to(self.device)
            denoised = self._model(tensor)
            if isinstance(denoised, dict):
                denoised = denoised.get("audio", next(iter(denoised.values())))
            denoised_np = denoised.squeeze().cpu().numpy().astype(np.float32)

        if sr != target_sr:
            denoised_np = librosa.resample(denoised_np, orig_sr=target_sr, target_sr=sr)

        if denoised_np.shape != audio.shape:
            new = np.zeros_like(audio, dtype=np.float32)
            n = min(new.size, denoised_np.size)
            new[:n] = denoised_np[:n]
            denoised_np = new

        return denoised_np


def get_denoiser(name: str | None) -> DenoiserCallable:
    """Resolve a denoiser-name string to a callable."""
    if not name or name.lower() in ("none", "identity", "off"):
        return identity_denoiser
    if name.lower() in ("biodenoising", "earthspecies", "diffusion"):
        return BiodenoisingAdapter()
    raise ValueError(f"Unknown denoiser: {name!r}. Known: biodenoising, none.")
