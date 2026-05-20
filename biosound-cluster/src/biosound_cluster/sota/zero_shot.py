"""Zero-shot acoustic-family captioning of clusters via BioLingual / CLAP.

This module gives each cluster a textual label by:

1. Picking a few representative audio clips per cluster.
2. Running them through a CLAP-style joint audio-text encoder (BioLingual).
3. Computing cosine similarity between the cluster's audio embedding mean
   and each prompt embedding.
4. Assigning the best-matching prompt as the cluster caption.

BioLingual is *not* the main encoder of the pipeline — it lives in its own
embedding space. We use it only here, on top of the audio clips, to provide
a human-readable description without altering the clustering signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ZeroShotResult:
    """Zero-shot caption assigned to a cluster."""

    cluster_id: int
    label: str
    score: float


class _BioLingualClient:
    """Lazy wrapper around davidrrobinson/BioLingual (CLAP for bioacoustics)."""

    HF_REPO = "davidrrobinson/BioLingual"
    SAMPLE_RATE = 48_000

    def __init__(self, device: str = "auto") -> None:
        self._device_pref = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise ImportError(
                "Zero-shot captioning requires torch and transformers. "
                "Install with: pip install 'biosound-cluster[sota]'"
            ) from exc

        if self._device_pref == "cpu":
            device = torch.device("cpu")
        elif self._device_pref == "cuda":
            device = torch.device("cuda")
        elif self._device_pref == "mps":
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self._processor = ClapProcessor.from_pretrained(self.HF_REPO)
        self._model = ClapModel.from_pretrained(self.HF_REPO).to(device).eval()
        self._torch = torch
        self._device = device

    def text_embeddings(self, prompts: list[str]) -> np.ndarray:
        self._load()
        assert self._processor is not None and self._model is not None
        torch = self._torch
        inputs = self._processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            features = self._model.get_text_features(**inputs)
        arr = features.detach().cpu().numpy().astype(np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return arr

    def audio_embeddings(self, clips: list[np.ndarray]) -> np.ndarray:
        self._load()
        assert self._processor is not None and self._model is not None
        torch = self._torch
        inputs = self._processor(
            audios=clips,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            features = self._model.get_audio_features(**inputs)
        arr = features.detach().cpu().numpy().astype(np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return arr


def caption_clusters(
    cluster_clips: dict[int, list[np.ndarray]],
    cluster_source_rates: dict[int, int],
    prompts: list[str],
    device: str = "auto",
) -> list[ZeroShotResult]:
    """Assign one prompt per cluster using BioLingual cosine similarity.

    Args:
        cluster_clips: mapping cluster_id -> list of audio clip arrays.
        cluster_source_rates: per-cluster source sample rate of the clips.
        prompts: list of candidate text descriptions.
        device: torch device preference.
    """
    if not cluster_clips or not prompts:
        return []
    import librosa

    client = _BioLingualClient(device=device)
    text_emb = client.text_embeddings(prompts)

    results: list[ZeroShotResult] = []
    for cluster_id, clips in cluster_clips.items():
        if not clips:
            continue
        sr_source = cluster_source_rates.get(cluster_id, 32_000)
        resampled = []
        for clip in clips:
            if sr_source != _BioLingualClient.SAMPLE_RATE:
                clip = librosa.resample(
                    clip.astype(np.float32, copy=False),
                    orig_sr=sr_source,
                    target_sr=_BioLingualClient.SAMPLE_RATE,
                )
            resampled.append(clip)
        audio_emb = client.audio_embeddings(resampled)
        mean_audio = audio_emb.mean(axis=0, keepdims=True)
        mean_audio /= np.linalg.norm(mean_audio, axis=1, keepdims=True) + 1e-9
        sims = (mean_audio @ text_emb.T)[0]
        best = int(np.argmax(sims))
        results.append(
            ZeroShotResult(
                cluster_id=int(cluster_id),
                label=str(prompts[best]),
                score=float(sims[best]),
            )
        )
    return results
