"""Few-shot prototype refinement.

Given a small set of human-labeled events (cluster_id -> label_name), this
module computes a prototype embedding per label and re-assigns every event by
cosine similarity to those prototypes. Standard prototypical-network setup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class FewShotResult:
    """Per-event re-assignment after prototype matching."""

    new_label: str | None
    similarity: float


def load_labels_from_json(path: str | Path) -> dict[str, str]:
    """Load a ``{event_id: label}`` mapping from a JSON file."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Few-shot labels file must be a JSON object {event_id: label}")
    return {str(k): str(v) for k, v in data.items()}


def compute_prototypes(
    embeddings: np.ndarray,
    event_ids: list[str],
    labels: dict[str, str],
) -> tuple[list[str], np.ndarray]:
    """Compute a unit-norm prototype per label by averaging labeled embeddings."""
    if embeddings.shape[0] != len(event_ids):
        raise ValueError("embeddings and event_ids must have the same length")

    label_to_indices: dict[str, list[int]] = {}
    for idx, event_id in enumerate(event_ids):
        if event_id in labels:
            label_to_indices.setdefault(labels[event_id], []).append(idx)

    if not label_to_indices:
        return [], np.zeros((0, embeddings.shape[1]), dtype=np.float32)

    label_names = sorted(label_to_indices.keys())
    proto_matrix = np.stack(
        [embeddings[idxs].mean(axis=0) for idxs in (label_to_indices[name] for name in label_names)],
        axis=0,
    )
    proto_matrix /= np.linalg.norm(proto_matrix, axis=1, keepdims=True) + 1e-9
    return label_names, proto_matrix.astype(np.float32)


def reassign(
    embeddings: np.ndarray,
    label_names: list[str],
    prototypes: np.ndarray,
    min_confidence: float,
) -> list[FewShotResult]:
    """Assign each row to its nearest prototype above ``min_confidence``."""
    if embeddings.size == 0 or prototypes.size == 0:
        return [FewShotResult(new_label=None, similarity=0.0) for _ in range(embeddings.shape[0])]

    norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    sims = norms @ prototypes.T
    best = sims.argmax(axis=1)
    best_scores = sims[np.arange(sims.shape[0]), best]
    results: list[FewShotResult] = []
    for idx, score in zip(best, best_scores, strict=False):
        if score < min_confidence:
            results.append(FewShotResult(new_label=None, similarity=float(score)))
        else:
            results.append(FewShotResult(new_label=label_names[int(idx)], similarity=float(score)))
    return results
