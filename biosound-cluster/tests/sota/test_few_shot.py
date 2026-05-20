from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from biosound_cluster.sota.few_shot import (
    compute_prototypes,
    load_labels_from_json,
    reassign,
)


def test_compute_prototypes_and_reassign(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    bird_center = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    frog_center = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    embeddings = np.stack(
        [bird_center + 0.05 * rng.standard_normal(3) for _ in range(4)]
        + [frog_center + 0.05 * rng.standard_normal(3) for _ in range(4)],
        axis=0,
    ).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    event_ids = [f"e{i}" for i in range(8)]
    labels = {"e0": "bird", "e1": "bird", "e4": "frog", "e5": "frog"}
    label_names, prototypes = compute_prototypes(embeddings, event_ids, labels)
    assert label_names == ["bird", "frog"]

    results = reassign(embeddings, label_names, prototypes, min_confidence=0.5)
    new_labels = [r.new_label for r in results]
    assert new_labels[:4] == ["bird"] * 4
    assert new_labels[4:] == ["frog"] * 4


def test_load_labels_from_json(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"a": "x", "b": "y"}))
    labels = load_labels_from_json(path)
    assert labels == {"a": "x", "b": "y"}
