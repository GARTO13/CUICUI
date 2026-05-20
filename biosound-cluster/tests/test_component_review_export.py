from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.export import export_outputs
from biosound_cluster.schemas import AudioEvent


def test_component_review_folder_manifest_and_events_csv(tmp_path: Path) -> None:
    sr = 8000
    audio = np.zeros(sr, dtype=np.float32)
    event = AudioEvent(
        event_id="parent_component_0",
        parent_event_id="parent",
        component_id=0,
        start_sec=0.1,
        end_sec=0.3,
        duration_sec=0.2,
        is_component=True,
        source_type="component_review",
        is_component_review=True,
        clusterable=False,
        routing_reason="component_quality_below_threshold",
        separated_audio=np.zeros(int(sr * 0.2), dtype=np.float32),
    )

    export_outputs(
        audio,
        sr,
        tmp_path / "input.wav",
        tmp_path / "out",
        events=[],
        clusters=[],
        config=BioSoundConfig(sample_rate=sr, min_cluster_size=2, export_clips=False, generate_spectrograms=False),
        duration_sec=1.0,
        component_review_events=[event],
    )

    folder = tmp_path / "out" / "component_review_size_001"
    assert folder.exists()
    assert (folder / "_cluster_manifest.csv").exists()
    events = pd.read_csv(tmp_path / "out" / "events.csv")
    assert "is_component_review" in events.columns
    assert "routing_reason" in events.columns
    assert events.loc[0, "routing_reason"] == "component_quality_below_threshold"
