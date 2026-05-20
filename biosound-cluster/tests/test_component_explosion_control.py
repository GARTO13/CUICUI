from __future__ import annotations

import numpy as np

from biosound_cluster.config import BioSoundConfig
from biosound_cluster.polyphony import TFComponent, should_component_enter_clustering
from biosound_cluster.schemas import AudioEvent
from biosound_cluster.pipeline import _split_component_review


def _component(idx: int, quality: float, purity: float = 0.9) -> TFComponent:
    return TFComponent(
        component_id=idx,
        time_start_frame=idx,
        time_end_frame=idx + 3,
        freq_start_bin=1,
        freq_end_bin=5,
        area=20,
        energy=1.0,
        energy_ratio=0.2,
        duration_sec=0.2,
        bandwidth_hz=500.0,
        mask=np.ones((6, 6), dtype=bool),
        purity_score=purity,
        compactness=0.5,
        snr_db=12.0,
        quality_score=quality,
    )


def test_component_limit_routes_extra_components_to_review() -> None:
    config = BioSoundConfig(max_components_for_clustering_per_parent=2)
    decisions = [should_component_enter_clustering(_component(idx, 0.9), idx, config)[0] for idx in range(4)]
    assert decisions.count(True) == 2
    assert decisions.count(False) == 2


def test_weak_components_are_component_review() -> None:
    config = BioSoundConfig(min_component_quality_for_clustering=0.62)
    ok, reason = should_component_enter_clustering(_component(0, 0.2), 0, config)
    assert not ok
    assert reason == "component_quality_below_threshold"

    events = [
        AudioEvent("c0", 0.0, 0.2, 0.2, is_component=True, source_type="component", clusterable=True),
        AudioEvent("c1", 0.2, 0.4, 0.2, is_component=True, source_type="component_review", is_component_review=True, clusterable=False),
    ]
    clusterable, review = _split_component_review(events)
    assert len(clusterable) == 1
    assert len(review) == 1
