"""Map an AudioProfile (+ optional SemanticTags) to a BioSoundConfig.

The profile captures low-level acoustics; the semantic tags add a "what does
AudioSet think this is" prior. Rules below combine both to choose detection
parameters that suit the recording's regime, instead of hard-coding values
tuned to one subset.
"""

from __future__ import annotations

from dataclasses import replace

from biosound_cluster.audio_profile import AudioProfile
from biosound_cluster.config import BioSoundConfig
from biosound_cluster.semantic_tagging import SemanticTags


def adaptive_config_from_profile(
    profile: AudioProfile,
    base: BioSoundConfig | None = None,
    tags: SemanticTags | None = None,
) -> BioSoundConfig:
    """Produce a BioSoundConfig tuned to the recording's profile + semantic tags."""
    cfg = base if base is not None else BioSoundConfig()
    overrides: dict[str, object] = {}

    overrides.update(_threshold_from_dynamic_range(profile))
    overrides.update(_event_duration_from_regime(profile))
    overrides.update(_segmentation_votes_from_regime(profile))
    overrides.update(_tonality_floor_from_profile(profile))
    overrides.update(_quality_gate_from_profile(profile))

    if tags is not None:
        overrides.update(_overrides_from_semantic_tags(tags))

    if not overrides:
        return cfg
    return replace(cfg, **overrides)


def _threshold_from_dynamic_range(profile: AudioProfile) -> dict[str, object]:
    """Lower threshold_db on quiet recordings so detection doesn't starve.

    Songbird and faint-mammal recordings often have <20 dB dynamic range; the
    default 8 dB above-floor threshold is too high there.
    """
    dr = profile.dynamic_range_db
    if dr < 12.0:
        return {
            "threshold_db": 4.0,
            "flux_percentile": 82.0,
            "flux_min_snr_db": 0.5,
            "multiband_min_snr_db": 3.0,
            "multiband_min_flux_z": 1.0,
        }
    if dr < 18.0:
        return {"threshold_db": 3.5}
    if dr < 25.0:
        return {"threshold_db": 5.0}
    return {"threshold_db": 8.0}


def _event_duration_from_regime(profile: AudioProfile) -> dict[str, object]:
    """Set min/max event duration based on whether the file looks transient or sustained."""
    regime = profile.regime
    if regime == "sustained_narrowband":
        # Insect-like: allow long buzzes, drop the short-floor (already part of min duration).
        return {
            "min_event_duration": 0.5,
            "max_event_duration": 30.0,
            "min_review_event_duration": 0.40,
            "merge_gap": 0.6,
        }
    if regime == "transient_dense":
        # Whale-click / fast-bird-like: many short events, tight merging.
        return {
            "min_event_duration": 0.08,
            "max_event_duration": 2.0,
            "min_review_event_duration": 0.06,
            "merge_gap": 0.1,
        }
    if regime == "dense_variable":
        # RD-like: medium-long calls, broader merging.
        return {
            "min_event_duration": 0.25,
            "max_event_duration": 8.0,
            "min_review_event_duration": 0.20,
            "merge_gap": 0.4,
        }
    return {}


def _segmentation_votes_from_regime(profile: AudioProfile) -> dict[str, object]:
    """For sustained narrowband signals, the flux-onset vote misfires; relax it."""
    if profile.regime == "sustained_narrowband":
        return {
            "enable_flux_detection": False,
            "enable_spectral_concentration_gate": True,
            "segmentation_min_active_votes": 1,
        }
    if profile.regime == "low_dynamic_range":
        # Quiet/faint recordings need sensitive segmentation, but we keep review routing
        # downstream so extra candidates do not automatically pollute normal clusters.
        return {"segmentation_min_active_votes": 1}
    return {}


def _tonality_floor_from_profile(profile: AudioProfile) -> dict[str, object]:
    """If the recording's median tonality is low, the tonality gate would over-filter."""
    if profile.median_tonality < 0.10:
        return {"weak_tonality_floor": 0.05}
    if profile.median_tonality < 0.20:
        return {"weak_tonality_floor": 0.10}
    return {}


def _quality_gate_from_profile(profile: AudioProfile) -> dict[str, object]:
    """If the recording is generally quiet, relax the quality gate."""
    if profile.dynamic_range_db < 12.0:
        return {
            "noise_mode": "exploratory",
            "min_quality_for_clustering": 0.25,
            "min_eventness_for_clustering": 0.10,
            "min_clusterability_for_clustering": 0.30,
            "min_clusterability_for_review": 0.10,
            "enable_short_event_review": False,
        }
    if profile.dynamic_range_db < 18.0:
        return {"min_quality_for_clustering": 0.40}
    return {}


def _overrides_from_semantic_tags(tags: SemanticTags) -> dict[str, object]:
    """Semantic-tag-based overrides. AudioSet hints refine the profile-only choices."""
    hint = tags.regime_hint
    if hint == "insect_sustained":
        return {
            "enable_flux_detection": False,
            "min_event_duration": 0.5,
            "max_event_duration": 60.0,
            "segmentation_min_active_votes": 1,
            "weak_tonality_floor": 0.05,
        }
    if hint == "marine_mammal":
        return {
            "min_event_duration": 0.05,
            "max_event_duration": 2.0,
            "merge_gap": 0.08,
            "weak_tonality_floor": 0.05,
            "min_quality_for_clustering": 0.40,
        }
    if hint == "bird_calls":
        return {
            "threshold_db": 3.0,
            "min_event_duration": 0.10,
            "max_event_duration": 3.0,
            "min_quality_for_clustering": 0.45,
        }
    if hint == "wind_dominant":
        # Wind makes detection unreliable; tighten quality so we route to noise more.
        return {"min_quality_for_clustering": 0.65}
    return {}


def describe_overrides(profile: AudioProfile, tags: SemanticTags | None) -> str:
    """Human-readable summary of what would be overridden — for logging in the pipeline."""
    cfg_default = BioSoundConfig()
    cfg_adaptive = adaptive_config_from_profile(profile, base=cfg_default, tags=tags)
    lines: list[str] = []
    for field_name in (
        "threshold_db",
        "flux_percentile",
        "flux_min_snr_db",
        "multiband_min_snr_db",
        "multiband_min_flux_z",
        "min_event_duration",
        "max_event_duration",
        "min_review_event_duration",
        "merge_gap",
        "enable_flux_detection",
        "enable_spectral_concentration_gate",
        "segmentation_min_active_votes",
        "noise_mode",
        "weak_tonality_floor",
        "min_quality_for_clustering",
        "min_eventness_for_clustering",
        "min_clusterability_for_clustering",
        "min_clusterability_for_review",
        "enable_short_event_review",
    ):
        before = getattr(cfg_default, field_name)
        after = getattr(cfg_adaptive, field_name)
        if before != after:
            lines.append(f"  {field_name}: {before} -> {after}")
    regime = profile.regime
    hint = tags.regime_hint if tags is not None else "no semantic tags"
    header = f"Adaptive overrides (profile regime={regime}, semantic hint={hint}):"
    if not lines:
        return f"{header}\n  (no changes from default)"
    return header + "\n" + "\n".join(lines)
