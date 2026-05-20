"""Optional AudioSet-based semantic tagging using PANNs (CNN14).

This provides a per-window probability distribution over 527 AudioSet classes,
collapsed to a biological-activity map that the pipeline can use to:
  • boost detection sensitivity where it expects animal sounds,
  • suppress detection in stretches dominated by Vehicle/Wind/Speech/etc.,
  • add a per-event semantic confidence feature.

PANNs is heavy (a ~340 MB checkpoint). The import lives at the top of functions
that need it so the rest of biosound-cluster keeps working without the dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Curated AudioSet class names that suggest "biological activity worth detecting."
# Anything not in this set contributes to the noise/background score.
BIOLOGICAL_CLASS_NAMES = frozenset({
    "Animal", "Domestic animals, pets", "Wild animals",
    "Bird", "Bird vocalization, bird call, bird song",
    "Chirp, tweet", "Squawk", "Pigeon, dove", "Crow", "Owl", "Hoot",
    "Cluck", "Coo", "Caw",
    "Insect", "Cricket", "Mosquito", "Fly, housefly", "Bee, wasp, etc.",
    "Frog", "Croak",
    "Whale vocalization",
    "Roar", "Growling", "Bark", "Howl", "Yip", "Whimper (dog)",
    "Bow-wow", "Bay",
    "Meow", "Hiss", "Purr",
    "Bleat", "Neigh, whinny",
    "Moo", "Cowbell",
    "Pig", "Oink",
    "Squeak",
    "Chicken, rooster", "Cluck", "Crowing, cock-a-doodle-doo",
    "Turkey", "Gobble",
    "Duck", "Quack",
    "Goose", "Honk",
    "Rodents, rats, mice", "Mouse",
    "Bat",
})

NOISE_CLASS_NAMES = frozenset({
    "Vehicle", "Car", "Truck", "Motorcycle", "Engine", "Aircraft",
    "Wind", "Wind noise (microphone)", "Wind chime",
    "Rain", "Thunder", "Thunderstorm",
    "Speech", "Conversation", "Male speech, man speaking", "Female speech, woman speaking",
    "Music", "Singing", "Musical instrument",
    "Mechanisms", "Machinery", "Industrial",
    "Silence",
    "White noise", "Pink noise",
    "Static",
})


@dataclass(slots=True)
class SemanticTags:
    """Aggregated AudioSet-classifier output over a full recording."""

    sample_rate: int
    n_windows: int
    window_sec: float
    hop_sec: float
    top_global_classes: list[tuple[str, float]]
    biological_score_per_window: np.ndarray
    noise_score_per_window: np.ndarray
    mean_biological_score: float
    mean_noise_score: float
    biological_fraction: float
    dominant_class: str

    @property
    def regime_hint(self) -> str:
        """Suggest an acoustic regime based on top classes."""
        if not self.top_global_classes:
            return "unknown"
        top = self.top_global_classes[0][0]
        if any("Insect" in c or "Mosquito" in c or "Cricket" in c or "Bee" in c for c, _ in self.top_global_classes[:3]):
            return "insect_sustained"
        if any("Whale" in c for c, _ in self.top_global_classes[:5]):
            return "marine_mammal"
        if any("Bird" in c or "Chirp" in c or "Tweet" in c for c, _ in self.top_global_classes[:3]):
            return "bird_calls"
        if any("Frog" in c or "Croak" in c for c, _ in self.top_global_classes[:3]):
            return "amphibian"
        if any("Wind" in c for c, _ in self.top_global_classes[:3]):
            return "wind_dominant"
        return "general_animal" if self.mean_biological_score > self.mean_noise_score else "background_dominant"


def tag_audio(audio: np.ndarray, sr: int, top_k: int = 10) -> SemanticTags:
    """Run PANNs CNN14 over the audio, return aggregated AudioSet tags."""
    try:
        from panns_inference import AudioTagging, labels
    except ImportError as exc:
        raise RuntimeError(
            "panns-inference is required for semantic tagging. Install with: "
            "`pip install panns-inference`."
        ) from exc

    if sr != 32000:
        import librosa
        audio_for_model = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=32000)
        model_sr = 32000
    else:
        audio_for_model = audio.astype(np.float32)
        model_sr = sr

    tagger = AudioTagging(checkpoint_path=None, device="cpu")
    window_samples = model_sr  # 1-second windows
    hop_samples = model_sr // 2  # 0.5 s hop
    n_windows = max(1, 1 + (len(audio_for_model) - window_samples) // hop_samples)

    bio_scores = np.zeros(n_windows, dtype=np.float32)
    noise_scores = np.zeros(n_windows, dtype=np.float32)
    label_list = list(labels)
    bio_idx = [i for i, name in enumerate(label_list) if name in BIOLOGICAL_CLASS_NAMES]
    noise_idx = [i for i, name in enumerate(label_list) if name in NOISE_CLASS_NAMES]
    global_class_acc = np.zeros(len(label_list), dtype=np.float32)

    for w in range(n_windows):
        start = w * hop_samples
        end = start + window_samples
        if end > len(audio_for_model):
            chunk = np.zeros(window_samples, dtype=np.float32)
            chunk[: len(audio_for_model) - start] = audio_for_model[start:]
        else:
            chunk = audio_for_model[start:end]
        clipwise_output, _ = tagger.inference(chunk[np.newaxis, :])
        probs = clipwise_output[0]
        bio_scores[w] = float(probs[bio_idx].max()) if bio_idx else 0.0
        noise_scores[w] = float(probs[noise_idx].max()) if noise_idx else 0.0
        global_class_acc += probs

    global_class_avg = global_class_acc / max(1, n_windows)
    top_idx = np.argsort(global_class_avg)[::-1][:top_k]
    top_classes = [(label_list[i], float(global_class_avg[i])) for i in top_idx]
    mean_bio = float(np.mean(bio_scores))
    mean_noise = float(np.mean(noise_scores))
    bio_fraction = float(np.mean(bio_scores > 0.30))
    dominant_class = top_classes[0][0] if top_classes else "unknown"

    return SemanticTags(
        sample_rate=int(model_sr),
        n_windows=int(n_windows),
        window_sec=float(window_samples / model_sr),
        hop_sec=float(hop_samples / model_sr),
        top_global_classes=top_classes,
        biological_score_per_window=bio_scores,
        noise_score_per_window=noise_scores,
        mean_biological_score=mean_bio,
        mean_noise_score=mean_noise,
        biological_fraction=bio_fraction,
        dominant_class=dominant_class,
    )


def semantic_gate_mask(
    tags: SemanticTags,
    frame_times_sec: np.ndarray,
    bio_threshold: float = 0.20,
    noise_threshold: float = 0.60,
) -> np.ndarray:
    """Build a per-frame boolean mask from the semantic tagging output.

    True where YAMNet/PANNs thinks biological signal is plausible and noise is not dominant.
    The mask aligns to `frame_times_sec` (in seconds, one per audio frame).
    """
    if tags.n_windows == 0 or frame_times_sec.size == 0:
        return np.ones_like(frame_times_sec, dtype=bool)

    window_centers = tags.hop_sec * np.arange(tags.n_windows) + tags.window_sec / 2.0
    bio_at_frame = np.interp(frame_times_sec, window_centers, tags.biological_score_per_window)
    noise_at_frame = np.interp(frame_times_sec, window_centers, tags.noise_score_per_window)
    return (bio_at_frame >= bio_threshold) & (noise_at_frame < noise_threshold)
