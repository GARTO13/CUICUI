# biosound-cluster SOTA pipeline

This subpackage (`biosound_cluster.sota`) is a state-of-the-art rewrite of the
original biosound-cluster pipeline. It coexists with the legacy pipeline
(`biosound_cluster.pipeline`) — neither replaces the other.

The motivation is documented in the project handoff: the legacy pipeline uses
handcrafted log-mel features + RMS/flux segmentation + UMAP/HDBSCAN and plateaus
around precision 0.11 / recall 0.56 on DCASE 2024 Task 5 (ME, 2 files). The
SOTA pipeline replaces the brain (the embedding space) with a pretrained
bioacoustic encoder and drops several modules (segmentation, eventness,
candidate selection, polyphony, short-event routing) by deriving events from
window-level clustering instead.

---

## What this pipeline does

```
audio long                                      .wav (any length)
   │
   ▼
sliding windows  ◄── encoder-native window/hop (5 s / 1 s by default)
   │
   ▼
silence gating   ◄── per-window RMS dB threshold
   │
   ▼
pretrained encoder   ◄── Perch 2.0 / AVES / BirdNET / mock
   │   (1280-d / 768-d / 1024-d / 256-d embeddings, L2-normalized)
   ▼
k-NN cosine graph + Leiden community detection
   │   (with stability scoring via subsampling)
   ▼
contiguous-run grouping → events
   │
   ▼
temporal NMS + onset/offset refinement on energy envelope
   │
   ▼
representative selection per cluster (centroid distance)
   │
   ▼
[optional] zero-shot caption via BioLingual (CLAP)
   │
   ▼
[optional] few-shot prototype refinement from human labels
   │
   ▼
export: wav clips + spectrograms + CSV + report.md + index.html + embeddings.npz
```

The product output (folders of wav clips per acoustic family, one folder per
cluster, ready for human review) is preserved — only the engine that decides
"this clip and that clip are the same kind of sound" changes.

---

## File map

```
biosound_cluster/sota/
├── __init__.py          public exports: SOTAConfig, process_audio_file_sota, schemas
├── config.py            SOTAConfig dataclass — every knob lives here
├── schemas.py           SOTAEvent, SOTACluster, SOTAResult
├── windowing.py         sliding-window extraction + silence gating
├── graph_clustering.py  k-NN graph + Leiden + stability via subsampling
├── event_extraction.py  runs → events, temporal NMS, boundary refinement
├── zero_shot.py         BioLingual CLAP captioning of clusters
├── few_shot.py          prototypical-network reassignment from human labels
├── export.py            wav clips, spectrograms, CSV, report.md, index.html
├── pipeline.py          orchestrator: process_audio_file_sota
├── cli.py               typer CLI: biosound-cluster-sota
└── encoders/
    ├── base.py          AudioEncoder protocol + EncoderInfo dataclass
    ├── factory.py       load_encoder(config) → backend instance
    ├── perch.py         Perch 2.0 (Google, TF SavedModel via Kaggle Hub)
    ├── aves.py          AVES (Earth Species Project, torch.hub)
    ├── birdnet.py       BirdNET (Cornell, birdnet-analyzer)
    └── mock.py          deterministic mel-statistics encoder (tests, no GPU)
```

---

## Installation

The base package keeps the legacy pipeline working without any new heavy
dependency. SOTA features live behind an extras install:

```bash
pip install -e '.[sota]'
```

This pulls in:

- `torch`, `torchaudio` — for AVES, BioLingual, and any other torch encoder
- `tensorflow`, `tensorflow-hub`, `kagglehub` — for Perch 2.0
- `transformers` — for BioLingual CLAP
- `python-igraph`, `leidenalg` — for Leiden community detection

Optional BirdNET backend:

```bash
pip install -e '.[sota-birdnet]'
```

GPU is not strictly required but heavily recommended for files >1 h. Perch
runs comfortably on CPU for clips up to a few minutes.

---

## Quickstart

### From the CLI

```bash
biosound-cluster-sota path/to/recording.wav \
    --output outputs/sota_run \
    --encoder perch \
    --device auto \
    --knn 15 \
    --resolution 1.0 \
    --min-cluster-size 5
```

Common flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--encoder` | `perch`, `aves`, `birdnet`, `mock` | `perch` |
| `--device` | `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `--window-sec` | Override encoder-native window (s) | encoder default |
| `--hop-sec` | Sliding-window hop (s) | `1.0` |
| `--silence-db` | RMS dB threshold for silent windows | `-55.0` |
| `--knn` | Neighbours per node in k-NN graph | `15` |
| `--resolution` | Leiden resolution (higher = more clusters) | `1.0` |
| `--min-cluster-size` | Drop clusters smaller than this | `5` |
| `--no-zero-shot` | Skip BioLingual captioning | off |
| `--no-clips` / `--no-spectrograms` | Metadata-only export | off |
| `--few-shot-labels` | `{event_id: label}` JSON for prototype reassignment | none |
| `--cache-dir` | Override the model cache dir | platform default |

### From Python

```python
from biosound_cluster.sota import SOTAConfig, process_audio_file_sota

result = process_audio_file_sota(
    "recording.wav",
    "outputs/sota_run",
    SOTAConfig(
        encoder="perch",
        knn_neighbors=15,
        leiden_resolution=1.0,
        min_cluster_size=5,
        enable_zero_shot=True,
    ),
)
print(result.n_events, result.n_clusters)
print(result.report_md)
```

`SOTAResult` is JSON-serialisable via `.to_dict()`.

---

## Output structure

```
outputs/sota_run/
├── cluster_000_size_042_a_bird_call/        ← one folder per cluster
│   ├── event_000003.wav
│   ├── event_000003.png                     ← spectrogram (unless --no-spectrograms)
│   └── ...
├── cluster_001_size_018_a_frog_or_amphibian_call/
├── cluster_002_size_007_an_insect_call_or_stridulation/
├── noise_unknown/                           ← windows that Leiden put in -1
├── events.csv                               ← one row per detected event
├── clusters.csv                             ← per-cluster summary
├── embeddings.npz                           ← (embeddings, start_times) of all loud windows
├── report.md                                ← human-readable report
├── index.html                               ← browsable review UI
└── run_metadata.json                        ← config + counts
```

Cluster folder names include the zero-shot label when captioning is enabled.
None of these folders should be treated as throwaway — `noise_unknown/` can
contain rare or very atypical events worth a second pass.

---

## How each step works

### 1. Encoder selection (`encoders/`)

All encoders satisfy the `AudioEncoder` protocol:

```python
class AudioEncoder(Protocol):
    @property
    def info(self) -> EncoderInfo: ...
    def embed(self, windows: np.ndarray) -> np.ndarray: ...
```

`EncoderInfo` carries the encoder's native `sample_rate`, `window_sec`, and
`embedding_dim`. The pipeline reads these so it can match audio loading,
window slicing, and downstream array shapes to the backend.

Backends are imported lazily inside `factory.load_encoder`, so the package
imports cleanly even if torch/tf/kagglehub aren't installed — the error only
surfaces when you actually instantiate the heavy backend.

| Backend | Native rate | Window | Embedding | Weight source |
| --- | --- | --- | --- | --- |
| `perch` | 32 000 Hz | 5.0 s | 1280-d | `google/bird-vocalization-classifier` on Kaggle Hub |
| `aves`  | 16 000 Hz | 5.0 s | 768-d  | `earthspecies/aves` on torch.hub |
| `birdnet` | 48 000 Hz | 3.0 s | 1024-d | `birdnet-analyzer` PyPI |
| `mock`  | configurable | configurable | 256-d | deterministic mel-stats (for tests) |

Embeddings are L2-normalised before being returned so cosine similarity
equals dot product downstream.

### 2. Windowing (`windowing.py`)

`extract_windows(audio, sr, window_sec, hop_sec)` slices the waveform into a
`(n_windows, n_samples)` float32 matrix and returns the start time of each
window. The last window is zero-padded so the matrix has uniform shape.

`silence_mask(windows, threshold_db)` returns a boolean mask of windows whose
RMS (in dB ref=1.0) exceeds `threshold_db`. Only those windows are embedded —
this is the cheap-but-effective filter that keeps the encoder from running on
~50 % silent recordings.

### 3. Clustering (`graph_clustering.py`)

`cluster_embeddings(...)` does three things:

1. **k-NN graph**: cosine k-NN (`sklearn.neighbors.NearestNeighbors`) →
   weighted edge list (`weight = 1 - cosine_distance`).
2. **Leiden**: `leidenalg.find_partition` with the
   `RBConfigurationVertexPartition` and the user's `resolution`. Higher
   resolution → more, smaller clusters.
3. **Stability**: re-run Leiden on `stability_subsamples` random subsamples
   (default 5 subsamples, 80 % each) and compute, for each window, the
   fraction of other windows whose co-assignment relationship matches the
   base run. This gives a per-window stability in [0, 1] used both to weight
   events and to optionally route fragile clusters to a review folder.

Clusters smaller than `min_cluster_size` are demoted to noise (label `-1`).

Why Leiden over HDBSCAN on dense high-dim embeddings:

- robust to noise without a hard min-cluster-size cliff,
- stable partitions across runs (it's deterministic given a seed),
- no `-1` mega-cluster swallowing most of the data,
- resolution parameter gives controllable granularity per dataset.

### 4. Event extraction (`event_extraction.py`)

`runs_to_events` walks the window stream and emits one event for each
contiguous run of windows sharing the same cluster ID. A run breaks when:

- the label changes,
- the gap between windows exceeds `max_event_gap`,
- extending would push the event past `max_event_duration`.

`temporal_nms` then de-duplicates overlapping events that came from
overlapping windows: it keeps the event with the highest mean stability.

`refine_boundaries` tightens each event's `[start, end]` using a smoothed
`|audio|` envelope: it finds the first / last sample above
`peak * 10^(-activity_db/20)` and crops to that range (plus padding).

### 5. Zero-shot captioning (`zero_shot.py`)

For each cluster, we take up to `zero_shot_clips_per_cluster` representative
clips (chosen earlier by centroid distance) and embed them with **BioLingual**
(`davidrrobinson/BioLingual`), a CLAP model trained on animal vocalisation
audio + text pairs. We embed each prompt in `zero_shot_prompts` with the same
model and pick the prompt with the highest cosine similarity to the cluster's
mean audio embedding.

Important: BioLingual lives in its **own** embedding space, separate from
Perch. We never mix the two — captioning is a post-hoc explanation, not a
clustering signal. This keeps the clustering driven by the strongest available
audio model and uses CLAP only where it adds value (text labels).

The default prompt list covers birds, frogs, insects, mammals, bats, weather,
human noise, water, anthropogenic noise. Pass `--no-zero-shot` to skip.

### 6. Few-shot refinement (`few_shot.py`)

This is the real human-in-the-loop differentiator and is **disabled by
default**. Workflow:

1. User reviews one full SOTA run, labels ~3-5 events per acoustic family
   they care about in a JSON file: `{"event_000123": "puffin", ...}`.
2. Re-run with `--few-shot-labels labels.json`.
3. The pipeline computes a prototype embedding per label (unit-norm mean of
   labelled events) and re-assigns each cluster's centroid to the nearest
   prototype, above `few_shot_min_confidence` cosine similarity.
4. Cluster folder names get renamed accordingly on the next export.

This is the DCASE 2024 Task 5 winning recipe (prototypical networks on
pretrained embeddings) applied as a second pass instead of as the entire
pipeline.

### 7. Export (`export.py`)

`write_event_assets` writes wav clips and spectrograms under per-cluster
folders. `write_csvs` writes `events.csv` and `clusters.csv` (one row per
event / cluster, with all metadata). `write_embeddings` saves the raw
embedding matrix to `embeddings.npz` so you can run other analyses without
re-encoding. `write_report` produces a markdown summary, and
`write_index_html` produces a minimal browsable HTML page with `<audio>`
controls and inline spectrograms.

---

## Schemas

```python
@dataclass
class SOTAEvent:
    event_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    cluster_id: int             # -1 if noise
    mean_window_score: float    # mean per-window stability
    n_windows: int              # how many windows this event spans
    rms_db: float
    is_noise: bool
    is_low_stability: bool
    clip_path: str | None
    spectrogram_path: str | None
    representative_rank: int | None   # 0 = closest to centroid
    centroid_distance: float | None
    embedding: np.ndarray | None      # not serialised to CSV
```

```python
@dataclass
class SOTACluster:
    cluster_id: int
    size: int
    folder_name: str
    mean_stability: float
    representative_event_ids: list[str]   # ordered: closest to farthest from centroid
    zero_shot_label: str | None
    zero_shot_score: float | None
    is_noise: bool
```

```python
@dataclass
class SOTAResult:
    input_path: str
    output_dir: str
    encoder: str
    embedding_dim: int
    duration_sec: float
    sample_rate: int
    n_windows: int
    n_events: int
    n_clusters: int
    n_noise_events: int
    events_csv: str
    clusters_csv: str
    report_md: str
    index_html: str | None
    embeddings_npy: str | None
```

---

## Configuration reference

Every knob is on `SOTAConfig` (see `config.py` for full docstring). The most
useful groups:

**Audio**
- `sample_rate` (default 32 000) — overridden by the encoder's native rate
  when needed.

**Encoder**
- `encoder` — `perch | aves | birdnet | mock`
- `encoder_device` — `auto | cpu | cuda | mps`
- `encoder_batch_size` — batch size used inside the encoder
- `encoder_cache_dir` — where Kaggle / torch.hub put weights

**Windowing**
- `window_sec` (None → encoder native)
- `hop_sec` (1.0)
- `silence_rms_db` (-55.0)

**Clustering**
- `knn_neighbors` (15)
- `knn_metric` (`cosine`)
- `leiden_resolution` (1.0) — increase for more granular clusters
- `leiden_n_iterations` (10)
- `leiden_seed` (42)
- `min_cluster_size` (5)
- `stability_subsamples` (5), `stability_subsample_fraction` (0.8)
- `min_stability_for_keep` (0.0) — route low-stability events to a review folder

**Event extraction**
- `min_event_duration` (0.10), `max_event_duration` (30.0)
- `max_event_gap` (1.0)
- `event_nms_iou` (0.5)
- `refine_onset_offset` (True), `refinement_activity_db` (8.0),
  `refinement_padding` (0.05), `refinement_smoothing_sec` (0.02)

**Zero-shot**
- `enable_zero_shot` (True)
- `zero_shot_model` (`biolingual`)
- `zero_shot_prompts` (curated default list)
- `zero_shot_clips_per_cluster` (4)

**Few-shot**
- `enable_few_shot` (False)
- `few_shot_labels_path` (None)
- `few_shot_min_confidence` (0.45)

**Export**
- `export_clips` / `export_spectrograms` / `export_embeddings` /
  `export_index_html`
- `representatives_per_cluster` (16)
- `cluster_folder_prefix`, `noise_folder_name`, `low_stability_folder_name`

---

## Testing

The mock encoder lets the entire pipeline run on CI without GPU / heavy
deps. The smoke test (`tests/sota/test_sota_pipeline_smoke.py`) generates a
synthetic recording with two distinct acoustic families and asserts:

- events are detected,
- clusters are produced,
- wav clips, report, index, embeddings, run_metadata are written.

Run the full SOTA test suite:

```bash
PYTHONPATH=src python3 -m pytest tests/sota/ -v
```

Currently: **11 tests, all passing.**

---

## Relationship with the legacy pipeline

| Concern | Legacy (`biosound_cluster.pipeline`) | SOTA (`biosound_cluster.sota`) |
| --- | --- | --- |
| Segmentation | RMS + spectral flux + refinement + polyphony + eventness + candidate selection + short-event routing (~7 modules) | sliding window + cluster runs (1 module) |
| Embedding | handcrafted log-mel + global features | pretrained encoder (Perch / AVES / BirdNET) |
| Clustering | UMAP + HDBSCAN with fallback | k-NN cosine + Leiden + stability subsampling |
| Captioning | none | optional BioLingual zero-shot |
| HITL | review folders (mixed / low-confidence / short) | review folders + optional few-shot prototypes |
| Heavy deps | none | torch / tf / igraph / transformers (extras-only) |
| Output structure | per-cluster folders + review folders + CSVs + report + HTML | same shape, simpler review folders, plus embeddings.npz |

Both APIs return analogous result objects (`ProcessResult` vs `SOTAResult`)
so downstream tooling can branch on which engine produced a run.

---

## Roadmap

Not implemented yet but designed-for:

1. **DCASE evaluation harness** mirroring `biosound_cluster.evaluation.cli`
   so we can publish a head-to-head precision/recall/F1 against the legacy
   pipeline across all ME / BV / JD / MT / WMW / PB subsets.
2. **Active-learning loop**: after few-shot refinement, suggest the most
   informative next clips to label (highest entropy under the prototype
   posterior).
3. **Multi-encoder ensembling**: concatenate Perch + AVES embeddings before
   clustering. Costs ~2× compute, often nets a few F1 points.
4. **Time-aware Leiden**: bias the k-NN graph with temporal proximity to
   prefer cohesive events.
