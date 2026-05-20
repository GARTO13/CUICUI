# Website context for biosound-cluster

This document summarizes the project context, current product positioning, recent engineering work,
outputs, metrics, and useful wording for building a website or product presentation.

## Project name

`biosound-cluster`

## One-line positioning

`biosound-cluster` is an unsupervised bioacoustic clustering tool that helps researchers turn long
field recordings into acoustic-family folders ready for expert review.

## Core idea

The project is a passive bioacoustics pipeline for human-in-the-loop annotation.

It does not classify species. It does not predict "this is species X". Instead, it detects candidate
sound events, groups acoustically similar clips, and exports review folders so a human expert can
listen, inspect, validate, and label the clusters manually.

The model says:

> These sounds are acoustically similar.

It does not say:

> This sound is a specific animal species.

## Why this matters

Bioacoustic researchers often have long recordings containing many sparse, overlapping, noisy, or
rare events. Listening linearly is slow. `biosound-cluster` aims to reduce expert listening time by:

- detecting candidate sound events,
- cutting events into short audio clips,
- generating spectrograms,
- grouping similar events,
- isolating ambiguous/noisy/mixed clips,
- ranking representative examples for each cluster,
- preserving deployment metadata such as sensor coordinates, recording start time, and environment.

The expert can then inspect a few representative clips per acoustic family instead of manually
scrubbing through an entire recording.

## Current product workflow

Command-line usage:

```bash
biosound-cluster input.wav --output outputs/run_001
```

Python API:

```python
from biosound_cluster import process_audio_file

result = process_audio_file(
    input_path="mon_audio.wav",
    output_dir="output_clusters",
)
```

Browser API for a Lovable/front-end app:

```bash
biosound-api
```

Default local server:

```text
http://127.0.0.1:8000
```

Main endpoints:

```text
GET  /health
POST /api/jobs
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/result
GET  /api/jobs/{job_id}/clusters
GET  /api/jobs/{job_id}/events
GET  /api/jobs/{job_id}/files/{relative_output_path}
```

`POST /api/jobs` accepts `multipart/form-data` with:

- `file`: one `.wav` recording,
- `sensor_id`,
- `sensor_latitude`,
- `sensor_longitude`,
- `sensor_elevation_m`,
- `environment_type`,
- `recording_start_time`,
- `recording_timezone`,
- processing options such as `sample_rate`, `min_cluster_size`, `max_events`,
  `generate_spectrograms`, `enable_polyphony_handling`, and
  `enable_clusterability_filtering`.

The frontend should create a job, poll `GET /api/jobs/{job_id}`, then call
`GET /api/jobs/{job_id}/result` when `status` becomes `done`. The result contains cluster records,
events, review folders, audio file URLs, spectrogram URLs, and embedded metadata.

Long-audio safeguards:

- uploads are streamed to disk instead of being read fully into memory,
- only `.wav` files with a RIFF/WAVE header are accepted,
- upload size is capped with `BIOSOUND_API_MAX_UPLOAD_MB`,
- processing is asynchronous through a bounded worker pool,
- generated files are served only from the job output directory,
- every exported clip has metadata linking it to the original recording and cut timing.

Expected output:

```text
outputs/run_001/
  report.md
  run_metadata.json
  event_metadata.json
  events.csv
  clusters.csv
  index.html

  cluster_000_size_042/
    _cluster_manifest.csv
    _representatives/
      event_000012__12.340-13.820.wav
      event_000012__12.340-13.820.json
      event_000012__12.340-13.820.png
    event_000012__12.340-13.820.wav
    event_000012__12.340-13.820.json
    event_000012__12.340-13.820.png

  cluster_001_size_018/
    ...

  mixed_overlapping_size_009/
    ...

  low_confidence_noise_size_023/
    ...

  short_events_review_size_014/
    ...

  noise_unknown_size_011/
    ...
```

## What each folder means

### `cluster_XXX_size_YYY/`

Normal acoustic-family folders. These contain clips that the system thinks are clean enough and
similar enough to be useful for expert review.

The number is not a species label. `cluster_000` means only "acoustic group 0".

### `_representatives/`

Representative examples from a cluster. These are selected to be typical, stable, and listenable,
not only close to the mathematical centroid.

The representative ranking uses:

- cluster centroid similarity,
- eventness,
- local SNR,
- embedding stability,
- duration confidence,
- overlap penalty.

Representative clips copy their `.json` sidecar too, so every audio file remains connected to the
recording metadata and cut timing.

## Metadata outputs

The pipeline can receive deployment metadata as inputs:

- sensor or deployment ID,
- sensor latitude and longitude,
- optional elevation,
- environment type,
- recording start time,
- timezone label.

These are written to:

- `run_metadata.json`,
- `event_metadata.json`,
- one `.json` sidecar next to each exported `.wav`.

Each event metadata JSON includes:

- recording start time,
- clip start/end seconds inside the recording,
- absolute clip start/end time when `recording_start_time` is ISO-8601 parseable,
- sensor coordinates,
- environment type,
- clip and spectrogram paths,
- source folder type,
- cluster ID when available,
- key scoring fields.

Example:

```bash
biosound-cluster path/to/recording.wav \
  --output outputs/demo \
  --sensor-id GUYANE_001 \
  --sensor-latitude 4.9372 \
  --sensor-longitude -52.3260 \
  --sensor-elevation-m 18.5 \
  --environment-type tropical_forest \
  --recording-start-time "2026-05-20T06:30:00+02:00" \
  --recording-timezone Europe/Paris
```

### `mixed_overlapping_size_XXX/`

Clips containing probable overlapping sources. These are excluded from normal clustering when the
system cannot separate them reliably.

Important: mixed clips are not useless. They may contain real biological signals, but they require
careful expert review.

### `low_confidence_noise_size_XXX/`

Events that are probably noisy, low-SNR, broadband, weakly structured, or ambiguous. They are kept
for traceability but excluded from normal clusters.

This folder is not a trash bin. Rare biological signals can sometimes look noisy.

### `short_events_review_size_XXX/`

Very short clips excluded from normal clusters because they are often poor representatives for
quick labeling. They are still preserved for secondary review.

### `noise_unknown_size_XXX/`

HDBSCAN noise or fallback unknown events. These may include rare sounds, ambiguous sounds, or actual
background noise.

## Current pipeline

The pipeline currently does:

1. Load local audio with `librosa`.
2. Normalize safely while preserving relative dynamics.
3. Detect candidate events using energy, spectral flux, spectral concentration, and multi-band cues.
4. Refine boundaries and remove near-duplicate intervals.
5. Detect and optionally split overlapping sounds with time-frequency masks.
6. Score acoustic quality and route low-confidence noise away from normal clusters.
7. Score eventness and route weak temporal candidates away from normal clusters.
8. Select best candidates and prune near duplicates.
9. Route very short events to secondary review.
10. Compute clusterability and embedding stability.
11. Assign broad acoustic pre-families.
12. Extract handcrafted log-mel and global acoustic embeddings.
13. Cluster with UMAP + HDBSCAN.
14. Export clips, spectrograms, CSVs, JSON, Markdown report, and static HTML index.

## Key modules

```text
src/biosound_cluster/
  audio_io.py
  segmentation.py
  segmentation_refinement.py
  polyphony.py
  noise.py
  eventness.py
  candidate_selection.py
  review_routing.py
  clusterability.py
  acoustic_prefamily.py
  embeddings.py
  clustering.py
  export.py
  visualization.py
  pipeline.py
  cli.py
  api.py
```

Evaluation:

```text
src/biosound_cluster/evaluation/
  dcase2024.py
  metrics.py
  reporting.py
  tuning.py
  cli.py
```

## Recent engineering work

### 1. Polyphony handling

Added logic to detect events with multiple overlapping time-frequency components.

The system can:

- keep clean original events,
- split separable components,
- route inseparable mixed events to `mixed_overlapping_size_XXX/`.

Goal: prevent clips containing several animals or sources from polluting clean clusters.

### 2. Noise and quality filtering

Added acoustic quality scoring:

- SNR,
- noise floor,
- spectral flatness,
- tonality,
- bandwidth,
- peak-band SNR,
- aggregate quality score.

Low-confidence candidates are preserved but routed out of normal clustering.

### 3. Eventness scoring

Added temporal salience scoring:

- `eventness_score`,
- `temporal_contrast_db`,
- `active_ratio`.

This helps reject diffuse, continuous, or weak candidates before clustering.

### 4. Short-event review

Added `short_events_review_size_XXX/`.

Default threshold:

```text
min_review_event_duration = 0.20 seconds
```

Reason: clips shorter than this may be real but are often bad cluster representatives.

### 5. Multi-band segmentation

Added conservative multi-band detection.

Default bands:

- low: 100-800 Hz,
- mid: 800-3000 Hz,
- high: 3000-8000 Hz,
- ultra-high: 8000 Hz to Nyquist if possible.

For each band, the system checks:

- local energy,
- local noise floor,
- local SNR,
- temporal variation / flux,
- spectral concentration.

Goal: keep faint structured calls while reducing broadband false positives.

### 6. Clusterability score

Added explicit `clusterability_score`.

It combines:

- eventness,
- local SNR,
- spectral structure,
- duration confidence,
- embedding stability,
- broadband-noise penalty,
- overlap penalty,
- edge-case penalty.

Routing:

```text
clusterability_score >= 0.55
  -> normal clustering

0.30 <= clusterability_score < 0.55
  -> ambiguous review

clusterability_score < 0.30
  -> low-priority/noise review
```

The event is never deleted. It remains traceable in `events.csv`.

### 7. Embedding stability

Added `embedding_stability_score`.

The system compares embeddings from auxiliary views:

- original,
- center crop,
- simple band-limited view.

Important principle:

> The original audio remains the truth. Auxiliary views are only used for scoring and features.

No heavy denoising dependency is required.

### 8. Acoustic pre-families

Added broad morphology families before clustering:

- `tonal_whistle`,
- `harmonic_call`,
- `pulse_train`,
- `broadband_click`,
- `noisy_burst`,
- `insect_trill`,
- `low_frequency_call`,
- `unknown`.

These are not species labels. They are only broad acoustic shapes used to stabilize clustering.

### 9. Better representative ranking

Representatives are now ranked using a `representative_score`, not only centroid distance.

The system prefers clips that are:

- typical of the cluster,
- temporally salient,
- higher SNR,
- stable across embedding views,
- reasonable duration,
- less overlapped.

This matters because a researcher may only listen to 10-15 clips per cluster.

### 10. Browser API for Lovable

Added a FastAPI service in `src/biosound_cluster/api.py`.

The API lets a website:

- upload a long `.wav` recording with sensor and environment metadata,
- start processing asynchronously,
- poll job status,
- fetch clusters with audio/spectrogram URLs,
- fetch event-level metadata and JSON sidecars,
- retrieve review folders for mixed, noisy, ambiguous, or short events.

This keeps the existing local pipeline intact while exposing a web-friendly contract.

### 11. Evaluation on DCASE 2024 Task 5

Added evaluation tooling for DCASE 2024 Task 5 Few-shot Bioacoustic Event Detection.

Example:

```bash
biosound-evaluate-dcase \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/eval_dcase \
  --split validation \
  --max-files 10 \
  --no-clips
```

The evaluation is not species classification accuracy.

It measures usefulness as an annotation assistant:

- event detection quality,
- cluster purity,
- annotation compression,
- temporal alignment,
- polyphony handling,
- review-folder preservation,
- representative precision.

## Important metrics

### Precision

Among predicted events, how many match annotated events?

Low precision means too many false positives.

For product UX, low precision means the expert has to inspect too many clips.

### Recall

Among annotated events, how many did we recover?

Recall matters because missing rare calls is bad.

But high recall alone is not enough if clusters are polluted.

### F1

Combines precision and recall.

Useful for quick comparison, but not the full product story.

### Mean IoU

Measures temporal overlap between predicted clips and annotations.

Low IoU means boundaries are too broad, too short, or shifted.

### Weighted cluster purity

Measures whether matched events with the same annotation label tend to land in the same clusters.

This is closer to the clustering objective than raw detection F1.

### Annotation compression ratio

Number of clustered events divided by number of non-noise clusters.

Example:

```text
300 events / 20 clusters = 15 events summarized per cluster
```

Higher compression can reduce expert effort, but only if cluster purity remains good.

### Normal cluster precision

Precision restricted to events that entered normal clusters.

This helps answer:

> Are the cluster folders clean?

### Global recall any folder

Recall across all folders, including review folders.

This helps answer:

> Did we preserve the event somewhere, even if it was not sent to normal clustering?

### Representative precision@K

Precision of the top K representatives per cluster.

This is important for the product because experts may only listen to a small number of examples.

### Cluster stability summary

Summarizes cluster stability scores.

By default this uses HDBSCAN membership probability. Optional ensemble runs can provide more robust
stability estimates.

## Recent quick results

The latest quick DCASE sanity check was intentionally small, not a full benchmark.

Command:

```bash
biosound-evaluate-dcase \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/eval_metrics_quick \
  --split validation \
  --subset ME \
  --max-files 1 \
  --min-cluster-size 5 \
  --no-clips \
  --force \
  --disable-polyphony-handling
```

Observed output:

```text
Final score:                45.3 / 100
Detection F1:               0.237
Precision:                  0.150
Recall:                     0.562
Mean IoU:                   0.044
Weighted purity:            1.000
Normal cluster precision:   0.160
Recall any folder:          0.562
Representative precision@5: 0.200
Compression:                2.00 events/cluster
Clusters:                   4
Low-confidence noise:       19
Ambiguous review:           75
Short review events:        0
```

Interpretation:

- The system preserves many events, but precision is still the main weakness.
- More events are routed to review, which protects normal clusters.
- Temporal boundaries still need work; low mean IoU points to segmentation/boundary issues.
- Weighted purity can be high on small tests but should not be overinterpreted.

## Current strengths

- Fully local pipeline.
- No species classifier.
- No external dataset required at runtime.
- GPU not required.
- Human-in-the-loop outputs are concrete and inspectable.
- Every routed event remains traceable.
- Original audio is preserved as truth.
- Static HTML output works without a web server.
- DCASE evaluation and tuning are available.

## Current limitations

The biggest technical weakness is still segmentation quality:

- some false positives remain,
- some event boundaries are not tight enough,
- noisy files can still produce many candidates,
- DCASE temporal IoU can be low.

The next highest-value improvements are:

- better onset/offset refinement,
- better local threshold calibration by acoustic regime,
- better representative precision,
- better review-folder metrics,
- tuning across several DCASE subsets, not only ME.

## Suggested website messaging

### Hero headline options

Option 1:

> Turn long bioacoustic recordings into acoustic families for expert review.

Option 2:

> Unsupervised sound clustering for bioacoustic annotation.

Option 3:

> Find, group, and review similar sounds without species classification.

### Supporting copy

`biosound-cluster` detects candidate sound events, groups acoustically similar clips, and exports
audio/spectrogram folders that researchers can validate manually. It is designed to reduce expert
listening time while preserving traceability for ambiguous, noisy, short, or overlapping sounds.

### What it does

- Detects candidate sound events.
- Extracts handcrafted acoustic embeddings.
- Groups similar sounds with UMAP + HDBSCAN.
- Separates or isolates overlapping sounds.
- Routes noisy and ambiguous events to review folders.
- Ranks representative examples.
- Exports clips, spectrograms, CSV metadata, reports, and a static HTML index.

### What it does not do

- It does not classify species.
- It does not require a species database.
- It does not train a supervised classifier.
- It does not require GPU.
- It does not require internet at runtime.
- It does not delete ambiguous events.

### Trust and scientific positioning

Recommended sentence:

> Cluster IDs are acoustic groups, not biological labels. Human validation is required.

Another useful sentence:

> The tool is designed to prioritize listening, not replace expert judgment.

## Website feature sections

### 1. From recording to review folders

Show:

```text
input.wav -> event detection -> embeddings -> clustering -> folder export
```

### 2. Human-in-the-loop by design

Explain:

- no species prediction,
- manual validation required,
- clusters reduce review time.

### 3. Clean clusters, preserved ambiguity

Explain:

- normal clusters are protected,
- mixed/noisy/short/ambiguous events go to review folders,
- nothing important is silently thrown away.

### 4. Representative clips

Explain:

- each cluster includes top examples,
- representatives are ranked by acoustic quality and typicality,
- experts can start by listening to the best clips.

### 5. Evaluation

Explain:

- DCASE 2024 Task 5 evaluation support,
- metrics measure annotation-assistant usefulness,
- score is not species classification accuracy.

## UI / website ideas

For a future web app, useful screens would be:

1. Upload or select local audio.
2. Processing summary.
3. Cluster browser.
4. Representative clip player.
5. Spectrogram view.
6. Review folders for mixed/noisy/short/ambiguous events.
7. CSV/metadata export.
8. Manual cluster label entry by the researcher.
9. Before/after evaluation dashboard.

Important UI rule:

The interface should avoid implying automatic species identification. Use terms like:

- acoustic family,
- candidate event,
- review folder,
- representative clip,
- expert label,
- validation status.

Avoid terms like:

- predicted species,
- classified as,
- animal identity,
- automatic species label.

## Current commands for demos

Run on one file:

```bash
biosound-cluster path/to/recording.wav --output outputs/demo
```

Fast metadata-only run:

```bash
biosound-cluster path/to/recording.wav \
  --output outputs/demo_fast \
  --no-clips
```

Enable adaptive profiling:

```bash
biosound-cluster path/to/recording.wav \
  --output outputs/demo_profiled \
  --enable-auto-profile
```

DCASE quick evaluation:

```bash
biosound-evaluate-dcase \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/eval_quick \
  --split validation \
  --subset ME \
  --max-files 2 \
  --no-clips \
  --force
```

Run tests:

```bash
.venv/bin/python -m pytest -q
```

## Development status

The code currently passes:

```text
8 passed, 6 warnings
```

Warnings come from dependencies such as `audioread`, `librosa`, `umap`, and local font cache setup.
They are not currently blocking.

## Best next technical focus

The next engineering focus should be:

> Improve temporal segmentation and boundary refinement to raise precision and IoU without losing
> recall of rare or faint events.

In product terms:

> Make the cluster folders cleaner and the representative clips more trustworthy, while keeping
> ambiguous events available for expert review.
