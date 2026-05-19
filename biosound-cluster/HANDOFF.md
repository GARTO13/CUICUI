# biosound-cluster handoff

Date: 2026-05-19  
Project path: `/Users/thomas/CUICUI/biosound-cluster`

## Product goal

`biosound-cluster` is a human-in-the-loop bioacoustic annotation assistant.

It does not classify species. It does not say "this is bird X" or "this is frog Y".

The tool takes a local long audio file, detects candidate acoustic events, extracts handcrafted
audio embeddings, groups similar sounds with UMAP + HDBSCAN, and exports review folders:

- one folder per acoustic family,
- `.wav` clips,
- spectrogram `.png` images,
- CSV/JSON metadata,
- `report.md`,
- `index.html`.

The target workflow is:

```bash
biosound-cluster input.wav --output outputs/run_001
```

Then a researcher opens the output folders or `index.html`, listens to representative clips, and
manually labels acoustic families.

## Current pipeline

Main API:

```python
from biosound_cluster import process_audio_file

result = process_audio_file("mon_audio.wav", "output_clusters")
```

Core steps:

1. Load audio with `librosa`, mono, safe normalization.
2. Detect broad candidate events with RMS energy + spectral flux.
3. Refine temporal boundaries and remove near-duplicate intervals.
4. Analyze polyphony and split separable overlapping time-frequency components.
5. Route low-confidence/noisy candidates out of normal clusters.
6. Score eventness and route weak temporal candidates out of normal clusters.
7. Prune duplicate or excessive component candidates.
8. Route very short events to secondary review.
9. Extract handcrafted log-mel/global acoustic embeddings.
10. Cluster with UMAP + HDBSCAN, with robust fallback.
11. Export all media and metadata.

## Important modules

- `pipeline.py`: orchestrates the full processing flow.
- `segmentation.py`: broad candidate event detection.
- `segmentation_refinement.py`: boundary tightening and temporal NMS.
- `polyphony.py`: overlapping-sound detection and rough STFT-mask separation.
- `noise.py`: acoustic quality/noise routing.
- `eventness.py`: temporal salience scoring.
- `candidate_selection.py`: component limiting and duplicate pruning.
- `review_routing.py`: short-event review routing.
- `features.py`, `embeddings.py`: handcrafted acoustic embeddings.
- `clustering.py`: UMAP + HDBSCAN clustering.
- `export.py`: folder/media/CSV/report/HTML export.
- `evaluation/`: DCASE 2024 Task 5 evaluation, metrics, reporting, tuning.

## Recent implementations

### 1. Segmentation refinement

Added `segmentation_refinement.py`.

Purpose:

- tighten broad event boundaries,
- reduce oversized clips,
- remove near-duplicate temporal detections before clustering.

Useful config fields:

```python
enable_segmentation_refinement = True
refinement_activity_db = 3.0
refinement_peak_drop_db = 18.0
refinement_padding = 0.03
refinement_min_trim_sec = 0.04
refinement_nms_iou = 0.85
```

### 2. Polyphony handling

Added `polyphony.py`.

Purpose:

- detect events that likely contain overlapping acoustic sources,
- split separable time-frequency components using STFT masks,
- send clean components to clustering,
- route inseparable mixed clips to `mixed_overlapping_size_XXX/`.

Important product point:

This is not source separation for perfect audio quality. It is a practical cluster-purity tool.

### 3. Noise filtering

Added `noise.py`.

Purpose:

- compute acoustic quality features such as SNR, flatness, tonality and bandwidth,
- route low-confidence broadband/weakly structured events to
  `low_confidence_noise_size_XXX/`,
- keep those clips for review instead of deleting them.

Useful config:

```python
enable_noise_filtering = True
noise_mode = "balanced"  # exploratory, balanced, conservative
min_quality_for_clustering = 0.45
```

### 4. Eventness filtering

Added `eventness.py`.

Purpose:

- reduce false positives that look like weak or temporally diffuse candidates,
- compute `eventness_score`, `temporal_contrast_db`, and `active_ratio`,
- route weak candidates to `low_confidence_noise_size_XXX/`.

Useful config:

```python
enable_eventness_filtering = True
min_eventness_for_clustering = 0.28
min_component_eventness_for_clustering = 0.42
eventness_min_contrast_db = 4.0
eventness_max_active_ratio = 0.92
```

Current default compromise:

```python
min_component_eventness_for_clustering = 0.42
```

This improved precision a little without killing recall too much on the ME mini-evaluation.

### 5. Candidate selection

Added `candidate_selection.py`.

Purpose:

- keep only the strongest separated components per original parent event,
- prune near-duplicate temporal candidates with NMS,
- route pruned candidates to review instead of deleting them.

Useful config:

```python
enable_candidate_selection = True
max_components_per_parent = 3
candidate_nms_iou = 0.98
```

Tested values:

- `max_components_per_parent = 2` was too aggressive and reduced recall.
- `max_components_per_parent = 3` is the current compromise.

### 6. Short-event review routing

Added `review_routing.py`.

Purpose:

- remove very short clips from normal cluster folders,
- keep them in `short_events_review_size_XXX/`,
- avoid making cluster representatives full of tiny, hard-to-judge snippets.

Useful config:

```python
enable_short_event_review = True
min_review_event_duration = 0.20
export_short_events_review = True
```

CLI:

```bash
--min-review-event-duration 0.20
--disable-short-event-review
```

Important result:

- `0.30 s` was too aggressive on DCASE ME and hurt recall badly.
- `0.20 s` is the current default because it keeps the product benefit while preserving more recall.

## Output folders and meaning

Normal folders:

- `cluster_000_size_042/`: normal acoustic-family cluster.
- `noise_unknown_size_011/`: HDBSCAN noise/unknown events.

Review folders:

- `mixed_overlapping_size_XXX/`: overlapping sounds not trusted for normal clustering.
- `low_confidence_noise_size_XXX/`: weak/noisy/broadband candidates excluded before clustering.
- `short_events_review_size_XXX/`: very short events excluded from normal clusters.

None of these review folders should be treated as trash. They can contain real biological signals.
They are separated because they are poor material for clean first-pass acoustic-family clustering.

## DCASE evaluation

Evaluation command used for quick tests:

```bash
biosound-evaluate-dcase \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/eval_short_review_me2_020 \
  --split validation \
  --subset ME \
  --max-files 2 \
  --min-cluster-size 5 \
  --min-review-event-duration 0.20 \
  --no-clips \
  --force
```

Latest quick result on `ME`, 2 validation files:

```text
Final score:      60.5 / 100
Precision:        0.113
Recall:           0.565
Detection F1:     0.188
Mean IoU:         0.421
Weighted purity:  1.000
Compression:      25.00 events/cluster
Clusters:         2
Mixed events:     0
Component events: 295
Low-confidence:   74
Short review:     166
```

Output:

```text
outputs/eval_short_review_me2_020/
  evaluation_summary.json
  evaluation_report.md
  per_file_metrics.csv
  matched_predictions.csv
  runs/
```

For comparison, with `min_review_event_duration = 0.30`:

```text
Final score:  49.9 / 100
Precision:    0.058
Recall:       0.194
F1:           0.089
Short review: 332
```

Conclusion: `0.30 s` removes too many annotated events from normal detection metrics.

## Metrics and what they mean

### Precision

Precision answers:

> Among the segments we predicted, how many correspond to annotated events?

Low precision means the system exports many false positives.

Current issue:

- precision is still low, around `0.11` on the quick ME test.
- The tool still detects too many extra candidates.

Product meaning:

- more clips for the researcher to inspect,
- noisier folders,
- more manual effort.

### Recall

Recall answers:

> Among the annotated events, how many did we recover?

Current quick ME recall:

```text
0.565
```

Product meaning:

- recall matters because missing rare calls is bad.
- But for this tool, recall is not the only goal. If recall is high but precision is terrible, the
  researcher gets too much junk.

Desired direction:

- keep recall roughly stable,
- raise precision by improving segmentation and candidate rejection.

### Detection F1

F1 combines precision and recall.

Current quick ME F1:

```text
0.188
```

This is low mostly because precision is low.

### Mean IoU

IoU measures temporal alignment between predicted segments and annotations.

Current quick ME mean IoU:

```text
0.421
```

Product meaning:

- predicted clips overlap annotations reasonably,
- but boundaries can still be too broad or slightly offset.

Improvement area:

- better onset/offset refinement,
- less padding when events are dense,
- event-specific boundary trimming.

### Weighted cluster purity

Cluster purity answers:

> When predicted events match known annotations, do similar labels land together?

Current quick ME weighted purity:

```text
1.000
```

Important caution:

- high purity on only 2 files and only 2 clusters is encouraging but not enough evidence.
- It can be artificially high if many uncertain events are routed away or if labels are simple.

Product meaning:

- the clusters that remain are cleaner.
- This supports the human-review goal.

### Annotation compression ratio

Compression ratio answers:

> How many events are summarized by each cluster?

Current quick ME:

```text
25 events/cluster
```

Product meaning:

- good for expert review: one cluster can summarize many similar clips.
- But compression is only useful if clusters are pure and representatives are good.

### Short review events

Short review count answers:

> How many tiny clips were kept out of normal clusters?

Current quick ME:

```text
166
```

Product meaning:

- normal clusters should be easier to listen to.
- tiny signals are preserved for a second pass.

Risk:

- if threshold is too high, recall drops because real annotated events are removed from normal
  detections. That happened at `0.30 s`.

## Current interpretation

The system is moving in the right product direction:

- clusters are cleaner,
- low-quality candidates are preserved but isolated,
- overlapping and short clips no longer pollute normal clusters as much,
- the output structure is much more useful for manual review.

The main weakness remains precision.

In plain terms:

> The system still finds too many candidate segments. It often keeps recall, but it makes the
> researcher inspect too many extra clips.

The next work should focus less on clustering and more on segmentation/candidate quality.

## What to improve next

### Priority 1: better segmentation

Most metric weakness comes from false-positive segments.

Possible next improvements:

- adaptive thresholds per frequency band,
- better local noise-floor modeling,
- multi-scale event detection for very short vs medium events,
- stricter flux-only detections,
- onset/offset refinement based on local energy valleys,
- automatic per-file threshold calibration.

### Priority 2: representative selection

Even with imperfect detection, the product can improve if the first 10-15 clips per cluster are the
best examples.

Recommended:

- compute representative quality score,
- prefer high eventness, high purity, high SNR, medium duration,
- avoid representatives from review-like edge cases,
- export `_representatives/` sorted by review quality, not only centroid distance.

### Priority 3: parameter tuning on more subsets

The latest numbers are only from `ME` with 2 files.

Need test:

```bash
biosound-evaluate-dcase \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/eval_more \
  --split validation \
  --max-files 10 \
  --min-cluster-size 5 \
  --no-clips \
  --force
```

Also test by subset:

- `ME`
- `BV`
- `JD`
- `MT`
- `WMW`
- `PB`

Reason:

Each subset has different acoustics. A threshold that works on ME may fail elsewhere.

### Priority 4: review-folder evaluation

Currently DCASE metrics exclude `mixed`, `low_confidence_noise`, and `short_review` from normal
detection/clustering metrics.

This is correct for evaluating clean clusters, but we also need product metrics:

- how many GT events are preserved somewhere, including review folders?
- how many GT events were removed from normal clusters but still available to humans?
- are short-review folders mostly real events or mostly junk?

Recommended new metrics:

- `global_recall_any_folder`
- `review_folder_gt_retention`
- `normal_cluster_precision`
- `representative_precision_at_k`

### Priority 5: optional denoising, carefully

ICA was discussed, but ICA is not a good default here because most field recordings are mono.
ICA needs multiple independent channels/sensors to work properly.

Better next denoising options for mono bioacoustics:

- spectral gating with conservative thresholds,
- stationary noise profile estimation,
- band-limited event extraction,
- median-filter harmonic/percussive separation,
- robust quality scoring rather than destructive denoising.

Avoid heavy supervised models for now unless used only as optional non-species eventness filtering.

## Useful commands

Run tests:

```bash
.venv/bin/python -m pytest -q
```

Compile:

```bash
.venv/bin/python -m compileall src/biosound_cluster
```

Run one audio:

```bash
biosound-cluster "Data/your_audio.wav" --output outputs/your_run
```

Run DCASE quick:

```bash
biosound-evaluate-dcase \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/eval_quick \
  --split validation \
  --subset ME \
  --max-files 2 \
  --min-cluster-size 5 \
  --no-clips \
  --force
```

Run DCASE tuning:

```bash
biosound-evaluate-dcase \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/tune_dcase \
  --split validation \
  --max-files 10 \
  --no-clips \
  --tune
```

## Last validation

After the short-event review implementation:

```text
8 passed, 9 warnings
```

Warnings are from dependencies such as `audioread`, `librosa`, `umap`, and font cache handling.
They are not currently blocking.

## Suggested next session starting point

Start by reading:

1. `src/biosound_cluster/pipeline.py`
2. `src/biosound_cluster/segmentation.py`
3. `src/biosound_cluster/segmentation_refinement.py`
4. `src/biosound_cluster/eventness.py`
5. `src/biosound_cluster/review_routing.py`
6. `src/biosound_cluster/evaluation/metrics.py`

Then run:

```bash
.venv/bin/python -m pytest -q
```

Then run one small DCASE evaluation and inspect:

```text
outputs/.../evaluation_summary.json
outputs/.../per_file_metrics.csv
outputs/.../runs/<file>/events.csv
```

The most useful next engineering goal is:

> increase precision while keeping recall reasonably stable, by improving segmentation and candidate
> quality before clustering.

