# biosound-cluster

`biosound-cluster` is a first working version of a human-in-the-loop bioacoustic labeling tool.
It takes one local audio file, detects candidate sound events, extracts handcrafted acoustic
embeddings, clusters similar sounds with UMAP + HDBSCAN, and exports each event into a folder
for researcher review.

The tool performs unsupervised acoustic grouping. Cluster IDs are not species names.
HDBSCAN noise points are not necessarily useless; they may contain rare sounds, ambiguous
sounds, or actual noise. Human validation is required. The purpose is to reduce expert
listening time by grouping similar events.

## What It Does

- Detects candidate acoustic events in a long recording.
- Computes handcrafted log-mel and spectral feature embeddings for each event.
- Detects likely overlapping sounds and either separates clean time-frequency components or isolates mixed clips for review.
- Groups acoustically similar events using UMAP + HDBSCAN.
- Exports short `.wav` clips and spectrogram `.png` files into cluster folders.
- Writes `events.csv`, `clusters.csv`, `run_metadata.json`, `report.md`, and a local `index.html`.
- Exposes both a Python API and a CLI.

## What It Does Not Do

- It does not classify species.
- It does not require a species database.
- It does not use XGBoost.
- It does not download external datasets.
- It does not require a GPU.
- It does not require internet at runtime.

## Installation

```bash
pip install -e .
```

WAV and FLAC inputs are preferred. AIFF and MP3 may work when your local audio backend can decode
them through `librosa`.

## CLI Usage

```bash
biosound-cluster path/to/recording.wav --output outputs/demo
```

Equivalent module invocation:

```bash
python -m biosound_cluster.cli path/to/recording.wav --output outputs/demo
```

Useful options:

```bash
biosound-cluster path/to/recording.wav \
  --output outputs/demo \
  --sample-rate 32000 \
  --min-cluster-size 10 \
  --min-event-duration 0.25 \
  --max-event-duration 8.0 \
  --merge-gap 0.4 \
  --padding 0.15 \
  --threshold-db 8.0 \
  --flux-percentile 90 \
  --flux-min-snr-db 1.5 \
  --max-events 200 \
  --verbose
```

Use `--no-spectrograms` to skip PNG generation for faster debugging runs. Use `--no-clips`
for large evaluation sweeps where you only need CSV/JSON metrics and do not want to fill the disk
with exported WAV/PNG media.

You can attach deployment metadata to a run. These values are stored in `run_metadata.json`,
`event_metadata.json`, and in a `.json` sidecar next to each exported audio clip:

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

Each event metadata JSON contains the recording start time, clip start/end seconds, absolute
clip start/end times when the recording start time is ISO-8601 parseable, sensor coordinates, and
the environment type.

The detector combines broadband energy, spectral flux, spectral concentration, and conservative
multi-band detection. The multi-band detector checks low, mid, high, and ultra-high frequency bands
for local SNR and temporal contrast, which helps keep faint structured calls without relying on one
global threshold. If a recording produces too many tiny structured false positives, try
`--disable-flux-detection`, `--disable-multiband-segmentation`, or let the DCASE tuner test those
options.

Segmentation refinement is enabled by default. It tightens event boundaries after the broad
detection pass and removes near-duplicate overlapping detections before clustering. Use
`--disable-segmentation-refinement` only when comparing against the older behavior.

Eventness filtering is also enabled by default. It computes temporal contrast and active-frame
occupancy, then routes weak candidates to `low_confidence_noise_size_...` for review instead of
letting them pollute normal clusters. Use `--disable-eventness-filtering` for maximum recall.

Candidate selection then keeps the strongest separated components per original event and removes
near-duplicate temporal candidates. Use `--max-components-per-parent` to control how many separated
components may enter normal clustering.

Short-event review is enabled by default. Events shorter than `--min-review-event-duration`
(default `0.20` seconds) are exported to `short_events_review_size_...` instead of entering the
main clusters. This keeps cluster representatives more useful when a researcher only wants to
listen to a handful of examples per acoustic family. Use `--disable-short-event-review` if you want
maximum recall in the main cluster folders.

Clusterability filtering is enabled by default. It combines eventness, local SNR, spectral
structure, duration confidence, embedding stability, broadband-noise penalty, and overlap penalty.
Events below `--min-clusterability-for-clustering` are kept in review metadata/folders instead of
entering normal clustering. This protects clusters while preserving traceability.

Embedding stability is also enabled by default. It compares handcrafted embeddings from the original
clip, a center crop, and a simple band-limited view. The original audio remains the truth; auxiliary
views are only used for scoring.

Acoustic pre-families are enabled by default. They are broad morphology buckets such as
`tonal_whistle`, `harmonic_call`, `pulse_train`, `broadband_click`, `noisy_burst`,
`insect_trill`, and `low_frequency_call`. They are not species labels.

Cluster stability scoring uses the main HDBSCAN membership probability by default. You can enable
a slower lightweight ensemble with `--cluster-ensemble-runs 3` or `5` to annotate
`cluster_stability_score` from multiple clustering runs.

Adaptive profiling is available but disabled by default. It profiles each recording and adjusts
thresholds, duration limits, merge gaps, and quality gates according to the acoustic regime:

```bash
biosound-cluster path/to/recording.wav \
  --output outputs/demo \
  --enable-auto-profile
```

Optional semantic tagging and denoising hooks are also present, but they require extra heavy
dependencies that are not installed by default. The core project still works without internet,
GPU, species labels, or external models.

Noise filtering is enabled by default. It computes SNR, spectral flatness, tonality, bandwidth,
and an aggregate acoustic quality score for each event. Low-confidence broadband or weakly
structured events are exported for review but excluded from normal clustering:

```bash
biosound-cluster path/to/recording.wav \
  --output outputs/demo \
  --noise-mode balanced \
  --min-quality-for-clustering 0.55
```

Use `--noise-mode exploratory` to keep more faint events, `--noise-mode conservative` to produce
cleaner clusters, or `--disable-noise-filtering` to restore the older behavior.

Polyphony handling is enabled by default. To disable it:

```bash
biosound-cluster path/to/recording.wav --output outputs/demo --disable-polyphony-handling
```

Useful polyphony controls:

```bash
biosound-cluster path/to/recording.wav \
  --output outputs/demo \
  --component-snr-db 10 \
  --min-purity-for-clustering 0.55 \
  --max-components-per-event 4
```

## Python API

```python
from biosound_cluster import process_audio_file

result = process_audio_file(
    input_path="mon_audio.wav",
    output_dir="output_clusters",
)

print(result)
```

The core MVP is:

```python
process_audio_file("mon_audio.wav", "output_clusters")
```

The concrete result is a folder per acoustic family, with short clips and spectrograms ready for
a researcher to listen to, inspect, and label manually.

## Browser / Lovable API

The project also exposes a small FastAPI service for a website or Lovable frontend. It accepts one
`.wav` upload plus deployment metadata, processes the recording in a background job, then returns
clusters, event audio URLs, spectrogram URLs, and metadata.

Start the API:

```bash
pip install -e .
biosound-api
```

Lovable/Cloudflare-style launch command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then expose it from a second terminal:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

or:

```bash
ngrok http 8000
```

Use the generated HTTPS tunnel URL as the frontend API base URL.

By default the API listens on `http://127.0.0.1:8000` and stores jobs in `outputs/api_jobs/`.

Useful environment variables:

```bash
BIOSOUND_API_HOST=127.0.0.1
BIOSOUND_API_PORT=8000
BIOSOUND_API_ROOT=outputs/api_jobs
BIOSOUND_API_MAX_UPLOAD_MB=2048
BIOSOUND_API_WORKERS=1
BIOSOUND_API_CORS_ORIGINS="*"
```

Upload from a frontend as `multipart/form-data`:

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F "file=@path/to/recording.wav" \
  -F "sensor_id=GUYANE_001" \
  -F "sensor_latitude=4.9372" \
  -F "sensor_longitude=-52.3260" \
  -F "environment_type=tropical_forest" \
  -F "recording_start_time=2026-05-20T06:30:00+02:00" \
  -F "recording_timezone=Europe/Paris"
```

Then poll and fetch results:

```bash
GET /api/jobs/{job_id}
GET /api/jobs/{job_id}/result
GET /api/jobs/{job_id}/clusters
GET /api/jobs/{job_id}/events
GET /api/jobs/{job_id}/files/{relative_output_path}
```

Long-audio safeguards:

- uploads are streamed to disk in chunks;
- only `.wav` files with a RIFF/WAVE header are accepted;
- upload size is capped by `BIOSOUND_API_MAX_UPLOAD_MB`;
- processing runs asynchronously in a bounded worker pool;
- generated files are served only from the job output folder;
- each clip keeps a JSON metadata sidecar with recording time, cut timing, sensor metadata, and
  scoring fields.

## Output Structure

```text
outputs/my_run/
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
      event_000012__12.340-13.820.png
    event_000012__12.340-13.820.wav
    event_000012__12.340-13.820.json
    event_000012__12.340-13.820.png
    event_000034__45.100-46.700.wav
    event_000034__45.100-46.700.png

  cluster_001_size_018/
    ...

  mixed_overlapping_size_009/
    event_000456__78.200-81.000.wav
    event_000456__78.200-81.000.png

  low_confidence_noise_size_023/
    event_000789__91.000-91.500.wav
    _cluster_manifest.csv

  short_events_review_size_014/
    event_000912__105.220-105.410.wav
    _cluster_manifest.csv

  noise_unknown_size_011/
    ...
```

Each cluster folder is an acoustic family discovered from the file. These folders are meant to make
manual review faster: listen to representative clips, inspect spectrograms, then assign your own
research labels outside the model.

`noise_unknown_size_...` contains HDBSCAN noise or fallback unknown events. These are not trash by
definition; they can include rare calls, overlapping events, very short sounds, ambiguous sounds,
or actual background noise.

`mixed_overlapping_size_...` contains clips excluded from normal clustering because the tool detected
several overlapping acoustic sources and could not separate them reliably. These sounds may still be
biologically valuable; they are simply routed for expert review instead of being allowed to pollute
the normal acoustic-family clusters.

`low_confidence_noise_size_...` contains events excluded before clustering because they look
broadband, low-SNR, weakly structured, or ambiguous by clusterability scoring. They are kept for
human review rather than deleted, because rare biological sounds can sometimes look noisy.

`short_events_review_size_...` contains very short events excluded from the main clusters. They may
be real signals, but they are often poor representatives for quick cluster labeling because there is
too little context to judge them confidently.

## Acceptance Scenario

After installing, this should work:

```bash
pip install -e .
pytest
biosound-cluster tests/generated_synthetic.wav --output outputs/demo
```

The smoke test creates `tests/generated_synthetic.wav` locally before running the pipeline.

## Evaluating on DCASE 2024 Task 5

This project can evaluate the unsupervised bioacoustic clustering pipeline on DCASE 2024 Task 5.

Example:

```bash
biosound-evaluate-dcase \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/eval_dcase \
  --split validation \
  --max-files 10
```

For longer sweeps, `--no-clips` keeps only metadata and reports:

```bash
biosound-evaluate-dcase \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/eval_dcase \
  --split validation \
  --max-files 10 \
  --no-clips
```

To compare the noise filter settings automatically:

```bash
biosound-evaluate-dcase \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/eval_noise_compare \
  --split validation \
  --max-files 10 \
  --no-clips \
  --compare-noise-modes
```

This writes `config_comparison.csv` and `config_comparison.json` with baseline, exploratory,
balanced, and conservative noise-filtering results. Pipeline runs also store a config hash in
`run_metadata.json`; cached outputs are reused only when the config matches.

To run a small deterministic parameter tuning sweep on DCASE:

```bash
biosound-evaluate-dcase \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/tune_dcase \
  --split validation \
  --max-files 10 \
  --no-clips \
  --tune
```

You can also use:

```bash
biosound-tune-dcase \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/tune_dcase \
  --split validation \
  --max-files 10 \
  --no-clips
```

Tuning writes `tuning_results.csv`, `tuning_summary.json`, and `best_config.json`. This is not
species classification or supervised model training; it only chooses detection/filtering/clustering
parameters that make the unsupervised annotation assistant more useful on the chosen validation set.

Equivalent script invocation:

```bash
python scripts/evaluate_dcase2024.py \
  --dataset-dir data/dcase2024_task5/Development_Set \
  --output-dir outputs/eval_dcase \
  --split validation \
  --max-files 10
```

The score is not a species classification accuracy. It measures the usefulness of the system as an
annotation assistant:

- event detection quality,
- cluster purity,
- compression of expert listening effort,
- temporal alignment,
- polyphony handling.

The DCASE development audio archive is large, so the evaluator does not download it silently. Use
`--download` only when you explicitly want download assistance; otherwise place the dataset locally
and pass `--dataset-dir`.
