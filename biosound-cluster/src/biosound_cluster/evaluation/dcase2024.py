"""DCASE 2024 Task 5 evaluation harness for biosound-cluster."""

from __future__ import annotations

import json
import logging
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from biosound_cluster.config import BioSoundConfig, config_fingerprint
from biosound_cluster.evaluation.metrics import (
    DetectionMetrics,
    aggregate_detection_metrics,
    assign_predictions_to_ground_truth,
    compute_annotation_assistant_metrics,
    compute_clustering_metrics,
    compute_detection_metrics,
    compute_global_score,
    compute_polyphony_metrics,
)
from biosound_cluster.evaluation.reporting import write_evaluation_report

LOGGER = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
ZENODO_RECORD_URL = "https://zenodo.org/records/10829604"
ZENODO_ANNOTATIONS_URL = "https://zenodo.org/records/10829604/files/Development_set_annotations.zip?download=1"


@dataclass(slots=True)
class DCASEAudioFile:
    audio_path: Path
    annotation_path: Path | None
    split: str
    subset: str | None
    file_id: str


@dataclass(slots=True)
class GroundTruthEvent:
    file_id: str
    start_sec: float
    end_sec: float
    label: str
    raw_label: str | None = None


def discover_dcase_files(
    dataset_dir: Path,
    annotations_dir: Path | None = None,
    split: str = "validation",
    subset: str | None = None,
    max_files: int | None = None,
) -> list[DCASEAudioFile]:
    """Discover DCASE audio files and best-effort matching annotation CSVs."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"DCASE dataset directory not found: {dataset_dir}")
    annotations_dir = Path(annotations_dir) if annotations_dir else dataset_dir
    split = split.lower()
    annotation_index = _build_annotation_index(annotations_dir)
    audio_files = sorted(
        path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    discovered: list[DCASEAudioFile] = []
    for audio_path in audio_files:
        file_stem = audio_path.stem
        inferred_split = _infer_split(audio_path, file_stem)
        inferred_subset = _infer_subset(audio_path, file_stem)
        if split != "all" and inferred_split != split:
            continue
        if subset and inferred_subset and inferred_subset.lower() != subset.lower():
            continue
        if subset and inferred_subset is None and subset.lower() not in str(audio_path).lower():
            continue
        file_id = _make_file_id(inferred_split, inferred_subset, file_stem)
        annotation_path = _find_annotation(audio_path, annotation_index)
        if annotation_path is None:
            LOGGER.warning("No annotation CSV found for %s", audio_path)
        discovered.append(
            DCASEAudioFile(
                audio_path=audio_path,
                annotation_path=annotation_path,
                split=inferred_split,
                subset=inferred_subset,
                file_id=file_id,
            )
        )
        if max_files is not None and len(discovered) >= max_files:
            break
    return discovered


def load_dcase_annotations(annotation_path: Path, file_id: str) -> list[GroundTruthEvent]:
    """Load DCASE-style annotations, keeping POS and ignoring NEG for main metrics."""
    annotation_path = Path(annotation_path)
    frame = pd.read_csv(annotation_path)
    if frame.empty:
        return []
    columns = {str(col).strip().lower(): col for col in frame.columns}
    start_col = _find_column(columns, ["starttime", "start_time", "start", "onset", "begin"])
    end_col = _find_column(columns, ["endtime", "end_time", "end", "offset"])
    label_col = _find_column(columns, ["label", "class", "species", "event_label", "annotation", "q", "type"])
    if start_col is None or end_col is None:
        raise ValueError(
            f"Could not detect start/end columns in {annotation_path}. Columns: {list(frame.columns)}"
        )
    if label_col is None:
        label_col = "__constant_event_label__"
        frame[label_col] = "event"

    events: list[GroundTruthEvent] = []
    for _, row in frame.iterrows():
        start = float(row[start_col])
        end = float(row[end_col])
        raw_label = str(row[label_col]) if pd.notna(row[label_col]) else "event"
        normalized = raw_label.strip()
        if normalized.upper() == "NEG":
            continue
        if normalized.upper() == "POS":
            label = "POS"
        elif normalized.upper() == "UNK":
            continue
        else:
            label = normalized or "event"
        if end <= start:
            continue
        events.append(
            GroundTruthEvent(
                file_id=file_id,
                start_sec=start,
                end_sec=end,
                label=label,
                raw_label=raw_label,
            )
        )
    return events


def run_pipeline_on_dcase_file(
    dcase_file: DCASEAudioFile,
    output_root: Path,
    config: BioSoundConfig,
    force: bool = False,
) -> Path:
    """Run or reuse a biosound-cluster output folder for one DCASE audio file."""
    output_root = Path(output_root)
    run_dir = output_root / _safe_name(dcase_file.file_id)
    events_csv = run_dir / "events.csv"
    metadata_json = run_dir / "run_metadata.json"
    expected_hash = config_fingerprint(config)
    if events_csv.exists() and metadata_json.exists() and not force:
        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
        if metadata.get("config_hash") == expected_hash:
            LOGGER.info("Reusing existing pipeline output for %s", dcase_file.file_id)
            return run_dir
        LOGGER.info("Config changed for %s; rerunning pipeline output", dcase_file.file_id)
    elif events_csv.exists() and not force:
        LOGGER.info("No config hash for %s; rerunning pipeline output", dcase_file.file_id)

    from biosound_cluster.pipeline import process_audio_file

    process_audio_file(dcase_file.audio_path, run_dir, config=config)
    return run_dir


def evaluate_dcase2024(
    dataset_dir: Path,
    output_dir: Path,
    annotations_dir: Path | None = None,
    split: str = "validation",
    subset: str | None = None,
    max_files: int | None = None,
    config: BioSoundConfig | None = None,
    iou_threshold: float = 0.3,
    overlap_threshold: float = 0.3,
    force: bool = False,
) -> dict:
    """Evaluate biosound-cluster as a human-in-the-loop annotation assistant on DCASE 2024 Task 5."""
    cfg = config or BioSoundConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    files = discover_dcase_files(
        Path(dataset_dir),
        annotations_dir=Path(annotations_dir) if annotations_dir else None,
        split=split,
        subset=subset,
        max_files=max_files,
    )
    per_file_rows: list[dict[str, object]] = []
    detection_items: list[DetectionMetrics] = []
    matched_frames: list[pd.DataFrame] = []
    all_events_frames: list[pd.DataFrame] = []
    failed_files: list[dict[str, str]] = []

    for item in files:
        try:
            gt_events = load_dcase_annotations(item.annotation_path, item.file_id) if item.annotation_path else []
            run_dir = run_pipeline_on_dcase_file(item, runs_dir, cfg, force=force)
            events_csv = run_dir / "events.csv"
            events_df = pd.read_csv(events_csv) if events_csv.exists() else pd.DataFrame()
            if not events_df.empty:
                events_df["file_id"] = item.file_id
                all_events_frames.append(events_df)
            detection = compute_detection_metrics(
                gt_events,
                events_df,
                iou_threshold=iou_threshold,
                overlap_threshold=overlap_threshold,
            )
            matched = assign_predictions_to_ground_truth(item.file_id, gt_events, events_df)
            matched_frames.append(matched)
            clustering = compute_clustering_metrics(matched)
            polyphony = compute_polyphony_metrics(events_df)
            assistant_metrics = compute_annotation_assistant_metrics(
                matched,
                n_gt=len(gt_events),
                iou_threshold=iou_threshold,
                overlap_threshold=overlap_threshold,
            )
            n_low_confidence_noise = _count_low_confidence_noise(events_df)
            n_ambiguous_review = _count_ambiguous_review(events_df)
            n_short_review = _count_short_review(events_df)
            score = compute_global_score(detection, clustering, polyphony)
            detection_items.append(detection)
            per_file_rows.append(
                {
                    "file_id": item.file_id,
                    "audio_path": str(item.audio_path),
                    "annotation_path": str(item.annotation_path) if item.annotation_path else "",
                    "split": item.split,
                    "subset": item.subset or "",
                    "n_gt": detection.n_gt,
                    "n_pred": detection.n_pred,
                    "precision": detection.precision,
                    "recall": detection.recall,
                    "detection_f1": detection.f1,
                    "mean_iou": detection.mean_iou,
                    "weighted_cluster_purity": clustering.weighted_cluster_purity,
                    "normal_cluster_precision": assistant_metrics.get("normal_cluster_precision"),
                    "global_recall_any_folder": assistant_metrics.get("global_recall_any_folder"),
                    "representative_precision_at_k": assistant_metrics.get("representative_precision_at_k"),
                    "annotation_compression_ratio": clustering.annotation_compression_ratio,
                    "n_clusters": clustering.n_clusters,
                    "n_noise": clustering.n_noise,
                    "n_mixed_events": polyphony.n_mixed_events,
                    "n_low_confidence_noise": n_low_confidence_noise,
                    "n_ambiguous_review": n_ambiguous_review,
                    "n_short_review_events": n_short_review,
                    "n_component_events": polyphony.n_component_events,
                    "final_score_100": score.final_score_100,
                    "run_dir": str(run_dir),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised by malformed real data
            LOGGER.exception("Failed to evaluate %s", item.audio_path)
            failed_files.append({"file_id": item.file_id, "audio_path": str(item.audio_path), "error": str(exc)})

    detection_agg = aggregate_detection_metrics(detection_items)
    matched_all = pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame()
    events_all = pd.concat(all_events_frames, ignore_index=True) if all_events_frames else pd.DataFrame()
    clustering_agg = compute_clustering_metrics(matched_all)
    polyphony_agg = compute_polyphony_metrics(events_all)
    noise_filtering = _noise_filtering_summary(events_all)
    short_event_review = _short_review_summary(events_all)
    assistant_metrics = compute_annotation_assistant_metrics(
        matched_all,
        n_gt=detection_agg.n_gt,
        iou_threshold=iou_threshold,
        overlap_threshold=overlap_threshold,
    )
    score_agg = compute_global_score(detection_agg, clustering_agg, polyphony_agg)

    per_file_csv = output_dir / "per_file_metrics.csv"
    matched_csv = output_dir / "matched_predictions.csv"
    summary_json = output_dir / "evaluation_summary.json"
    report_md = output_dir / "evaluation_report.md"
    pd.DataFrame(per_file_rows).to_csv(per_file_csv, index=False)
    matched_all.to_csv(matched_csv, index=False)

    summary = {
        "final_score_100": score_agg.final_score_100,
        "dataset": {
            "dataset_dir": str(dataset_dir),
            "annotations_dir": str(annotations_dir) if annotations_dir else "",
            "split": split,
            "subset": subset,
            "files_evaluated": len(per_file_rows),
            "files_discovered": len(files),
            "failed_files": failed_files,
        },
        "detection": detection_agg.to_dict(),
        "clustering": clustering_agg.to_dict(),
        "polyphony": polyphony_agg.to_dict(),
        "noise_filtering": noise_filtering,
        "short_event_review": short_event_review,
        "assistant_metrics": assistant_metrics,
        "score": score_agg.to_dict(),
        "outputs": {
            "per_file_metrics_csv": str(per_file_csv),
            "matched_predictions_csv": str(matched_csv),
            "evaluation_report_md": str(report_md),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_evaluation_report(
        report_md,
        summary=summary,
        detection=detection_agg,
        clustering=clustering_agg,
        polyphony=polyphony_agg,
        score=score_agg,
    )
    return summary


def compare_dcase2024_noise_modes(
    dataset_dir: Path,
    output_dir: Path,
    annotations_dir: Path | None = None,
    split: str = "validation",
    subset: str | None = None,
    max_files: int | None = None,
    base_config: BioSoundConfig | None = None,
    iou_threshold: float = 0.3,
    overlap_threshold: float = 0.3,
    force: bool = False,
) -> dict:
    """Evaluate baseline and noise-filtering modes and write a comparison table."""
    base = base_config or BioSoundConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = {
        "baseline_no_noise_filter": BioSoundConfig(**{**asdict(base), "enable_noise_filtering": False}),
        "noise_exploratory": BioSoundConfig(**{**asdict(base), "enable_noise_filtering": True, "noise_mode": "exploratory"}),
        "noise_balanced": BioSoundConfig(**{**asdict(base), "enable_noise_filtering": True, "noise_mode": "balanced"}),
        "noise_conservative": BioSoundConfig(**{**asdict(base), "enable_noise_filtering": True, "noise_mode": "conservative"}),
    }

    summaries: dict[str, dict] = {}
    rows: list[dict[str, object]] = []
    for name, config in configs.items():
        summary = evaluate_dcase2024(
            dataset_dir=dataset_dir,
            annotations_dir=annotations_dir,
            output_dir=output_dir / name,
            split=split,
            subset=subset,
            max_files=max_files,
            config=config,
            iou_threshold=iou_threshold,
            overlap_threshold=overlap_threshold,
            force=force,
        )
        summaries[name] = summary
        detection = summary.get("detection", {})
        clustering = summary.get("clustering", {})
        noise_filtering = summary.get("noise_filtering", {})
        rows.append(
            {
                "config": name,
                "final_score_100": summary.get("final_score_100", 0.0),
                "precision": detection.get("precision", 0.0),
                "recall": detection.get("recall", 0.0),
                "f1": detection.get("f1", 0.0),
                "mean_iou": detection.get("mean_iou", 0.0),
                "weighted_cluster_purity": clustering.get("weighted_cluster_purity", 0.0),
                "annotation_compression_ratio": clustering.get("annotation_compression_ratio", 0.0),
                "n_clusters": clustering.get("n_clusters", 0),
                "n_noise": clustering.get("n_noise", 0),
                "n_low_confidence_noise": noise_filtering.get("n_low_confidence_noise", 0),
                "mean_quality_score": noise_filtering.get("mean_quality_score"),
            }
        )

    comparison_csv = output_dir / "config_comparison.csv"
    comparison_json = output_dir / "config_comparison.json"
    pd.DataFrame(rows).to_csv(comparison_csv, index=False)
    result = {
        "comparison_csv": str(comparison_csv),
        "configs": rows,
        "summaries": summaries,
    }
    comparison_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def download_dcase2024_annotations(output_dir: Path) -> Path:
    """Download the small DCASE 2024 Task 5 annotation archive from Zenodo."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "Development_set_annotations.zip"
    extract_dir = output_dir / "Development_set_annotations"
    LOGGER.info("Downloading DCASE annotations from %s", ZENODO_RECORD_URL)
    urllib.request.urlretrieve(ZENODO_ANNOTATIONS_URL, zip_path)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    return extract_dir


def _build_annotation_index(annotations_dir: Path) -> dict[str, list[Path]]:
    if not annotations_dir.exists():
        return {}
    index: dict[str, list[Path]] = {}
    for path in annotations_dir.rglob("*.csv"):
        keys = {path.stem.lower()}
        for suffix in ["_annotations", "-annotations", "_annotation", "-annotation"]:
            if path.stem.lower().endswith(suffix):
                keys.add(path.stem[: -len(suffix)].lower())
        for key in keys:
            index.setdefault(key, []).append(path)
    return index


def _find_annotation(audio_path: Path, annotation_index: dict[str, list[Path]]) -> Path | None:
    candidates = [
        audio_path.with_suffix(".csv"),
        audio_path.parent / f"{audio_path.stem}_annotations.csv",
        audio_path.parent / f"{audio_path.stem}_annotation.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    key = audio_path.stem.lower()
    if key in annotation_index:
        return sorted(annotation_index[key])[0]
    loose = [path for stem, paths in annotation_index.items() if key in stem or stem in key for path in paths]
    return sorted(loose)[0] if loose else None


def _infer_split(audio_path: Path, file_id: str) -> str:
    text = f"{audio_path} {file_id}".lower()
    if "validation" in text or "val" in text:
        return "validation"
    if "train" in text:
        return "train"
    return "all"


def _infer_subset(audio_path: Path, file_id: str) -> str | None:
    known = ["PB24", "WMW", "BV", "JD", "MT", "ME", "PB", "PW", "HB", "RD", "HT"]
    known_by_lower = {item.lower(): item for item in known}
    for part in audio_path.parts:
        match = known_by_lower.get(part.lower())
        if match:
            return match

    text = f"{audio_path.name} {file_id}".lower()
    for item in known:
        if item.lower() in text:
            return item
    token = file_id.split("_")[0].split("-")[0]
    return token.upper() if 1 < len(token) <= 5 and token.isalpha() else None


def _find_column(columns: dict[str, str], names: list[str]) -> str | None:
    normalized_names = [name.lower() for name in names]
    for name in normalized_names:
        if name in columns:
            return columns[name]
    compact = {key.replace("_", "").replace(" ", ""): value for key, value in columns.items()}
    for name in normalized_names:
        key = name.replace("_", "").replace(" ", "")
        if key in compact:
            return compact[key]
    return None


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe.strip("_") or "audio"


def _make_file_id(split: str, subset: str | None, file_stem: str) -> str:
    parts = [split, subset or "unknown", file_stem]
    return _safe_name("__".join(parts))


def _count_low_confidence_noise(events_df: pd.DataFrame) -> int:
    if events_df.empty or "source_type" not in events_df.columns:
        return 0
    return int((events_df["source_type"].fillna("") == "low_confidence_noise").sum())


def _count_ambiguous_review(events_df: pd.DataFrame) -> int:
    if events_df.empty or "source_type" not in events_df.columns:
        return 0
    return int((events_df["source_type"].fillna("") == "ambiguous_review").sum())


def _count_short_review(events_df: pd.DataFrame) -> int:
    if events_df.empty or "source_type" not in events_df.columns:
        return 0
    return int((events_df["source_type"].fillna("") == "short_review").sum())


def _noise_filtering_summary(events_df: pd.DataFrame) -> dict[str, object]:
    if events_df.empty or "source_type" not in events_df.columns:
        return {"enabled": False, "n_low_confidence_noise": 0, "low_confidence_rate": 0.0, "mean_quality_score": None}
    n_low = _count_low_confidence_noise(events_df)
    n_ambiguous = _count_ambiguous_review(events_df)
    quality = pd.to_numeric(events_df.get("quality_score"), errors="coerce").dropna() if "quality_score" in events_df.columns else pd.Series(dtype=float)
    return {
        "enabled": "quality_score" in events_df.columns,
        "n_low_confidence_noise": n_low,
        "n_ambiguous_review": n_ambiguous,
        "low_confidence_rate": float(n_low / len(events_df)) if len(events_df) else 0.0,
        "mean_quality_score": float(quality.mean()) if not quality.empty else None,
    }


def _short_review_summary(events_df: pd.DataFrame) -> dict[str, object]:
    if events_df.empty or "source_type" not in events_df.columns:
        return {"enabled": False, "n_short_review_events": 0, "short_review_rate": 0.0}
    n_short = _count_short_review(events_df)
    return {
        "enabled": "is_short_review_event" in events_df.columns,
        "n_short_review_events": n_short,
        "short_review_rate": float(n_short / len(events_df)) if len(events_df) else 0.0,
    }
