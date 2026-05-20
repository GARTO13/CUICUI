"""FastAPI service for browser-driven biosound-cluster processing."""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from biosound_cluster.config import BioSoundConfig, config_to_dict
from biosound_cluster.logging_utils import configure_logging, get_logger
from biosound_cluster.pipeline import process_audio_file

try:  # FastAPI is a runtime dependency for the API, but keep module helpers importable.
    from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
except ImportError:  # pragma: no cover - exercised only before API deps are installed
    FastAPI = None  # type: ignore[assignment]
    File = Form = Header = HTTPException = Request = UploadFile = None  # type: ignore[assignment]
    CORSMiddleware = None  # type: ignore[assignment]
    FileResponse = JSONResponse = None  # type: ignore[assignment]

LOGGER = get_logger(__name__)

API_ROOT = Path(os.environ.get("BIOSOUND_API_ROOT", "outputs/api_jobs"))
MAX_UPLOAD_MB = int(os.environ.get("BIOSOUND_API_MAX_UPLOAD_MB", "2048"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_CHUNK_MB = int(os.environ.get("BIOSOUND_API_MAX_CHUNK_MB", "50"))
MAX_CHUNK_BYTES = MAX_CHUNK_MB * 1024 * 1024
MAX_WORKERS = int(os.environ.get("BIOSOUND_API_WORKERS", "1"))
CHUNK_SIZE = 1024 * 1024


@dataclass(slots=True)
class ApiJob:
    """In-memory job status mirrored to disk as JSON."""

    job_id: str
    status: str
    created_at: str
    updated_at: str
    input_path: str
    output_dir: str
    error: str | None = None
    result: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_jobs: dict[str, ApiJob] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def create_app() -> Any:
    """Create the FastAPI app used by Lovable or any browser frontend."""
    if FastAPI is None:
        raise RuntimeError("FastAPI dependencies are not installed. Run `pip install -e .` again.")

    app = FastAPI(
        title="biosound-cluster API",
        version="0.1.0",
        description="Upload WAV recordings with metadata and retrieve unsupervised acoustic clusters.",
    )
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> Any:
        LOGGER.exception("Unhandled API error for %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": str(exc)})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_chunk_mb": MAX_CHUNK_MB,
            "workers": MAX_WORKERS,
            "api_root": str(API_ROOT),
        }

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        request: Request,
        file: UploadFile = File(...),
        content_length: int | None = Header(None),
        sensor_id: str | None = Form(None),
        sensor_latitude: str | None = Form(None),
        sensor_longitude: str | None = Form(None),
        sensor_elevation_m: str | None = Form(None),
        environment_type: str | None = Form(None),
        recording_start_time: str | None = Form(None),
        recording_timezone: str | None = Form(None),
        sample_rate: str = Form("32000"),
        min_cluster_size: str = Form("5"),
        max_events: str | None = Form(None),
        generate_spectrograms: str = Form("true"),
        enable_auto_profile: str = Form("true"),
        enable_polyphony_handling: str = Form("true"),
        enable_clusterability_filtering: str = Form("true"),
    ) -> dict[str, Any]:
        """Upload one WAV file and start asynchronous processing."""
        if content_length is not None and content_length > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Upload too large. Limit is {MAX_UPLOAD_MB} MB.")
        if not file.filename or not file.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only .wav uploads are accepted.")

        job_id = uuid.uuid4().hex
        job_root = API_ROOT / job_id
        upload_dir = job_root / "upload"
        output_dir = job_root / "run"
        upload_dir.mkdir(parents=True, exist_ok=False)
        safe_name = _sanitize_filename(file.filename)
        input_path = upload_dir / safe_name

        try:
            saved_size = await _save_upload_file(file, input_path)
        except Exception:
            shutil.rmtree(job_root, ignore_errors=True)
            raise

        config, metadata = _build_config(
            original_filename=file.filename,
            saved_size_bytes=saved_size,
            sensor_id=sensor_id,
            sensor_latitude=sensor_latitude,
            sensor_longitude=sensor_longitude,
            sensor_elevation_m=sensor_elevation_m,
            environment_type=environment_type,
            recording_start_time=recording_start_time,
            recording_timezone=recording_timezone,
            sample_rate=sample_rate,
            min_cluster_size=min_cluster_size,
            max_events=max_events,
            generate_spectrograms=generate_spectrograms,
            enable_auto_profile=enable_auto_profile,
            enable_polyphony_handling=enable_polyphony_handling,
            enable_clusterability_filtering=enable_clusterability_filtering,
        )
        job = _create_processing_job(job_id, input_path, output_dir, metadata, config)

        return {
            "job_id": job_id,
            "status": job.status,
            "status_url": str(request.url_for("get_job", job_id=job_id)),
            "result_url": str(request.url_for("get_job_result", job_id=job_id)),
        }

    @app.post("/api/uploads/init", status_code=201)
    async def init_chunked_upload(
        request: Request,
        filename: str = Form(...),
        total_size_bytes: str | None = Form(None),
    ) -> dict[str, Any]:
        """Create a chunked upload session for recordings too large for one request."""
        if not filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only .wav uploads are accepted.")
        total_size = _optional_int(total_size_bytes, "total_size_bytes")
        if total_size is not None and total_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Upload too large. Limit is {MAX_UPLOAD_MB} MB.")

        upload_id = uuid.uuid4().hex
        upload_root = API_ROOT / "chunked_uploads" / upload_id
        chunks_dir = upload_root / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "upload_id": upload_id,
            "filename": _sanitize_filename(filename),
            "original_filename": filename,
            "total_size_bytes": total_size,
            "created_at": _now_iso(),
        }
        (upload_root / "upload.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {
            "upload_id": upload_id,
            "max_chunk_mb": MAX_CHUNK_MB,
            "chunk_url_template": str(request.url_for("upload_chunk", upload_id=upload_id, chunk_index=0)).replace(
                "/0", "/{chunk_index}"
            ),
            "complete_url": str(request.url_for("complete_chunked_upload", upload_id=upload_id)),
        }

    @app.post("/api/uploads/{upload_id}/chunks/{chunk_index}", name="upload_chunk")
    async def upload_chunk(
        upload_id: str,
        chunk_index: int,
        file: UploadFile = File(...),
        content_length: int | None = Header(None),
    ) -> dict[str, Any]:
        """Upload one chunk for a chunked upload session."""
        if chunk_index < 0:
            raise HTTPException(status_code=400, detail="chunk_index must be non-negative.")
        if content_length is not None and content_length > MAX_CHUNK_BYTES:
            raise HTTPException(status_code=413, detail=f"Chunk too large. Limit is {MAX_CHUNK_MB} MB.")
        upload_root = _chunked_upload_root(upload_id)
        chunks_dir = upload_root / "chunks"
        path = chunks_dir / f"{chunk_index:08d}.part"
        size = await _save_chunk_file(file, path)
        return {"upload_id": upload_id, "chunk_index": chunk_index, "size_bytes": size}

    @app.post("/api/uploads/{upload_id}/complete", name="complete_chunked_upload", status_code=202)
    async def complete_chunked_upload(
        upload_id: str,
        request: Request,
        sensor_id: str | None = Form(None),
        sensor_latitude: str | None = Form(None),
        sensor_longitude: str | None = Form(None),
        sensor_elevation_m: str | None = Form(None),
        environment_type: str | None = Form(None),
        recording_start_time: str | None = Form(None),
        recording_timezone: str | None = Form(None),
        sample_rate: str = Form("32000"),
        min_cluster_size: str = Form("5"),
        max_events: str | None = Form(None),
        generate_spectrograms: str = Form("true"),
        enable_auto_profile: str = Form("true"),
        enable_polyphony_handling: str = Form("true"),
        enable_clusterability_filtering: str = Form("true"),
    ) -> dict[str, Any]:
        """Assemble uploaded chunks and start asynchronous processing."""
        upload_root = _chunked_upload_root(upload_id)
        upload_metadata = _read_json(upload_root / "upload.json")
        chunks = sorted((upload_root / "chunks").glob("*.part"))
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks uploaded.")

        job_id = uuid.uuid4().hex
        job_root = API_ROOT / job_id
        upload_dir = job_root / "upload"
        output_dir = job_root / "run"
        upload_dir.mkdir(parents=True, exist_ok=False)
        input_path = upload_dir / str(upload_metadata.get("filename") or "input.wav")
        saved_size = _assemble_chunks(chunks, input_path)
        expected_size = upload_metadata.get("total_size_bytes")
        if isinstance(expected_size, int) and expected_size != saved_size:
            shutil.rmtree(job_root, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Assembled size mismatch: expected {expected_size}, got {saved_size}.",
            )
        if not _file_looks_like_wav(input_path):
            shutil.rmtree(job_root, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Assembled file does not look like a RIFF/WAVE file.")

        config, metadata = _build_config(
            original_filename=str(upload_metadata.get("original_filename") or input_path.name),
            saved_size_bytes=saved_size,
            sensor_id=sensor_id,
            sensor_latitude=sensor_latitude,
            sensor_longitude=sensor_longitude,
            sensor_elevation_m=sensor_elevation_m,
            environment_type=environment_type,
            recording_start_time=recording_start_time,
            recording_timezone=recording_timezone,
            sample_rate=sample_rate,
            min_cluster_size=min_cluster_size,
            max_events=max_events,
            generate_spectrograms=generate_spectrograms,
            enable_auto_profile=enable_auto_profile,
            enable_polyphony_handling=enable_polyphony_handling,
            enable_clusterability_filtering=enable_clusterability_filtering,
        )
        metadata["chunked_upload_id"] = upload_id
        job = _create_processing_job(job_id, input_path, output_dir, metadata, config)
        shutil.rmtree(upload_root, ignore_errors=True)
        return {
            "job_id": job_id,
            "status": job.status,
            "status_url": str(request.url_for("get_job", job_id=job_id)),
            "result_url": str(request.url_for("get_job_result", job_id=job_id)),
        }

    @app.get("/api/jobs/{job_id}", name="get_job")
    def get_job(job_id: str) -> dict[str, Any]:
        return _get_job_or_404(job_id).to_dict()

    @app.get("/api/jobs/{job_id}/result", name="get_job_result")
    def get_job_result(job_id: str, request: Request) -> dict[str, Any]:
        job = _get_job_or_404(job_id)
        if job.status != "done":
            return job.to_dict()
        return _build_result_payload(job_id, request)

    @app.get("/api/jobs/{job_id}/clusters")
    def get_job_clusters(job_id: str, request: Request) -> dict[str, Any]:
        job = _get_job_or_404(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"Job is {job.status}, not done.")
        payload = _build_result_payload(job_id, request)
        return {"job_id": job_id, "clusters": payload["clusters"], "review_folders": payload["review_folders"]}

    @app.get("/api/jobs/{job_id}/events")
    def get_job_events(job_id: str, request: Request) -> dict[str, Any]:
        job = _get_job_or_404(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"Job is {job.status}, not done.")
        payload = _build_result_payload(job_id, request)
        return {"job_id": job_id, "events": payload["events"]}

    @app.get("/api/jobs/{job_id}/files/{file_path:path}", name="get_job_file")
    def get_job_file(job_id: str, file_path: str) -> Any:
        job = _get_job_or_404(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"Job is {job.status}, not done.")
        output_dir = Path(job.output_dir)
        path = _safe_output_file(output_dir, file_path)
        return FileResponse(path)

    return app


async def _save_upload_file(file: Any, path: Path) -> int:
    """Save an UploadFile in chunks, enforcing size and WAV header checks."""
    total = 0
    first = True
    with path.open("wb") as handle:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            if first:
                first = False
                if not _looks_like_wav(chunk):
                    raise HTTPException(status_code=400, detail="Uploaded file does not look like a RIFF/WAVE file.")
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"Upload too large. Limit is {MAX_UPLOAD_MB} MB.")
            handle.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return total


async def _save_chunk_file(file: Any, path: Path) -> int:
    """Save one chunk to disk without assuming it is a complete WAV file."""
    total = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_CHUNK_BYTES:
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"Chunk too large. Limit is {MAX_CHUNK_MB} MB.")
            handle.write(chunk)
    if total == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded chunk is empty.")
    return total


def _assemble_chunks(chunks: list[Path], output_path: Path) -> int:
    """Concatenate uploaded chunk files into one WAV file without loading it into memory."""
    total = 0
    with output_path.open("wb") as destination:
        for chunk_path in chunks:
            with chunk_path.open("rb") as source:
                while True:
                    data = source.read(CHUNK_SIZE)
                    if not data:
                        break
                    total += len(data)
                    if total > MAX_UPLOAD_BYTES:
                        output_path.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail=f"Upload too large. Limit is {MAX_UPLOAD_MB} MB.")
                    destination.write(data)
    return total


def _file_looks_like_wav(path: Path) -> bool:
    with path.open("rb") as handle:
        return _looks_like_wav(handle.read(16))


def _create_processing_job(
    job_id: str,
    input_path: Path,
    output_dir: Path,
    metadata: dict[str, Any],
    config: BioSoundConfig,
) -> ApiJob:
    job = ApiJob(
        job_id=job_id,
        status="queued",
        created_at=_now_iso(),
        updated_at=_now_iso(),
        input_path=str(input_path),
        output_dir=str(output_dir),
        metadata={"request": metadata, "config": config_to_dict(config)},
    )
    _store_job(job)
    _executor.submit(_run_job, job_id, input_path, output_dir, config)
    return job


def _build_config(
    *,
    original_filename: str,
    saved_size_bytes: int,
    sensor_id: str | None,
    sensor_latitude: str | None,
    sensor_longitude: str | None,
    sensor_elevation_m: str | None,
    environment_type: str | None,
    recording_start_time: str | None,
    recording_timezone: str | None,
    sample_rate: str,
    min_cluster_size: str,
    max_events: str | None,
    generate_spectrograms: str,
    enable_auto_profile: str,
    enable_polyphony_handling: str,
    enable_clusterability_filtering: str,
) -> tuple[BioSoundConfig, dict[str, Any]]:
    metadata = {
        "sensor_id": _blank_to_none(sensor_id),
        "sensor_latitude": _optional_float(sensor_latitude, "sensor_latitude"),
        "sensor_longitude": _optional_float(sensor_longitude, "sensor_longitude"),
        "sensor_elevation_m": _optional_float(sensor_elevation_m, "sensor_elevation_m"),
        "environment_type": _blank_to_none(environment_type),
        "recording_start_time": _blank_to_none(recording_start_time),
        "recording_timezone": _blank_to_none(recording_timezone),
        "saved_size_bytes": saved_size_bytes,
        "original_filename": original_filename,
    }
    config_values = {
        "sample_rate": _required_int(sample_rate, "sample_rate"),
        "min_cluster_size": _required_int(min_cluster_size, "min_cluster_size"),
        "max_events": _optional_int(max_events, "max_events"),
        "generate_spectrograms": _parse_bool(generate_spectrograms, "generate_spectrograms"),
        "export_clips": True,
        "enable_auto_profile": _parse_bool(enable_auto_profile, "enable_auto_profile"),
        "enable_polyphony_handling": _parse_bool(enable_polyphony_handling, "enable_polyphony_handling"),
        "enable_clusterability_filtering": _parse_bool(
            enable_clusterability_filtering, "enable_clusterability_filtering"
        ),
        "sensor_id": metadata["sensor_id"],
        "sensor_latitude": metadata["sensor_latitude"],
        "sensor_longitude": metadata["sensor_longitude"],
        "sensor_elevation_m": metadata["sensor_elevation_m"],
        "environment_type": metadata["environment_type"],
        "recording_start_time": metadata["recording_start_time"],
        "recording_timezone": metadata["recording_timezone"],
    }
    try:
        return BioSoundConfig(**config_values), metadata
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _run_job(job_id: str, input_path: Path, output_dir: Path, config: BioSoundConfig) -> None:
    _update_job(job_id, status="running")
    try:
        result = process_audio_file(input_path, output_dir, config=config)
        _update_job(job_id, status="done", result=result.to_dict())
    except Exception as exc:  # pragma: no cover - defensive API boundary
        LOGGER.exception("API job failed: %s", job_id)
        _update_job(job_id, status="failed", error=str(exc))


def _build_result_payload(job_id: str, request: Any) -> dict[str, Any]:
    job = _get_job_or_404(job_id)
    output_dir = Path(job.output_dir)
    run_metadata = _read_json(output_dir / "run_metadata.json")
    event_metadata = _read_json(output_dir / "event_metadata.json")
    events_df = _read_csv(output_dir / "events.csv")
    clusters_df = _read_csv(output_dir / "clusters.csv")
    events = _event_records(events_df, event_metadata, job_id, request)
    clusters = _cluster_records(clusters_df, events)
    review_folders = _review_folder_records(events)
    return {
        "job": job.to_dict(),
        "run_metadata": run_metadata,
        "event_metadata_url": _file_url(request, job_id, "event_metadata.json"),
        "events_csv_url": _file_url(request, job_id, "events.csv"),
        "clusters_csv_url": _file_url(request, job_id, "clusters.csv"),
        "report_url": _file_url(request, job_id, "report.md"),
        "index_url": _file_url(request, job_id, "index.html"),
        "clusters": clusters,
        "review_folders": review_folders,
        "events": events,
    }


def _event_records(events_df: pd.DataFrame, event_metadata: dict[str, Any], job_id: str, request: Any) -> list[dict[str, Any]]:
    metadata_by_event = {
        item.get("event_id"): item
        for item in event_metadata.get("events", [])
        if isinstance(item, dict)
    }
    records: list[dict[str, Any]] = []
    for row in events_df.to_dict(orient="records"):
        event_id = row.get("event_id")
        metadata = metadata_by_event.get(event_id, {})
        record = _json_clean(row)
        for key in ("clip_path", "spectrogram_path", "context_clip_path"):
            value = row.get(key)
            if isinstance(value, str) and value and value != "nan":
                record[f"{key}_url"] = _file_url(request, job_id, value)
        clip_path = row.get("clip_path")
        if isinstance(clip_path, str) and clip_path and clip_path != "nan":
            sidecar = str(Path(clip_path).with_suffix(".json").as_posix())
            record["metadata_url"] = _file_url(request, job_id, sidecar)
        record["metadata"] = metadata
        records.append(record)
    return records


def _cluster_records(clusters_df: pd.DataFrame, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events_by_cluster: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        cluster_id = event.get("cluster_id")
        if cluster_id is None or pd.isna(cluster_id):
            continue
        try:
            events_by_cluster.setdefault(int(cluster_id), []).append(event)
        except (TypeError, ValueError):
            continue

    clusters: list[dict[str, Any]] = []
    for row in clusters_df.to_dict(orient="records"):
        record = _json_clean(row)
        cluster_id = int(row["cluster_id"])
        record["events"] = events_by_cluster.get(cluster_id, [])
        record["representatives"] = [
            event
            for event in record["events"]
            if event.get("event_id") in str(row.get("representative_event_ids", "")).split(",")
        ]
        clusters.append(record)
    return clusters


def _review_folder_records(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    review_sources = {"mixed", "low_confidence_noise", "ambiguous_review", "short_review"}
    grouped: dict[str, list[dict[str, Any]]] = {source: [] for source in review_sources}
    for event in events:
        source = str(event.get("source_type") or "")
        if source in grouped:
            grouped[source].append(event)
    return grouped


def _get_job_or_404(job_id: str) -> ApiJob:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        disk_job = _read_job_status(API_ROOT / job_id / "job_status.json")
        if disk_job is not None:
            _store_job(disk_job)
            return disk_job
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def _store_job(job: ApiJob) -> None:
    with _jobs_lock:
        _jobs[job.job_id] = job
    _write_job_status(job)


def _update_job(
    job_id: str,
    *,
    status: str,
    error: str | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.status = status
        job.updated_at = _now_iso()
        if error is not None:
            job.error = error
        if result is not None:
            job.result = result
    _write_job_status(job)


def _write_job_status(job: ApiJob) -> None:
    path = API_ROOT / job.job_id / "job_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")


def _read_job_status(path: Path) -> ApiJob | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return ApiJob(**data)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _chunked_upload_root(upload_id: str) -> Path:
    if not upload_id or not all(char.isalnum() for char in upload_id):
        raise HTTPException(status_code=404, detail="Upload not found.")
    root = (API_ROOT / "chunked_uploads" / upload_id).resolve()
    uploads_root = (API_ROOT / "chunked_uploads").resolve()
    if not _is_relative_to(root, uploads_root) or not root.is_dir():
        raise HTTPException(status_code=404, detail="Upload not found.")
    return root


def _safe_output_file(output_dir: Path, file_path: str) -> Path:
    root = output_dir.resolve()
    target = (root / file_path).resolve()
    if not _is_relative_to(target, root) or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return target


def _file_url(request: Any, job_id: str, relative_path: str) -> str:
    return str(request.url_for("get_job_file", job_id=job_id, file_path=relative_path))


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _looks_like_wav(chunk: bytes) -> bool:
    return len(chunk) >= 12 and chunk[:4] == b"RIFF" and chunk[8:12] == b"WAVE"


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in name)
    if safe in {"", ".", ".."}:
        safe = "input"
    if not safe.lower().endswith(".wav"):
        safe += ".wav"
    if safe == ".wav":
        safe = "input.wav"
    return safe


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _optional_float(value: str | None, field_name: str) -> float | None:
    value = _blank_to_none(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a number.") from exc


def _required_int(value: str, field_name: str) -> int:
    value = _blank_to_none(value)
    if value is None:
        raise HTTPException(status_code=400, detail=f"{field_name} is required.")
    try:
        return int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be an integer.") from exc


def _optional_int(value: str | None, field_name: str) -> int | None:
    value = _blank_to_none(value)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be an integer.") from exc


def _parse_bool(value: str, field_name: str) -> bool:
    value = (_blank_to_none(value) or "").lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise HTTPException(status_code=400, detail=f"{field_name} must be a boolean.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _cors_origins() -> list[str]:
    raw = os.environ.get("BIOSOUND_API_CORS_ORIGINS", "*")
    return ["*"] if raw.strip() == "*" else [item.strip() for item in raw.split(",") if item.strip()]


def _with_cors(asgi_app: Any) -> Any:
    return CORSMiddleware(
        asgi_app,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    """Run the API with uvicorn."""
    if FastAPI is None:
        raise RuntimeError("FastAPI dependencies are not installed. Run `pip install -e .` again.")
    import uvicorn

    configure_logging(verbose=True)
    host = os.environ.get("BIOSOUND_API_HOST", "127.0.0.1")
    port = int(os.environ.get("BIOSOUND_API_PORT", "8000"))
    uvicorn.run("biosound_cluster.api:app", host=host, port=port, reload=False)


fastapi_app = create_app() if FastAPI is not None else None
app = _with_cors(fastapi_app) if fastapi_app is not None else None


if __name__ == "__main__":
    main()
