"""Local-only CBCT dataset and optional AI segmentation service.

The browser never downloads a neural-network model. It asks this service, running
on the same computer, to open a DICOM series once and keeps a dataset id for the
viewer session. No patient volume is sent to an external server.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

import onnx_infer
import mps_infer
from sync import HEARTBEAT_SECONDS, EventBus, FolderPickBroker, sse

HERE = Path(__file__).parent
ENGINE = os.environ.get("TS_ENGINE", "auto").lower()
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cbct-ai")

# Viewer and manager compare these to detect a restarted or incompatible service.
API_VERSION = 2
INSTANCE_ID = uuid.uuid4().hex


def _code_version() -> str:
    """Short fingerprint of the running backend code, so a test run can be tied to the exact code that produced it."""
    digest = hashlib.sha1()
    if getattr(sys, "frozen", False):
        info = Path(sys.executable).stat()
        digest.update(f"{info.st_size}:{int(info.st_mtime)}".encode())
    else:
        here = Path(__file__).resolve().parent
        for name in sorted(p.name for p in here.glob("*.py") if not p.name.startswith("test_")):
            digest.update(name.encode()); digest.update((here / name).read_bytes())
    return digest.hexdigest()[:8]


CODE_VERSION = _code_version()
STARTED_AT = time.time()
# Electron passes a random per-launch token; control endpoints are disabled without it.
CONTROL_TOKEN = os.environ.get("PINECLINIC_CONTROL_TOKEN", "")
# Datasets are released this long after the last viewer disconnects.
IDLE_RELEASE_SECONDS = float(os.environ.get("PINECLINIC_IDLE_RELEASE", str(30 * 60)))
BUS = EventBus()
PICKS = FolderPickBroker()
SERVER: object | None = None  # uvicorn.Server, set by __main__ (or tests) for graceful shutdown


@dataclass
class Dataset:
    image: sitk.Image
    fingerprint: str
    source: str
    created_at: float = field(default_factory=time.time)


@dataclass
class Job:
    dataset_id: str
    state: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = "queued"
    progress: float = 0.0
    detail: str = ""
    provider: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    mask: np.ndarray | None = None
    cancel_requested: bool = False


DATASETS: dict[str, Dataset] = {}
JOBS: dict[str, Job] = {}
STORE_LOCK = threading.Lock()


def dataset_summary(dataset_id: str, dataset: Dataset) -> dict[str, object]:
    return {
        "datasetId": dataset_id, "source": dataset.source, "openedAt": dataset.created_at,
        "dimensions": list(dataset.image.GetSize()), "spacing": list(dataset.image.GetSpacing()),
    }


def job_summary(job_id: str, job: Job) -> dict[str, object]:
    return {
        "jobId": job_id, "datasetId": job.dataset_id, "state": job.state, "stage": job.stage,
        "progress": job.progress, "detail": job.detail, "provider": job.provider,
        "startedAt": job.started_at, "finishedAt": job.finished_at,
    }


def publish_job(job_id: str, job: Job, *, progress_only: bool = False) -> None:
    """Broadcast a job snapshot. Progress ticks are throttled; state changes never are."""
    with STORE_LOCK:
        data = job_summary(job_id, job)
    if progress_only:
        BUS.publish("job.progress", data, throttle_key=job_id)
    else:
        BUS.publish(f"job.{job.state}", data)


def snapshot() -> dict[str, object]:
    engine, provider = selected_engine()
    with STORE_LOCK:
        datasets = [dataset_summary(i, d) for i, d in DATASETS.items()]
        jobs = [job_summary(i, j) for i, j in JOBS.items()]
    return {
        "instanceId": INSTANCE_ID, "apiVersion": API_VERSION, "codeVersion": CODE_VERSION, "startedAt": STARTED_AT,
        "engine": engine, "provider": provider, "mpsAvailable": onnx_infer.mps_available(),
        "datasets": datasets, "jobs": jobs, "clients": BUS.counts(),
    }


def require_control(token: str | None) -> None:
    if not CONTROL_TOKEN or not token or not hmac.compare_digest(token, CONTROL_TOKEN):
        raise HTTPException(403, "Control endpoint requires the service manager token")


def release_dataset(dataset_id: str, reason: str) -> bool:
    with STORE_LOCK:
        dataset = DATASETS.pop(dataset_id, None)
        released = {job_id: job for job_id, job in JOBS.items() if job.dataset_id == dataset_id}
        interrupted = []
        for job_id, job in released.items():
            job.cancel_requested = True
            if job.state in {"queued", "running"}:
                job.state, job.stage, job.finished_at = "cancelled", "dataset released", time.time()
                interrupted.append(job_id)
            job.mask = None
            JOBS.pop(job_id, None)
    if not dataset:
        return False
    for job_id in interrupted:
        BUS.publish("job.cancelled", job_summary(job_id, released[job_id]))
    BUS.publish("dataset.released", {"datasetId": dataset_id, "reason": reason})
    log(f"released dataset {dataset_id} ({reason})")
    return True


async def idle_release_loop() -> None:
    while True:
        await asyncio.sleep(60)
        if BUS.counts()["viewers"] == 0 and time.time() - BUS.last_viewer_seen > IDLE_RELEASE_SECONDS:
            with STORE_LOCK:
                ids = list(DATASETS)
            for dataset_id in ids:
                release_dataset(dataset_id, "idle")


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(idle_release_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="PineClinic Local CBCT Service", lifespan=lifespan)


@app.middleware("http")
async def no_result_storage(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


app.add_middleware(
    CORSMiddleware,
    # Electron's packaged renderer has the opaque `null` origin. The service
    # still binds only to loopback, so images never leave this computer.
    allow_origins=["http://localhost:5180", "http://127.0.0.1:5180", "null"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Nx", "X-Ny", "X-Nz", "X-Sx", "X-Sy", "X-Sz", "X-Origin", "X-Direction", "Content-Length"],
)


class PathRequest(BaseModel):
    path: str


class PickResult(BaseModel):
    path: str | None = None


def log(*args: object) -> None:
    print("[cbct]", *args, file=sys.stderr, flush=True)


def read_series(folder: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(folder))
    if not ids:
        raise HTTPException(400, "No DICOM series found in the selected folder")
    best = max(ids, key=lambda series: len(reader.GetGDCMSeriesFileNames(str(folder), series)))
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(folder), best))
    return reader.Execute()


def load_path(path: str) -> sitk.Image:
    p = Path(path).expanduser()
    if not p.exists():
        raise HTTPException(404, f"Path not found: {p}")
    # SimpleITK preserves the image geometry for these single-file volume
    # formats. Treating them as DICOM directories would otherwise hide a
    # valid dataset behind the misleading "No DICOM series" error.
    if p.is_file() and p.name.lower().endswith((
        ".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd",
    )):
        return sitk.ReadImage(str(p))
    return read_series(p if p.is_dir() else p.parent)


def path_fingerprint(path: str) -> str:
    p = Path(path).expanduser()
    files = sorted(p.iterdir()) if p.is_dir() else [p]
    digest = hashlib.sha256()
    for file in files:
        if file.is_file():
            stat = file.stat()
            digest.update(f"{file.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def register_dataset(image: sitk.Image, fingerprint: str, source: str) -> tuple[str, Dataset]:
    with STORE_LOCK:
        dataset_id = uuid.uuid4().hex
        dataset = Dataset(image=image, fingerprint=fingerprint, source=source)
        DATASETS[dataset_id] = dataset
    BUS.publish("dataset.opened", dataset_summary(dataset_id, dataset))
    return dataset_id, dataset


def get_dataset(dataset_id: str) -> Dataset:
    with STORE_LOCK:
        dataset = DATASETS.get(dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset is no longer available. Open the series again.")
    return dataset


def image_metadata(image: sitk.Image, dataset_id: str | None = None) -> dict[str, object]:
    nx, ny, nz = image.GetSize()
    sx, sy, sz = image.GetSpacing()
    result: dict[str, object] = {
        "dimensions": [nx, ny, nz],
        "spacing": [sx, sy, sz],
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }
    if dataset_id:
        result["datasetId"] = dataset_id
    return result


def volume_response(image: sitk.Image) -> Response:
    arr = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkInt16))  # z, y, x; x fastest on wire
    nx, ny, nz = image.GetSize()
    sx, sy, sz = image.GetSpacing()
    origin = ",".join(str(value) for value in image.GetOrigin())
    direction = ",".join(str(value) for value in image.GetDirection())
    return Response(
        content=np.ascontiguousarray(arr).tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Nx": str(nx), "X-Ny": str(ny), "X-Nz": str(nz),
            "X-Sx": str(sx), "X-Sy": str(sy), "X-Sz": str(sz),
            "X-Origin": origin, "X-Direction": direction,
        },
    )


def mask_response(mask: np.ndarray) -> Response:
    nz, ny, nx = mask.shape
    return Response(
        content=np.ascontiguousarray(mask).tobytes(),
        media_type="application/octet-stream",
        headers={"X-Nx": str(nx), "X-Ny": str(ny), "X-Nz": str(nz)},
    )


def selected_engine() -> tuple[str, str]:
    # Core ML is not used for this 3D graph. Apple Silicon uses the source
    # checkpoint with PyTorch/MPS; Windows can use an installed ONNX GPU EP.
    if ENGINE in {"auto", "mps"} and mps_infer.available():
        return "mps", "PyTorch MPS"
    if ENGINE in {"auto", "onnx"} and onnx_infer.available():
        return "onnx", onnx_infer.provider_name()
    return "unavailable", "none"


def run_segmentation(job_id: str) -> None:
    """Worker entry point: run the job and always broadcast its final state."""
    try:
        _run_segmentation(job_id)
    finally:
        with STORE_LOCK:
            job = JOBS.get(job_id)
        if job:
            publish_job(job_id, job)


def _run_segmentation(job_id: str) -> None:
    with STORE_LOCK:
        job = JOBS[job_id]
        dataset = DATASETS.get(job.dataset_id)
        if job.cancel_requested:
            job.state, job.stage, job.finished_at = "cancelled", "cancelled", time.time()
            return
        if not dataset:
            job.state, job.detail, job.finished_at = "failed", "Dataset no longer exists", time.time()
            return
        job.state, job.stage, job.started_at = "running", "preparing volume", time.time()
    engine, provider = selected_engine()
    job.provider = provider
    publish_job(job_id, job)
    if engine == "unavailable":
        with STORE_LOCK:
            job.state, job.detail, job.finished_at = "failed", "No local segmentation engine is installed", time.time()
        return
    try:
        def progress(done: int, total: int, elapsed: float) -> None:
            with STORE_LOCK:
                job.stage = f"AI tile {done}/{total}"
                job.progress = done / max(total, 1)
            publish_job(job_id, job, progress_only=True)

        def cancelled() -> bool:
            with STORE_LOCK:
                return job.cancel_requested

        infer = mps_infer.segment_mandible if engine == "mps" else onnx_infer.segment_mandible
        mask = infer(dataset.image, log=log, on_progress=progress, is_cancelled=cancelled)
        with STORE_LOCK:
            if job.cancel_requested:
                job.state, job.stage, job.finished_at = "cancelled", "cancelled", time.time()
                return
        with STORE_LOCK:
            job.mask, job.state, job.stage, job.progress, job.finished_at = mask, "done", "complete", 1.0, time.time()
    except onnx_infer.SegmentationCancelled:
        with STORE_LOCK:
            job.state, job.stage, job.finished_at = "cancelled", "cancelled", time.time()
    except Exception as exc:  # surface a concise error to the local UI
        log("segmentation failed", repr(exc))
        with STORE_LOCK:
            job.state, job.detail, job.finished_at = "failed", str(exc)[:500], time.time()


@app.get("/api/health")
def health() -> dict[str, object]:
    engine, provider = selected_engine()
    return {
        "status": "ok", "localOnly": True, "engine": engine, "provider": provider,
        "resultCacheEnabled": False,
        "mpsAvailable": onnx_infer.mps_available(),
        "instanceId": INSTANCE_ID, "apiVersion": API_VERSION, "codeVersion": CODE_VERSION, "startedAt": STARTED_AT,
        "clients": BUS.counts(),
        "note": "The browser never receives a model. MPS uses the local PyTorch checkpoint; ONNX uses its selected local provider.",
    }


@app.get("/api/events")
async def events(request: Request, role: str = "viewer", x_control_token: str | None = Header(None)) -> StreamingResponse:
    """Live state stream shared by the viewer and the Electron service manager."""
    if role not in {"viewer", "manager"}:
        raise HTTPException(400, "role must be viewer or manager")
    if role == "manager":
        # Only the manager may answer folder-picker requests or see them.
        require_control(x_control_token)
    subscriber_id, queue = BUS.subscribe(role)
    BUS.publish("clients", BUS.counts())

    async def stream():
        try:
            yield sse("hello", snapshot())
            while True:
                try:
                    frame = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        break
                    frame = sse("heartbeat", {"t": time.time(), "instanceId": INSTANCE_ID})
                yield frame
                if frame.startswith("event: service.stopping"):
                    break
        finally:
            BUS.unsubscribe(subscriber_id)
            BUS.publish("clients", BUS.counts())

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/pick-folder")
async def pick_folder() -> JSONResponse:
    """Ask the Electron manager to show a native folder dialog for the viewer."""
    if BUS.counts()["managers"] == 0:
        return JSONResponse({"reason": "no-manager"}, status_code=409)
    request_id = uuid.uuid4().hex
    future = PICKS.create(request_id)
    BUS.publish("pick-folder.request", {"requestId": request_id}, roles={"manager"})
    try:
        result = await asyncio.wait_for(future, 300)
    except asyncio.TimeoutError:
        result = {"cancelled": True}
    finally:
        PICKS.discard(request_id)
    return JSONResponse(result)


@app.post("/api/control/pick-folder/{request_id}")
async def resolve_pick_folder(request_id: str, body: PickResult, x_control_token: str | None = Header(None)) -> dict[str, bool]:
    require_control(x_control_token)
    result: dict[str, object] = {"path": body.path} if body.path else {"cancelled": True}
    return {"delivered": PICKS.resolve(request_id, result)}


def request_exit() -> None:
    if SERVER is not None:
        SERVER.should_exit = True  # type: ignore[attr-defined]
    else:
        os._exit(0)


@app.post("/api/control/shutdown")
async def control_shutdown(x_control_token: str | None = Header(None)) -> dict[str, str]:
    require_control(x_control_token)
    with STORE_LOCK:
        active = [(job_id, job) for job_id, job in JOBS.items() if job.state in {"queued", "running"}]
        for _, job in active:
            job.cancel_requested = True
    BUS.publish("service.stopping", {"reason": "user", "instanceId": INSTANCE_ID})
    log("shutdown requested by the service manager")
    # Give the SSE streams a moment to deliver `service.stopping` first.
    asyncio.get_running_loop().call_later(0.3, request_exit)
    return {"state": "stopping"}


@app.post("/api/datasets/path")
def open_dataset_path(request: PathRequest) -> JSONResponse:
    fingerprint = path_fingerprint(request.path)
    # Each open reads the requested source anew; prior case data and AI results
    # are never substituted for this load.
    image = load_path(request.path)
    dataset_id, dataset = register_dataset(image, fingerprint, str(Path(request.path).expanduser()))
    return JSONResponse({**image_metadata(dataset.image, dataset_id), "source": dataset.source})


@app.post("/api/datasets/upload")
async def upload_dataset(files: list[UploadFile] = File(...)) -> JSONResponse:
    if not files:
        raise HTTPException(400, "Select at least one DICOM file")
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="pineclinic-upload-") as temp:
        folder = Path(temp)
        for index, upload in enumerate(files):
            name = Path(upload.filename or f"slice-{index}").name
            data = await upload.read()
            digest.update(name.encode()); digest.update(data)
            (folder / name).write_bytes(data)
        image = read_series(folder)
    dataset_id, dataset = register_dataset(image, digest.hexdigest(), "uploaded DICOM series")
    return JSONResponse({**image_metadata(dataset.image, dataset_id), "source": dataset.source})


@app.get("/api/datasets/{dataset_id}/volume")
def get_volume(dataset_id: str) -> Response:
    return volume_response(get_dataset(dataset_id).image)


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, bool]:
    return {"released": release_dataset(dataset_id, "viewer")}


@app.post("/api/datasets/{dataset_id}/jobs/segment")
def start_segmentation(dataset_id: str) -> dict[str, object]:
    get_dataset(dataset_id)
    job_id = uuid.uuid4().hex
    with STORE_LOCK:
        job = JOBS[job_id] = Job(dataset_id=dataset_id)
    publish_job(job_id, job)
    EXECUTOR.submit(run_segmentation, job_id)
    return {"jobId": job_id, "state": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    with STORE_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job_summary(job_id, job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, str]:
    with STORE_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        job.cancel_requested = True
        queued = job.state == "queued"
        if queued:
            job.state, job.stage = "cancelled", "cancelled"
    if queued:
        publish_job(job_id, job)
    else:
        BUS.publish("job.progress", {**job_summary(job_id, job), "stage": "cancelling"})
    return {"state": "cancelling"}


@app.get("/api/jobs/{job_id}/mask")
def get_job_mask(job_id: str) -> Response:
    with STORE_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.state != "done" or job.mask is None:
            raise HTTPException(409, f"Mask is not ready ({job.state})")
        mask = job.mask
    return mask_response(mask)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="PineClinic local CBCT service")
    parser.add_argument("--host", default=os.environ.get("PINECLINIC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PINECLINIC_PORT", "8000")))
    args = parser.parse_args()
    log(f"Starting local-only service at http://{args.host}:{args.port} (instance {INSTANCE_ID[:8]}, api v{API_VERSION}, code {CODE_VERSION})")
    # Open SSE streams must not hold the process open after a stop request.
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="info", timeout_graceful_shutdown=2))
    SERVER = server

    manager_pid = int(os.environ.get("PINECLINIC_MANAGER_PID", "0") or 0)
    if manager_pid and os.name == "posix":
        def _watch_manager() -> None:
            while True:
                time.sleep(2)
                try:
                    os.kill(manager_pid, 0)
                except ProcessLookupError:
                    log("Manager process is gone; shutting down.")
                    os._exit(0)
                except PermissionError:
                    pass
        threading.Thread(target=_watch_manager, daemon=True).start()
    server.run()
    # The AI worker checks its cancel flag per tile; do not wait for a long
    # inference to finish once the manager has asked the service to stop.
    os._exit(0)
