"""FastAPI job endpoints. Local capability IDs match the existing app's isolation model."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import traceback
from threading import BoundedSemaphore, Event, RLock
from uuid import uuid4
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .pretrained import model_info
from .runner import execute
from .schemas import BenchmarkConfig, BenchmarkResults, ConvergenceResults, ModelInfo, RunCreated, RunStatus
from .storage import directory, root, save_json

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="benchmark")
_capacity = BoundedSemaphore(1)
_lock = RLock()
_active = set()
_stopping = Event()


class BenchmarkInterrupted(Exception):
    """Cooperative stop between bounded provider calls."""


def startup():
    _stopping.clear()


def shutdown():
    _stopping.set()


@asynccontextmanager
async def lifespan(app):
    startup()
    try:
        yield
    finally:
        shutdown()


router = APIRouter(prefix="/benchmark", tags=["Optimization Benchmark"], lifespan=lifespan)


def failure_detail(error: Exception) -> str:
    """Safe diagnostics without provider messages, prompts or credential values."""
    if isinstance(error, PermissionError):
        filename = getattr(error, "filename2", None) or getattr(error, "filename", None)
        file_hint = f" File: {Path(filename).name}." if filename else ""
        code = getattr(error, "winerror", None)
        code_hint = f" Windows error {code}." if code is not None else ""
        return (f"PermissionError: a file or OS operation was denied.{file_hint}{code_hint} "
                "A Windows sharing lock, OneDrive sync, antivirus, or folder permissions may be responsible. "
                "Partial artifacts are downloadable.")
    return f"{type(error).__name__}: benchmark stopped. Check credentials, dependencies and pinned model files; partial artifacts are downloadable."


def _directory(run_id):
    try:
        return directory(run_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Benchmark not found.") from error


def _read_status(path):
    with _lock:
        try:
            status = RunStatus.model_validate_json((path / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=500, detail="Could not read benchmark progress.") from error
        if status.status in ("running", "queued") and status.run_id not in _active:
            status.status = "interrupted"
            status.phase = "Interrupted by backend restart; partial artifacts remain downloadable"
            save_json(path / "status.json", status.model_dump(mode="json"))
        return status


def _job(path, config, model_names):
    def progress(completed_method=None, **updates):
        if _stopping.is_set():
            raise BenchmarkInterrupted()
        with _lock:
            status = RunStatus.model_validate_json((path / "status.json").read_text(encoding="utf-8"))
            status.status = "running"
            if completed_method:
                status.completed_evaluations += 1
                status.method_completed[completed_method] += 1
            for name, value in updates.items():
                setattr(status, name, value)
            save_json(path / "status.json", status.model_dump(mode="json"))
    try:
        execute(path, config, progress, model_names)
        progress(phase="Complete")
        with _lock:
            status = RunStatus.model_validate_json((path / "status.json").read_text(encoding="utf-8"))
            status.status = "completed"
            save_json(path / "status.json", status.model_dump(mode="json"))
    except Exception as error:
        # Persist only error class; provider error strings can contain credentials/prompts.
        locations = " -> ".join(f"{Path(frame.filename).name}:{frame.lineno} ({frame.name})"
                                for frame in traceback.extract_tb(error.__traceback__)[-5:])
        logger.error("Benchmark failed: %s; locations: %s", failure_detail(error), locations)
        with _lock:
            status = RunStatus.model_validate_json((path / "status.json").read_text(encoding="utf-8"))
            status.status = "interrupted" if isinstance(error, BenchmarkInterrupted) else "failed"
            status.phase = "Interrupted" if isinstance(error, BenchmarkInterrupted) else f"Failed during: {status.phase}"
            status.error = ("Backend stopped; partial artifacts are downloadable." if isinstance(error, BenchmarkInterrupted) else
                failure_detail(error))
            save_json(path / "status.json", status.model_dump(mode="json"))
    finally:
        with _lock:
            _active.discard(path.name)
        _capacity.release()


@router.get("/models", response_model=list[ModelInfo])
def models():
    return [model_info()]


@router.post("/run", response_model=RunCreated, status_code=202)
def start(config: BenchmarkConfig):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=422, detail="Set OPENAI_API_KEY before running an adaptive LLM benchmark.")
    if not model_info().available:
        raise HTTPException(status_code=422, detail="Pretrained artifact or dependencies unavailable. Install project dependencies and run scripts/prepare_benchmark_model.py.")
    if not _capacity.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="One benchmark is already running. Wait until it finishes.")
    run_id = str(uuid4())
    path = root() / run_id
    try:
        path.mkdir(parents=True)
        model_names = {"eda": os.getenv("EDA_MODEL", "openai:gpt-4.1-mini"),
            "selection": os.getenv("BO_MODEL", os.getenv("EDA_MODEL", "openai:gpt-4.1-mini")),
            "llm_optimizer": os.getenv("BENCHMARK_LLM_MODEL", os.getenv("EDA_MODEL", "openai:gpt-4.1-mini"))}
        save_json(path / "config.json", {**config.model_dump(mode="json"), "model_names": model_names,
                  "methods": ["llm_optimizer", "agent_selected_bo"]})
        status = RunStatus(run_id=run_id, name=config.name, status="queued",
            total_evaluations=config.optimization_iterations * config.random_seeds * 2)
        save_json(path / "status.json", status.model_dump(mode="json"))
        with _lock:
            _active.add(run_id)
        _executor.submit(_job, path, config, model_names)
    except Exception as error:
        with _lock:
            _active.discard(run_id)
        _capacity.release()
        raise HTTPException(status_code=500, detail="Could not start the benchmark.") from error
    return RunCreated(run_id=run_id)


@router.get("/{run_id}/status", response_model=RunStatus)
def status(run_id: str):
    return _read_status(_directory(run_id))


def _summary(run_id):
    path = _directory(run_id)
    if _read_status(path).status != "completed":
        raise HTTPException(status_code=409, detail="Benchmark results are not complete yet.")
    return path, json.loads((path / "comparison_summary.json").read_text(encoding="utf-8"))


@router.get("/{run_id}/results", response_model=BenchmarkResults)
def results(run_id: str) -> dict:
    path, summary = _summary(run_id)
    return {"summary": summary, "config": json.loads((path / "config.json").read_text(encoding="utf-8")),
            "iterations": json.loads((path / "iterations.json").read_text(encoding="utf-8"))}


@router.get("/{run_id}/convergence", response_model=ConvergenceResults)
def convergence(run_id: str) -> dict:
    _, summary = _summary(run_id)
    return {"curves": summary["convergence"], "reference": summary["reference"]}


@router.get("/{run_id}/download")
def download(run_id: str, artifact: str = "all"):
    path = _directory(run_id)
    allowed = {"summary": ("comparison_summary.json", "application/json"),
               "llm": ("llm_results.csv", "text/csv"), "bo": ("bo_results.csv", "text/csv")}
    if artifact != "all":
        if artifact not in allowed:
            raise HTTPException(status_code=422, detail="Choose all, summary, llm or bo.")
        filename, media = allowed[artifact]
        file = path / filename
        if not file.is_file():
            raise HTTPException(status_code=404, detail="Artifact not available yet.")
        return Response(file.read_bytes(), media_type=media,
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in path.rglob("*"):
            if file.is_file() and file.suffix in (".json", ".csv"):
                archive.writestr(file.relative_to(path).as_posix(), file.read_bytes())
    return Response(buffer.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="benchmark_{run_id}.zip"'})
