"""HTTP adapter for the chat interface."""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .agent import load_csv, run_eda_detailed
from .outputs import markdown_filename, plots_zip_data_uri, render_markdown_report
from .plots import render_plot_images
from .bo import BOProblem, column_metadata, validate_problem
from .sessions import session_directory, storage_root
from .bo_analyzer import analyze_bo_problem, problem_fingerprint
from .bo_profile import BOAnalyzerConfig, BOProfile
from .bo_service import BO_ARTIFACT_LOCK, generate_session_profile, write_artifact
from .surrogates import SurrogateReport, recommend_surrogate, profile_fingerprint
from .benchmark.api import router as benchmark_router

load_dotenv()

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
logger = logging.getLogger(__name__)

app = FastAPI(title="Object EDA API", version="0.1.0")
app.include_router(benchmark_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    """Analyze one temporary CSV and return the validated report."""
    filename = Path(file.filename or "dataset.csv").name
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Please upload a .csv file.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The CSV must be smaller than 15 MB.")

    model_name = os.getenv("EDA_MODEL", "openai:gpt-4.1-mini")
    try:
        with TemporaryDirectory(prefix="object-eda-") as temp_dir:
            csv_path = Path(temp_dir) / filename
            output_path = Path(temp_dir) / "eda_report.json"
            csv_path.write_bytes(content)
            report, analysis_trace = await run_in_threadpool(
                run_eda_detailed, csv_path, output_path, model_name
            )
            frame = await run_in_threadpool(load_csv, csv_path)
            plot_images = await run_in_threadpool(render_plot_images, frame, report.plot_plan)
            session_id = str(uuid4())
            directory = storage_root() / session_id
            directory.mkdir(parents=True)
            (directory / "dataset.csv").write_bytes(content)
            (directory / "eda_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
            return {
                "session_id": session_id,
                "dataset_columns": column_metadata(frame),
                "report": report.model_dump(mode="json"),
                "plot_images": plot_images,
                "markdown_report": {
                    "filename": markdown_filename(report.source_file),
                    "content": render_markdown_report(report),
                },
                "plots_archive": {
                    "filename": "eda_plots.zip",
                    "data_uri": plots_zip_data_uri(plot_images),
                },
                "analysis_trace": analysis_trace,
            }
    except (UnicodeDecodeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Could not read this CSV: {error}") from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check the API key and model configuration, then try again.",
        ) from error


@app.get("/analyses/{session_id}/bo-setup")
def bo_setup(session_id: str) -> dict:
    directory = session_directory(session_id)
    report = json.loads((directory / "eda_report.json").read_text(encoding="utf-8"))
    return {"session_id": session_id, "dataset_columns": column_metadata(load_csv(directory / "dataset.csv")),
            "target_recommendations": report.get("target_recommendations", [])}


@app.post("/analyses/{session_id}/bo-problem", response_model=BOProblem)
def save_bo_problem(session_id: str, problem: BOProblem) -> BOProblem:
    directory = session_directory(session_id)
    try:
        with BO_ARTIFACT_LOCK:
            frame = load_csv(directory / "dataset.csv")
            validate_problem(problem, frame)
            report = json.loads((directory / "eda_report.json").read_text(encoding="utf-8"))
            profile = analyze_bo_problem(frame, report, problem)
            # Compute and validate both artifacts before replacing either one.
            write_artifact(directory / "bo_problem.json", problem.model_dump_json(indent=2))
            write_artifact(directory / "bo_profile.json", profile.model_dump_json(indent=2))
            (directory / "surrogate_report.json").unlink(missing_ok=True)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=500, detail="Could not read or save this session's BO artifacts.") from error
    return problem


@app.get("/analyses/{session_id}/bo-problem", response_model=BOProblem)
def get_bo_problem(session_id: str) -> BOProblem:
    path = session_directory(session_id) / "bo_problem.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No BO configuration has been saved for this analysis.")
    with BO_ARTIFACT_LOCK:
        return BOProblem.model_validate_json(path.read_text(encoding="utf-8"))


@app.post("/analyses/{session_id}/bo-profile", response_model=BOProfile)
def generate_bo_profile(session_id: str, config: BOAnalyzerConfig | None = None) -> BOProfile:
    directory = session_directory(session_id)
    with BO_ARTIFACT_LOCK:
        if not (directory / "bo_problem.json").is_file():
            raise HTTPException(status_code=404, detail="Save a BO problem before generating its profile.")
        try:
            return generate_session_profile(directory, config)
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=f"Could not profile this BO problem: {error}") from error
        except OSError as error:
            raise HTTPException(status_code=500, detail="Could not read or save this session's BO artifacts.") from error


@app.get("/analyses/{session_id}/bo-profile", response_model=BOProfile)
def get_bo_profile(session_id: str) -> BOProfile:
    directory = session_directory(session_id)
    with BO_ARTIFACT_LOCK:
        path = directory / "bo_profile.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="No BO profile has been generated for this analysis.")
        try:
            profile = BOProfile.model_validate_json(path.read_text(encoding="utf-8"))
            problem = BOProblem.model_validate_json((directory / "bo_problem.json").read_text(encoding="utf-8"))
        except ValueError as error:
            raise HTTPException(status_code=422, detail="Stored BO artifacts are invalid; regenerate the profile.") from error
        except OSError as error:
            raise HTTPException(status_code=500, detail="Could not read this session's BO artifacts.") from error
        if profile.problem_sha256 != problem_fingerprint(problem):
            raise HTTPException(status_code=409, detail="BO profile is stale; regenerate it for the current problem.")
        return profile


@app.post("/analyses/{session_id}/bo-recommend", response_model=SurrogateReport)
def generate_surrogate_report(session_id: str) -> SurrogateReport:
    directory = session_directory(session_id)
    if not (directory / "dataset.csv").is_file():
        raise HTTPException(status_code=404, detail="Session dataset is no longer available.")
    # Snapshot under the lock; provider calls do not block other artifact operations.
    with BO_ARTIFACT_LOCK:
        profile = get_bo_profile(session_id)
        problem = get_bo_problem(session_id)
    report = recommend_surrogate(profile, problem)
    with BO_ARTIFACT_LOCK:
        current = get_bo_profile(session_id)
        if profile_fingerprint(current) != report.profile_sha256:
            raise HTTPException(status_code=409, detail="BO profile changed during recommendation; retry.")
        try:
            write_artifact(directory / "surrogate_report.json", report.model_dump_json(indent=2))
            logger.info("Surrogate recommendation saved (source=%s)", report.source)
        except OSError as error:
            raise HTTPException(status_code=500, detail="Could not save surrogate report.") from error
    return report


@app.get("/analyses/{session_id}/bo-report", response_model=SurrogateReport)
def get_surrogate_report(session_id: str) -> SurrogateReport:
    directory = session_directory(session_id)
    with BO_ARTIFACT_LOCK:
        profile = get_bo_profile(session_id)
        path = directory / "surrogate_report.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Generate a surrogate recommendation first.")
        try:
            report = SurrogateReport.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise HTTPException(status_code=422, detail="Stored recommendation is invalid; regenerate it.") from error
        except OSError as error:
            raise HTTPException(status_code=500, detail="Could not read surrogate report.") from error
        if report.problem_sha256 != profile.problem_sha256 or report.profile_sha256 != profile_fingerprint(profile):
            raise HTTPException(status_code=409, detail="Recommendation is stale; regenerate it.")
        return report


def main() -> None:
    import uvicorn

    uvicorn.run("eda_agent.api:app", host="127.0.0.1", port=8000, reload=True)
