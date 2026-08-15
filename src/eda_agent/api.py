"""HTTP adapter for the chat interface."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .agent import run_eda

load_dotenv()

MAX_UPLOAD_BYTES = 15 * 1024 * 1024

app = FastAPI(title="Object EDA API", version="0.1.0")
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
            report = await run_in_threadpool(run_eda, csv_path, output_path, model_name)
            return report.model_dump(mode="json")
    except (UnicodeDecodeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Could not read this CSV: {error}") from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check the API key and model configuration, then try again.",
        ) from error


def main() -> None:
    import uvicorn

    uvicorn.run("eda_agent.api:app", host="127.0.0.1", port=8000, reload=True)
