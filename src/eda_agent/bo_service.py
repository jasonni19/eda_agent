"""Session-backed BO artifacts using the existing local file convention."""

import json
import logging
import time
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .agent import load_csv
from .bo import BOProblem
from .bo_analyzer import analyze_bo_problem
from .bo_profile import BOAnalyzerConfig, BOProfile

# Serialize BO artifact operations within the application's single API process.
BO_ARTIFACT_LOCK = RLock()


def write_artifact(path: Path, content: str) -> None:
    """Atomically replace artifacts, tolerating short Windows sync/reader locks."""
    temporary = path.with_name(f"{path.stem}.{uuid4().hex}.tmp")
    try:
        for attempt in range(6):
            try:
                temporary.write_text(content, encoding="utf-8")
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.05 * 2 ** attempt)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            # Cleanup must not hide the original failure or invalidate a saved artifact.
            logging.getLogger(__name__).warning("Temporary artifact cleanup was locked (%s)", temporary.name)


def generate_session_profile(directory: Path, config: BOAnalyzerConfig | None = None) -> BOProfile:
    """Read all three persisted inputs and regenerate the profile, without an LLM."""
    with BO_ARTIFACT_LOCK:
        problem = BOProblem.model_validate_json((directory / "bo_problem.json").read_text(encoding="utf-8"))
        report = json.loads((directory / "eda_report.json").read_text(encoding="utf-8"))
        profile = analyze_bo_problem(load_csv(directory / "dataset.csv"), report, problem, config)
        write_artifact(directory / "bo_profile.json", profile.model_dump_json(indent=2))
        return profile
