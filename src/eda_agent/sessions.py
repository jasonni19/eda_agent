"""Local analysis artifacts, keyed by an opaque session id."""

import os
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException


def storage_root() -> Path:
    return Path(os.getenv("EDA_SESSION_DIR", "analysis_sessions"))


def session_directory(session_id: str) -> Path:
    try:
        if str(UUID(session_id)) != session_id:
            raise ValueError
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Analysis session not found.") from error
    directory = storage_root() / session_id
    if not (directory / "eda_report.json").is_file():
        raise HTTPException(status_code=404, detail="Analysis session not found.")
    return directory
