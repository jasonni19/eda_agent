"""Atomic local artifacts and bounded single-process background-job admission."""
from __future__ import annotations
import csv
from io import StringIO
import json
import os
from pathlib import Path
from uuid import UUID
from ..bo_service import write_artifact


def root():
    return Path(os.getenv("BENCHMARK_RUN_DIR", "benchmark_runs"))


def directory(run_id):
    try:
        if str(UUID(run_id)) != run_id:
            raise ValueError
    except ValueError as error:
        raise FileNotFoundError("Benchmark not found.") from error
    path = root() / run_id
    if not (path / "config.json").is_file():
        raise FileNotFoundError("Benchmark not found.")
    return path


def save_json(path, value):
    write_artifact(path, json.dumps(value, indent=2, allow_nan=False, default=str))


def save_csv(path, rows):
    if not rows:
        return
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    write_artifact(path, output.getvalue())
