"""The contract for the JSON report produced by the agent. this defines the structure of the report to be produced by the agent."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ColumnSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    non_null_count: int = Field(ge=0)
    unique_count: int = Field(ge=0)


class DatasetSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[ColumnSummary]


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "missing_values",
        "statistics",
        "correlations",
        "outliers",
        "duplicates",
        "value_counts",
        "data_quality",
    ]
    severity: Literal["info", "low", "medium", "high"]
    summary: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)


class EDAReport(BaseModel):
    """This is the Machine-readable final deliverable. 
    It is a JSON schema that defines the structure of the report to be produced by the agent."""

    model_config = ConfigDict(extra="forbid")

    source_file: str
    generated_at: datetime
    executive_summary: str = Field(min_length=1)
    dataset: DatasetSummary
    findings: list[Finding]
    caveats: list[str]
    tools_used: list[Literal[
        "overview",
        "missing_values",
        "statistics",
        "correlations",
        "outliers",
        "duplicates",
        "value_counts",
    ]]

