"""this defines the structure of the report to be produced by the agent."""

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


# this is the schema for the findings that will be returned by the agent.
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
        "feature_types",
        "target",
        "distributions",
        "redundancy",
        "transformations",
        "feature_exclusions",
        "machine_learning",
    ]
    severity: Literal["info", "low", "medium", "high"]
    summary: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)


# this is the schema for the target recommendations that will be returned by the agent.
class TargetRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    confidence: Literal["low", "medium", "high"]
    likely_task: Literal["classification", "regression", "unknown"]
    rationale: str = Field(min_length=1)
    cautions: list[str]


# this is the schema for the transformation recommendations that will be returned by the agent.
class TransformationRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    transformation: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


# this is the schema for the feature exclusions that will be returned by the agent.
class FeatureExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    reason: str = Field(min_length=1)
    action: Literal["exclude", "review"]


# this is the schema for the plot plan that will be returned by the agent.
class PlotPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plot_type: Literal[
        "missingness_bar", "histogram", "box_plot", "count_plot",
        "correlation_heatmap", "target_relationship",
    ]
    columns: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


# this is the schema for the ML recommendations that will be returned by the agent.
class MLRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area: Literal["problem_framing", "preprocessing", "validation", "metrics", "modeling"]
    recommendation: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


# this is the schema for the EDA report that will be returned by the agent.
class EDAReport(BaseModel):
    """This is the Machine-readable final deliverable, a JSON schema that defines the structure of the report to be produced by the agent."""

    model_config = ConfigDict(extra="forbid")

    source_file: str
    generated_at: datetime
    executive_summary: str = Field(min_length=1)
    dataset: DatasetSummary
    target_recommendations: list[TargetRecommendation]
    transformations: list[TransformationRecommendation]
    feature_exclusions: list[FeatureExclusion]
    plot_plan: list[PlotPlan]
    ml_recommendations: list[MLRecommendation]
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
        "feature_analysis",
        "distributions",
        "redundancy_analysis",
        "preprocessing_advice",
        "plot_plan",
    ]]
