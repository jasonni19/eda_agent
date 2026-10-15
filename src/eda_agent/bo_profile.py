"""Versioned JSON contract and heuristic thresholds for deterministic BO profiling."""

from typing import Literal

from pydantic import Field

from .bo import Constraint, Number, Objective, StrictModel


class BOAnalyzerConfig(StrictModel):
    high_correlation_threshold: float = Field(default=0.85, ge=0, le=1)
    outlier_iqr_multiplier: float = Field(default=1.5, gt=0)
    high_outlier_fraction: float = Field(default=0.05, ge=0, le=1)
    strong_skewness_threshold: float = Field(default=1.0, gt=0)
    high_dimensional_threshold: int = Field(default=20, ge=1)
    low_sample_dimension_ratio: float = Field(default=10, gt=0)
    high_categorical_cardinality: int = Field(default=10, ge=1)
    near_constant_fraction: float = Field(default=0.95, gt=0, le=1)
    near_constant_absolute_tolerance: float = Field(default=1e-12, ge=0)
    near_constant_relative_tolerance: float = Field(default=1e-8, ge=0)


class DatasetProfile(StrictModel):
    n_samples: int = Field(ge=0)
    n_complete_samples: int = Field(ge=0)
    n_complete_in_domain_samples: int = Field(ge=0)
    n_training_samples: int = Field(ge=0)
    sample_dimension_ratio: float | None
    training_sample_dimension_ratio: float | None
    small_sample_relative_to_dimensions: bool


class NumericDomain(StrictModel):
    bounds: tuple[Number, Number]
    range: float | None


class SearchSpaceProfile(StrictModel):
    n_dimensions: int = Field(ge=0)
    continuous_variables: int = Field(ge=0)
    integer_variables: int = Field(ge=0)
    categorical_variables: int = Field(ge=0)
    space_type: Literal["continuous", "integer", "categorical", "mixed"]
    mixed_space: bool
    categorical_cardinality: dict[str, int]
    categorical_combinations: int
    discrete_search_space_size: int
    full_cardinality: Literal["finite", "non_finite"]
    full_search_space_size: int | None
    numeric_domains: dict[str, NumericDomain]
    high_dimensional: bool
    high_categorical_cardinality: bool


class CorrelationPair(StrictModel):
    left: str
    right: str
    correlation: float | None


class OutlierSummary(StrictModel):
    n_valid: int = Field(ge=0)
    count: int | None
    percentage: float | None
    lower_fence: float | None
    upper_fence: float | None


class RepeatedObservations(StrictModel):
    repeated_configurations: int
    observations_at_repeated_configurations: int
    configurations_with_varying_objective: int
    max_objective_range: float | None


class DataQualityProfile(StrictModel):
    missing_values_present: bool
    decision_missing_values_present: bool
    objective_missing_values_present: bool
    missing_percentage: dict[str, float | None]
    invalid_value_counts: dict[str, int]
    pairwise_correlations: list[CorrelationPair]
    max_absolute_correlation: float | None
    highly_correlated_pairs: int
    high_correlation: bool
    outliers: dict[str, OutlierSummary]
    outliers_present: bool
    substantial_outliers_present: bool
    duplicate_configurations: int
    constant_decision_variables: list[str]
    near_constant_decision_variables: list[str]
    repeated_objective: RepeatedObservations


class ObjectiveProfile(Objective):
    n_valid: int
    mean: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None
    skewness: float | None
    unique_values: int
    near_constant: bool | None
    strong_skewness: bool | None


class ConstraintObservation(Constraint):
    directly_controllable: bool
    evaluated_samples: int
    violation_count: int
    unevaluable_samples: int


class CoverageProfile(StrictModel):
    in_bounds_percentage: float | None
    in_bounds_samples: int
    out_of_bounds_samples: int
    unevaluable_domain_samples: int
    distinct_configurations: int
    constraint_observations: list[ConstraintObservation]
    constraint_violation_samples: int
    constraint_unevaluable_samples: int


class BOProfile(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    problem_sha256: str
    thresholds: BOAnalyzerConfig
    dataset: DatasetProfile
    search_space: SearchSpaceProfile
    data_quality: DataQualityProfile
    objective: ObjectiveProfile
    coverage: CoverageProfile
    warnings: list[str]
