"""Deterministic, selected-column-only analysis; no model or agent calls."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .bo import BOProblem, validate_problem
from .bo_profile import BOAnalyzerConfig, BOProfile
from .schemas import EDAReport


def problem_fingerprint(problem: BOProblem) -> str:
    content = json.dumps(problem.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _percentage(count: int, total: int) -> float | None:
    return round(100 * count / total, 6) if total else None


def _numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
        return numeric
    # Booleans are categories in the setup schema, not numeric observations.
    return numeric.mask(series.map(lambda value: isinstance(value, (bool, np.bool_))).astype(bool))


def _category_key(value):
    if hasattr(value, "item"):
        value = value.item()
    return (isinstance(value, bool), value)


def _category_valid(value) -> bool:
    if pd.isna(value):
        return False
    return not isinstance(value, (float, np.floating)) or math.isfinite(value)


def _near_constant(series: pd.Series, config: BOAnalyzerConfig) -> bool | None:
    values = series.dropna()
    if values.empty:
        return None
    if values.nunique() == 1 or values.value_counts(normalize=True).iloc[0] >= config.near_constant_fraction:
        return True
    if is_numeric_dtype(values.dtype):
        low, high = float(values.min()), float(values.max())
        scale = max(abs(low), abs(high), 1.0)
        return bool(high / scale - low / scale <= config.near_constant_absolute_tolerance / scale
                    + config.near_constant_relative_tolerance)
    return False


def analyze_correlations(numeric: pd.DataFrame, config: BOAnalyzerConfig) -> dict:
    matrix = numeric.corr(method="pearson", min_periods=2)
    pairs = []
    for index, left in enumerate(numeric.columns):
        for right in numeric.columns[index + 1:]:
            pairs.append({"left": left, "right": right, "correlation": _finite(matrix.loc[left, right])})
    available = [abs(pair["correlation"]) for pair in pairs if pair["correlation"] is not None]
    high = sum(value >= config.high_correlation_threshold for value in available)
    return {"pairwise_correlations": pairs, "max_absolute_correlation": max(available, default=None),
            "highly_correlated_pairs": high, "high_correlation": bool(high)}


def analyze_outliers(numeric: pd.DataFrame, config: BOAnalyzerConfig) -> dict:
    result = {}
    for name in numeric:
        values = numeric[name].dropna()
        lower = upper = percentage = count = None
        if len(values) >= 4:
            q1, q3 = values.quantile([0.25, 0.75])
            with np.errstate(over="ignore", invalid="ignore"):
                distance = config.outlier_iqr_multiplier * (q3 - q1)
                lower, upper = _finite(q1 - distance), _finite(q3 + distance)
            if lower is not None and upper is not None:
                count = int(((values < lower) | (values > upper)).sum())
                percentage = _percentage(count, len(values))
        result[name] = {"n_valid": len(values), "count": count, "percentage": percentage,
                        "lower_fence": lower, "upper_fence": upper}
    return result


def analyze_objective(series: pd.Series, problem: BOProblem, config: BOAnalyzerConfig) -> dict:
    values = series.dropna()
    mean = std = skewness = None
    if len(values):
        # Scaling avoids overflow in central moments for large but finite data.
        scale = max(float(values.abs().max()), 1.0)
        scaled = values / scale
        mean = _finite(scaled.mean() * scale)
        if len(values) >= 2:
            std = _finite(scaled.std(ddof=1) * scale)
        if len(values) >= 3:
            skewness = _finite(scaled.skew())
    return {**problem.objective.model_dump(), "n_valid": len(values), "mean": mean,
            "standard_deviation": std, "minimum": _finite(values.min()) if len(values) else None,
            "maximum": _finite(values.max()) if len(values) else None, "skewness": skewness,
            "unique_values": int(values.nunique()), "near_constant": _near_constant(values, config),
            "strong_skewness": abs(skewness) >= config.strong_skewness_threshold if skewness is not None else None}


def analyze_search_space(problem: BOProblem, config: BOAnalyzerConfig) -> dict:
    counts = {kind: sum(variable.type == kind for variable in problem.decision_variables)
              for kind in ("continuous", "integer", "categorical")}
    categories = {variable.name: len(variable.categories) for variable in problem.decision_variables if variable.type == "categorical"}
    combinations = math.prod(categories.values())
    discrete_size = combinations * math.prod(variable.bounds[1] - variable.bounds[0] + 1
                                             for variable in problem.decision_variables if variable.type == "integer")
    kinds = [kind for kind, count in counts.items() if count]
    domains = {}
    for variable in problem.decision_variables:
        if variable.type != "categorical":
            low, high = variable.bounds
            domains[variable.name] = {"bounds": [low, high], "range": _finite(high - low)}
    return {"n_dimensions": len(problem.decision_variables),
            **{f"{kind}_variables": count for kind, count in counts.items()},
            "space_type": "mixed" if len(kinds) > 1 else kinds[0], "mixed_space": len(kinds) > 1,
            "categorical_cardinality": categories, "categorical_combinations": combinations,
            "discrete_search_space_size": discrete_size,
            "full_cardinality": "non_finite" if counts["continuous"] else "finite",
            "full_search_space_size": None if counts["continuous"] else discrete_size,
            "numeric_domains": domains, "high_dimensional": len(problem.decision_variables) >= config.high_dimensional_threshold,
            "high_categorical_cardinality": any(count >= config.high_categorical_cardinality for count in categories.values())}


def analyze_constraints(frame: pd.DataFrame, problem: BOProblem) -> tuple[dict, pd.Series, list[str]]:
    observations, warnings = [], []
    any_violated = pd.Series(False, index=frame.index)
    any_unknown = pd.Series(False, index=frame.index)
    selected = {variable.name: variable for variable in problem.decision_variables}
    for constraint in problem.constraints:
        controllable = constraint.variable in selected
        if not controllable:
            warnings.append(f"Constraint on {constraint.variable} is not directly controllable through the decision variables.")
        known = pd.Series(False, index=frame.index)
        passed = pd.Series(False, index=frame.index)
        if constraint.variable in frame:
            series = frame[constraint.variable]
            variable = selected.get(constraint.variable)
            categorical = (variable is not None and variable.type == "categorical") or not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype)
            if categorical:
                known = series.map(_category_valid).astype(bool)
                passed = series.map(lambda value: _category_key(value) == _category_key(constraint.value)) & known
            else:
                series = _numeric(series)
                known = series.notna()
                passed = {"<=": series.le, ">=": series.ge, "==": series.eq}[constraint.operator](constraint.value) & known
        violated = known & ~passed
        any_violated |= violated
        any_unknown |= ~known
        observations.append({**constraint.model_dump(), "directly_controllable": controllable,
                             "evaluated_samples": int(known.sum()), "violation_count": int(violated.sum()),
                             "unevaluable_samples": int((~known).sum())})
        if constraint.variable not in frame or (~known).any():
            warnings.append(f"Constraint on {constraint.variable} cannot be evaluated for all observations; unknown rows are not assumed feasible.")
    return {"constraint_observations": observations, "constraint_violation_samples": int(any_violated.sum()),
            "constraint_unevaluable_samples": int(any_unknown.sum())}, ~any_violated & ~any_unknown, warnings


def _repeated_observations(decisions: pd.DataFrame, objective: pd.Series, complete: pd.Series) -> dict:
    rows = decisions.loc[complete].copy()
    # A unique object key cannot collide with an original dataset column name.
    objective_key = object()
    rows[objective_key] = objective.loc[complete]
    groups = rows.groupby(list(decisions.columns), sort=False, dropna=False)[objective_key]
    stats = groups.agg(["size", "nunique", "min", "max"])
    repeated = stats.loc[stats["size"] > 1]
    with np.errstate(over="ignore", invalid="ignore"):
        ranges = repeated["max"] - repeated["min"]
    return {"repeated_configurations": len(repeated),
            "observations_at_repeated_configurations": int(repeated["size"].sum()),
            "configurations_with_varying_objective": int((repeated["nunique"] > 1).sum()),
            "max_objective_range": _finite(ranges.max()) if len(repeated) else None}


def analyze_bo_problem(
    dataset: pd.DataFrame, eda_report: EDAReport | Mapping, bo_problem: BOProblem | Mapping,
    config: BOAnalyzerConfig | None = None,
) -> BOProfile:
    """Profile a BO problem without mutating inputs or invoking an LLM.

    Complete rows have finite numeric values (integral for integer variables),
    nonmissing finite categories, and finite objectives. Domain membership and
    constraint feasibility are counted separately. Training rows satisfy all three.
    """
    config = BOAnalyzerConfig.model_validate((config or BOAnalyzerConfig()).model_dump())
    if not isinstance(eda_report, (EDAReport, Mapping)):
        raise ValueError("EDA report must be a report object or a JSON object.")
    problem = BOProblem.model_validate(bo_problem.model_dump() if isinstance(bo_problem, BOProblem) else bo_problem)
    if not dataset.columns.is_unique:
        raise ValueError("Dataset column names must be unique.")
    names = [variable.name for variable in problem.decision_variables]
    selected = names + [problem.objective.column]
    missing = [name for name in selected if name not in dataset]
    if missing:
        raise ValueError(f"Missing required dataset columns: {', '.join(missing)}.")
    # Reset index so duplicate/custom input indexes do not change row semantics.
    relevant = list(dict.fromkeys(selected + [item.variable for item in problem.constraints if item.variable in dataset]))
    frame = dataset[relevant].copy().reset_index(drop=True)
    numeric_names = [variable.name for variable in problem.decision_variables if variable.type != "categorical"] + [problem.objective.column]
    numeric = pd.DataFrame({name: _numeric(frame[name]) for name in numeric_names})
    schema_frame = frame.copy()
    for name in numeric_names:
        raw = frame[name]
        if is_bool_dtype(raw.dtype) or (raw.notna().any() and not is_numeric_dtype(raw.dtype)
                                        and pd.to_numeric(raw, errors="coerce").notna().sum() == 0):
            raise ValueError(f"{name}: configured numeric column contains no numeric data.")
        schema_frame[name] = numeric[name]
    variable_types = {variable.name: variable.type for variable in problem.decision_variables}
    for constraint in problem.constraints:
        if (constraint.variable in schema_frame and constraint.variable not in numeric_names
            and variable_types.get(constraint.variable) != "categorical"
            and isinstance(constraint.value, (int, float)) and not isinstance(constraint.value, bool)):
            # A dirty constraint-only numeric column should yield unknown rows,
            # rather than turning every historical requirement into a fatal error.
            schema_frame[constraint.variable] = _numeric(frame[constraint.variable])
    validate_problem(problem, schema_frame, require_observations=False, allow_missing_constraint_columns=True)
    n = len(frame)
    warnings = []
    missing_percentage = {name: _percentage(int(frame[name].isna().sum()), n) for name in selected}
    invalid_counts = {name: int((frame[name].notna() & numeric[name].isna()).sum()) for name in numeric_names}
    complete = numeric[problem.objective.column].notna()
    domain_evaluable = pd.Series(True, index=frame.index)
    in_domain = domain_evaluable.copy()
    decisions = pd.DataFrame(index=frame.index)
    for variable in problem.decision_variables:
        if variable.type == "categorical":
            raw = frame[variable.name]
            valid = raw.map(_category_valid).astype(bool)
            invalid_counts[variable.name] = int((raw.notna() & ~valid).sum())
            decisions[variable.name] = raw.map(_category_key)
            allowed = {_category_key(value) for value in variable.categories}
            inside = valid & decisions[variable.name].isin(allowed)
            complete &= valid
        else:
            values = numeric[variable.name]
            valid = values.notna()
            inside = valid & values.between(*variable.bounds)
            integral = values.eq(np.floor(values)) if variable.type == "integer" else valid
            if variable.type == "integer":
                fractional = valid & ~integral
                invalid_counts[variable.name] += int(fractional.sum())
                if fractional.any():
                    warnings.append(f"{variable.name} contains non-integer observations outside its integer domain.")
            complete &= valid & integral
            inside &= integral
            decisions[variable.name] = values
        domain_evaluable &= valid
        in_domain &= inside
    decision_complete = domain_evaluable.copy()
    for variable in problem.decision_variables:
        if variable.type == "integer":
            decision_complete &= numeric[variable.name].eq(np.floor(numeric[variable.name]))
    constraint_stats, feasible, constraint_warnings = analyze_constraints(schema_frame, problem)
    warnings.extend(constraint_warnings)
    complete_in_domain = complete & in_domain
    training = complete_in_domain & feasible
    dimensions = len(names)
    search_space = analyze_search_space(problem, config)
    correlations = analyze_correlations(numeric[[name for name in numeric_names if name != problem.objective.column]], config)
    outliers = analyze_outliers(numeric, config)
    objective = analyze_objective(numeric[problem.objective.column], problem, config)
    distinct = len(decisions.loc[decision_complete].drop_duplicates())
    duplicates = int(decision_complete.sum()) - distinct
    # Per-variable constancy should not depend on missing values in other columns.
    constant, near_constant = [], []
    for variable in problem.decision_variables:
        values = numeric[variable.name].dropna() if variable.type != "categorical" else frame.loc[frame[variable.name].map(_category_valid), variable.name]
        if values.nunique() == 1:
            constant.append(variable.name)
        elif _near_constant(values, config):
            near_constant.append(variable.name)
    substantial = any(item["percentage"] is not None and item["percentage"] / 100 >= config.high_outlier_fraction
                      and item["count"] > 0 for item in outliers.values())
    quality = {"missing_values_present": bool(frame[selected].isna().any().any()),
               "decision_missing_values_present": bool(frame[names].isna().any().any()),
               "objective_missing_values_present": bool(frame[problem.objective.column].isna().any()),
               "missing_percentage": missing_percentage, "invalid_value_counts": invalid_counts,
               **correlations, "outliers": outliers,
               "outliers_present": any((item["count"] or 0) > 0 for item in outliers.values()),
               "substantial_outliers_present": substantial, "duplicate_configurations": duplicates,
               "constant_decision_variables": constant, "near_constant_decision_variables": near_constant,
               "repeated_objective": _repeated_observations(decisions, numeric[problem.objective.column], complete)}
    low_ratio = int(training.sum()) / dimensions < config.low_sample_dimension_ratio if dimensions else True
    flags = [
        (n == 0, "The dataset is empty; unavailable statistics are null."),
        (quality["missing_values_present"], "Selected columns contain missing values."),
        (any(invalid_counts.values()), "Selected columns contain invalid or non-finite values; they are excluded from complete observations."),
        (not training.any(), "No usable training observations satisfy completeness, domains, and all evaluable constraints."),
        (low_ratio, "Training observations are few relative to the number of decision variables."),
        (search_space["high_dimensional"], "The configured search space is high-dimensional under the configured heuristic."),
        (search_space["high_categorical_cardinality"], "At least one categorical variable has high configured cardinality."),
        (correlations["high_correlation"], "Selected numeric decision variables contain highly correlated pairs."),
        (any(item["correlation"] is None for item in correlations["pairwise_correlations"]), "Some correlations are unavailable because of constant columns, insufficient pairs, or numerical limits."),
        (quality["outliers_present"], "IQR outliers are review flags, not confirmed errors."),
        (any(item["count"] is None for item in outliers.values()), "Some outlier statistics are unavailable: fewer than four finite values or numerical limits."),
        (objective["skewness"] is None, "Objective skewness is unavailable with insufficient observations or numerical limits."),
        (any(domain["range"] is None for domain in search_space["numeric_domains"].values())
         or (objective["n_valid"] > 0 and objective["mean"] is None)
         or (objective["n_valid"] > 1 and objective["standard_deviation"] is None),
         "Some numeric summaries exceed finite floating-point limits and are reported as null."),
        (objective["near_constant"] is True, "The objective is constant or near-constant under the configured heuristic."),
        (objective["strong_skewness"] is True, "The objective has strong skewness under the configured heuristic."),
        (bool(constant or near_constant), "Some decision variables are constant or near-constant in the observations."),
        (duplicates > 0, "Duplicate decision-variable configurations are present."),
        (quality["repeated_objective"]["configurations_with_varying_objective"] > 0, "Repeated configurations have differing objectives; this does not establish a noise distribution or noise variance."),
        (bool((domain_evaluable & ~in_domain).any()), "Some observations are outside configured bounds, integer domains, or allowed categories."),
        (constraint_stats["constraint_violation_samples"] > 0, "Some observations violate configured constraints."),
    ]
    warnings.extend(message for flag, message in flags if flag)
    report = eda_report.model_dump(mode="json") if isinstance(eda_report, EDAReport) else dict(eda_report)
    # EDA measurements are prose findings in this project, not reusable raw statistics.
    # Structured metadata is used only to identify stale/mismatched inputs.
    metadata = report.get("dataset", {})
    if isinstance(metadata, Mapping):
        if metadata.get("row_count", n) != n:
            warnings.append("EDA row count differs from the CSV; all measurements use the CSV.")
        reported_columns = metadata.get("columns")
        if isinstance(reported_columns, list):
            reported_names = {column.get("name") for column in reported_columns if isinstance(column, Mapping)}
            if not set(selected).issubset(reported_names):
                warnings.append("EDA column metadata does not cover all configured BO columns; the CSV defines available columns.")
    return BOProfile.model_validate({
        "problem_sha256": problem_fingerprint(problem), "thresholds": config,
        "dataset": {"n_samples": n, "n_complete_samples": int(complete.sum()),
                    "n_complete_in_domain_samples": int(complete_in_domain.sum()), "n_training_samples": int(training.sum()),
                    "sample_dimension_ratio": n / dimensions if dimensions else None,
                    "training_sample_dimension_ratio": int(training.sum()) / dimensions if dimensions else None,
                    "small_sample_relative_to_dimensions": low_ratio},
        "search_space": search_space, "data_quality": quality, "objective": objective,
        "coverage": {"in_bounds_percentage": _percentage(int(in_domain.sum()), n),
                     "in_bounds_samples": int(in_domain.sum()), "out_of_bounds_samples": int((domain_evaluable & ~in_domain).sum()),
                     "unevaluable_domain_samples": int((~domain_evaluable).sum()), "distinct_configurations": distinct,
                     **constraint_stats}, "warnings": warnings,
    })
