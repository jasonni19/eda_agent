"""Deterministic BO problem schema, dataset metadata, and validation."""

from __future__ import annotations

import math
from typing import Annotated, Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

Category = StrictStr | StrictBool | StrictInt | StrictFloat
Number = StrictInt | StrictFloat


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Objective(StrictModel):
    column: str
    direction: Literal["maximize", "minimize"]


class ContinuousVariable(StrictModel):
    name: str
    type: Literal["continuous"]
    bounds: tuple[Number, Number]


class IntegerVariable(StrictModel):
    name: str
    type: Literal["integer"]
    bounds: tuple[StrictInt, StrictInt]


class CategoricalVariable(StrictModel):
    name: str
    type: Literal["categorical"]
    categories: list[Category] = Field(min_length=1)


DecisionVariable = Annotated[
    ContinuousVariable | IntegerVariable | CategoricalVariable, Field(discriminator="type")
]


class Constraint(StrictModel):
    variable: str
    operator: Literal["<=", ">=", "=="]
    value: Category


class BOProblem(StrictModel):
    objective: Objective
    decision_variables: list[DecisionVariable] = Field(min_length=1)
    constraints: list[Constraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_variables(self):
        names = [variable.name for variable in self.decision_variables]
        if len(names) != len(set(names)):
            raise ValueError("Decision variables must not be duplicated.")
        if self.objective.column in names:
            raise ValueError("The objective cannot be a decision variable.")
        for variable in self.decision_variables:
            if variable.type != "categorical" and variable.bounds[0] >= variable.bounds[1]:
                raise ValueError(f"{variable.name}: minimum must be less than maximum.")
        return self


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


def column_metadata(frame: pd.DataFrame) -> list[dict]:
    """Use the CSV itself, never LLM output, for selectable values and bounds."""
    result = []
    for name in frame.columns:
        series = frame[name]
        numeric = is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype)
        values = [_scalar(value) for value in series.dropna().unique()]
        values = [value for value in values if not isinstance(value, float) or math.isfinite(value)]
        finite = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        result.append({
            "name": name,
            "numeric": bool(numeric),
            "suggested_type": "integer" if is_integer_dtype(series.dtype) else "continuous" if numeric else "categorical",
            "bounds": [min(finite), max(finite)] if numeric and finite else None,
            "categories": values,
        })
    return result


def _same_value(left, right) -> bool:
    # Python otherwise treats True and 1 as the same category.
    return isinstance(left, bool) == isinstance(right, bool) and left == right


def validate_problem(problem: BOProblem, frame: pd.DataFrame) -> BOProblem:
    columns = {column["name"]: column for column in column_metadata(frame)}
    objective = columns.get(problem.objective.column)
    if objective is None or not objective["numeric"]:
        raise ValueError("Choose an existing numeric objective column.")
    if objective["bounds"] is None:
        raise ValueError("The objective needs at least one finite numeric observation.")
    variables = {variable.name: variable for variable in problem.decision_variables}
    for variable in problem.decision_variables:
        column = columns.get(variable.name)
        if column is None:
            raise ValueError(f"Unknown decision variable: {variable.name}.")
        if variable.type == "categorical":
            observed = {(isinstance(value, bool), value) for value in column["categories"]}
            chosen = [(isinstance(value, bool), value) for value in variable.categories]
            if not set(chosen).issubset(observed):
                raise ValueError(f"{variable.name}: choose categories observed in this dataset.")
            if len(chosen) != len(set(chosen)):
                raise ValueError(f"{variable.name}: categories must be unique.")
        elif not column["numeric"]:
            raise ValueError(f"{variable.name}: numeric bounds require a numeric column.")
    for constraint in problem.constraints:
        column = columns.get(constraint.variable)
        if column is None:
            raise ValueError(f"Unknown constraint column: {constraint.variable}.")
        variable = variables.get(constraint.variable)
        categorical = (variable is not None and variable.type == "categorical") or not column["numeric"]
        if categorical:
            allowed = variable.categories if variable and variable.type == "categorical" else column["categories"]
            if constraint.operator != "==" or not any(_same_value(constraint.value, value) for value in allowed):
                raise ValueError(f"{constraint.variable}: categorical constraints require == and an allowed category.")
        else:
            if isinstance(constraint.value, bool) or not isinstance(constraint.value, (int, float)):
                raise ValueError(f"{constraint.variable}: constraint value must be numeric.")
            if variable and variable.type == "integer" and constraint.operator == "==" and constraint.value != int(constraint.value):
                raise ValueError(f"{constraint.variable}: integer equality requires an integer value.")
        if variable and variable.type != "categorical":
            low, high = variable.bounds
            if ((constraint.operator == "<=" and constraint.value < low)
                or (constraint.operator == ">=" and constraint.value > high)
                or (constraint.operator == "==" and not low <= constraint.value <= high)):
                raise ValueError(f"{constraint.variable}: constraint conflicts with its bounds.")
    # Detect contradictory restrictions on each selected variable.
    for name, variable in variables.items():
        restrictions = [item for item in problem.constraints if item.variable == name]
        if variable.type == "categorical":
            equalities = [item.value for item in restrictions]
            if equalities and any(not _same_value(value, equalities[0]) for value in equalities):
                raise ValueError(f"{name}: constraints conflict.")
        else:
            low, high = variable.bounds
            for item in restrictions:
                if item.operator in (">=", "=="):
                    low = max(low, item.value)
                if item.operator in ("<=", "=="):
                    high = min(high, item.value)
            if low > high or (variable.type == "integer" and math.ceil(low) > math.floor(high)):
                raise ValueError(f"{name}: constraints leave no feasible values.")
    return problem
