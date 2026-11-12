"""Pure candidate validation and seeded feasible-space sampling; no eval()."""
import math
from collections.abc import Mapping, Sequence
import numpy as np
from ..bo import BOProblem, Category, _same_value
from .schemas import Observation


def candidate_key(candidate: Mapping[str, Category]) -> tuple:
    return tuple(sorted((key, "number" if isinstance(value, (int, float)) and not isinstance(value, bool) else type(value).__name__,
                         round(float(value), 12) if isinstance(value, (int, float)) and not isinstance(value, bool) else value)
                        for key, value in candidate.items()))


def validate_candidate(problem: BOProblem, candidate: dict[str, Category],
                       history: Sequence[Observation] = ()) -> dict[str, Category]:
    variables = {v.name: v for v in problem.decision_variables}
    if set(candidate) != set(variables):
        raise ValueError("Candidate must contain exactly the decision variables.")
    for name, variable in variables.items():
        value = candidate[name]
        if variable.type == "categorical":
            if not any(_same_value(value, allowed) for allowed in variable.categories):
                raise ValueError(f"Invalid category: {name}")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Finite numeric input required: {name}")
            if not variable.bounds[0] <= value <= variable.bounds[1]:
                raise ValueError(f"Out of bounds: {name}")
            if variable.type == "integer" and int(value) != value:
                raise ValueError(f"Integer input required: {name}")
    for constraint in problem.constraints:
        if constraint.variable not in candidate:
            raise ValueError("Output constraints require a feasibility model; this benchmark only supports input constraints.")
        value = candidate[constraint.variable]
        valid = (_same_value(value, constraint.value) if constraint.operator == "==" else
                 value <= constraint.value if constraint.operator == "<=" else value >= constraint.value)
        if not valid:
            raise ValueError("Candidate violates an input constraint.")
    if candidate_key(candidate) in {candidate_key(item.configuration) for item in history}:
        raise ValueError("Duplicate candidate.")
    return dict(candidate)


def sample_candidates(problem: BOProblem, rng: np.random.Generator, count: int,
                      history: Sequence[Observation] = (), *, require_count: bool = True) -> list[dict[str, Category]]:
    result, seen = [], {candidate_key(item.configuration) for item in history}
    for _ in range(max(10000, count * 100)):
        candidate = {}
        for variable in problem.decision_variables:
            restrictions = [c for c in problem.constraints if c.variable == variable.name]
            if variable.type == "categorical":
                choices = [value for value in variable.categories if all(c.operator == "==" and _same_value(value, c.value) for c in restrictions)]
                if not choices:
                    raise ValueError("Constraints leave no feasible categories.")
                candidate[variable.name] = choices[int(rng.integers(len(choices)))]
            else:
                low, high = variable.bounds
                for constraint in restrictions:
                    if constraint.operator in (">=", "=="):
                        low = max(low, constraint.value)
                    if constraint.operator in ("<=", "=="):
                        high = min(high, constraint.value)
                if low > high:
                    raise ValueError("Constraints leave no feasible numeric domain.")
                if variable.type == "integer":
                    low, high = math.ceil(low), math.floor(high)
                    if low > high:
                        raise ValueError("Constraints leave no feasible integers.")
                    candidate[variable.name] = int(rng.integers(low, high + 1))
                else:
                    candidate[variable.name] = float(rng.uniform(low, high))
        try:
            validate_candidate(problem, candidate)
        except ValueError:
            continue
        key = candidate_key(candidate)
        if key not in seen:
            result.append(candidate); seen.add(key)
        if len(result) == count:
            return result
    if result and not require_count:
        return result
    raise ValueError("Insufficient distinct feasible candidates; check constraints and search-space size.")
