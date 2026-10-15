import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from eda_agent.api import app
from eda_agent.bo import BOProblem
from eda_agent.bo_analyzer import analyze_bo_problem
from eda_agent.bo_profile import BOAnalyzerConfig, BOProfile
from eda_agent.bo_service import generate_session_profile


def continuous(name="x", bounds=(0, 10)):
    return {"name": name, "type": "continuous", "bounds": list(bounds)}


def integer(name="i", bounds=(1, 4)):
    return {"name": name, "type": "integer", "bounds": list(bounds)}


def categorical(name="material", categories=("A", "B")):
    return {"name": name, "type": "categorical", "categories": list(categories)}


def problem(variables=None, constraints=None, direction="maximize"):
    return {"objective": {"column": "y", "direction": direction},
            "decision_variables": variables if variables is not None else [continuous()],
            "constraints": constraints or []}


def analyze(data, config=None, variables=None, constraints=None, direction="maximize", eda=None):
    return analyze_bo_problem(pd.DataFrame(data), eda or {}, problem(variables, constraints, direction), config)


def test_counts_repetitions_and_dimensionality():
    profile = analyze({"x": [0, 1, 1, 2, 3, 0], "i": [1, 2, 2, 3, 4, 1],
                       "material": ["A", "B", "B", "A", "A", "A"], "y": [1, 2, 4, 3, 6, 1]},
                      variables=[continuous(bounds=(0, 3)), integer(), categorical()])
    assert profile.dataset.n_samples == 6
    assert profile.dataset.n_complete_samples == profile.dataset.n_training_samples == 6
    assert profile.dataset.sample_dimension_ratio == 2
    assert profile.search_space.n_dimensions == 3
    assert profile.search_space.continuous_variables == profile.search_space.integer_variables == profile.search_space.categorical_variables == 1
    assert profile.coverage.distinct_configurations == 4
    assert profile.data_quality.duplicate_configurations == 2
    repeated = profile.data_quality.repeated_objective
    assert repeated.repeated_configurations == 2
    assert repeated.observations_at_repeated_configurations == 4
    assert repeated.configurations_with_varying_objective == 1
    assert repeated.max_objective_range == 2


@pytest.mark.parametrize("variables,data,kind,size", [
    ([continuous()], {"x": [0, 1]}, "continuous", None),
    ([integer()], {"i": [1, 2]}, "integer", 4),
    ([categorical()], {"material": ["A", "B"]}, "categorical", 2),
    ([integer(), categorical()], {"i": [1, 2], "material": ["A", "B"]}, "mixed", 8),
])
def test_search_space_types_and_cardinality(variables, data, kind, size):
    profile = analyze({**data, "y": [1, 2]}, variables=variables)
    space = profile.search_space
    assert space.space_type == kind
    assert space.full_search_space_size == size
    assert space.full_cardinality == ("finite" if size is not None else "non_finite")
    assert space.mixed_space == (kind == "mixed")


def test_configured_categories_are_authoritative_and_need_not_be_observed():
    profile = analyze({"material": ["A", "outside"], "y": [1, 2]},
                      variables=[categorical(categories=("A", "B", "C"))])
    assert profile.search_space.categorical_cardinality == {"material": 3}
    assert profile.search_space.categorical_combinations == 3
    assert profile.coverage.out_of_bounds_samples == 1
    assert profile.dataset.n_complete_samples == 2
    assert profile.dataset.n_training_samples == 1


def test_missingness_domain_and_training_are_separate():
    profile = analyze({"x": [1, None, 4, 20, 5], "i": [1, 1, 2.5, 2, 3],
                       "material": ["A", "B", "A", "B", None], "y": [1, 2, 3, 4, None]},
                      variables=[continuous(), integer(), categorical()])
    assert profile.data_quality.missing_percentage == {"x": 20, "i": 0, "material": 20, "y": 20}
    assert profile.data_quality.decision_missing_values_present
    assert profile.data_quality.objective_missing_values_present
    assert profile.dataset.n_complete_samples == 2  # rows 0,3
    assert profile.dataset.n_complete_in_domain_samples == profile.dataset.n_training_samples == 1
    assert profile.coverage.in_bounds_samples == 1
    assert profile.coverage.out_of_bounds_samples == 2  # fractional integer, out-of-bounds x
    assert profile.coverage.unevaluable_domain_samples == 2
    assert profile.coverage.in_bounds_percentage == 20
    assert profile.data_quality.invalid_value_counts["i"] == 1


def test_nonfinite_numeric_values_and_numeric_coercion():
    profile = analyze({"x": [1, np.inf, -np.inf, "bad", None, "2"], "y": [1, 2, 3, 4, 5, 6]})
    assert profile.data_quality.invalid_value_counts["x"] == 3
    assert profile.data_quality.missing_percentage["x"] == pytest.approx(100 / 6, abs=1e-6)
    assert profile.dataset.n_complete_samples == profile.dataset.n_training_samples == 2
    assert profile.coverage.unevaluable_domain_samples == 4
    json.dumps(profile.model_dump(mode="json"), allow_nan=False)


def test_correlations_ignore_objective_categoricals_self_and_constant_columns():
    profile = analyze({"x": [1, 2, 3, 4], "z": [2, 4, 6, 8], "constant": [1, 1, 1, 1],
                       "material": ["A", "B", "A", "B"], "y": [4, 1, 3, 2]},
                      variables=[continuous(), continuous("z"), continuous("constant"), categorical()])
    pairs = profile.data_quality.pairwise_correlations
    assert len(pairs) == 3
    assert pairs[0].correlation == pytest.approx(1)
    assert pairs[1].correlation is None
    assert profile.data_quality.max_absolute_correlation == pytest.approx(1)
    assert profile.data_quality.highly_correlated_pairs == 1
    assert profile.data_quality.high_correlation
    assert profile.data_quality.constant_decision_variables == ["constant"]


def test_iqr_objective_distribution_and_direction():
    profile = analyze({"x": [0] * 7 + [100], "y": [0] * 7 + [100]},
                      variables=[continuous(bounds=(0, 100))], direction="minimize")
    outlier = profile.data_quality.outliers["y"]
    assert outlier.count == 1
    assert outlier.percentage == 12.5
    assert outlier.lower_fence == outlier.upper_fence == 0
    assert profile.data_quality.substantial_outliers_present
    objective = profile.objective
    assert objective.direction == "minimize"
    assert objective.mean == 12.5
    assert objective.standard_deviation == pytest.approx(np.sqrt(1250))
    assert objective.minimum == 0 and objective.maximum == 100
    assert objective.skewness == pytest.approx(np.sqrt(8))
    assert objective.strong_skewness
    assert objective.unique_values == 2


def test_constant_objective_and_single_sample_statistics():
    constant = analyze({"x": [1, 2, 3, 4], "y": [7, 7, 7, 7]})
    assert constant.objective.near_constant
    assert constant.objective.standard_deviation == 0
    assert constant.objective.skewness == 0
    single = analyze({"x": [1], "y": [7]})
    assert single.objective.standard_deviation is None
    assert single.objective.skewness is None
    assert single.data_quality.outliers["x"].count is None


def test_empty_and_entirely_missing_objective():
    empty = analyze({"x": [], "y": []})
    assert empty.dataset.n_samples == 0
    assert empty.dataset.n_training_samples == 0
    assert empty.coverage.in_bounds_percentage is None
    assert empty.objective.mean is empty.objective.near_constant is None
    missing = analyze({"x": [1, 2], "y": [np.nan, np.inf]})
    assert missing.objective.n_valid == missing.dataset.n_training_samples == 0
    assert missing.objective.mean is None
    assert missing.data_quality.objective_missing_values_present
    assert missing.data_quality.invalid_value_counts["y"] == 1
    assert any("No usable training" in warning for warning in missing.warnings)


def test_threshold_configuration():
    config = BOAnalyzerConfig(high_dimensional_threshold=1, high_categorical_cardinality=2,
                              low_sample_dimension_ratio=0.1, near_constant_fraction=0.7)
    profile = analyze({"x": [0, 0, 0, 1], "material": ["A"] * 4, "y": [1, 1, 1, 2]},
                      config=config, variables=[continuous(), categorical()])
    assert profile.search_space.high_dimensional
    assert profile.search_space.high_categorical_cardinality
    assert not profile.dataset.small_sample_relative_to_dimensions
    assert profile.data_quality.near_constant_decision_variables == ["x"]
    assert profile.objective.near_constant
    assert profile.thresholds == config
    with pytest.raises(ValidationError):
        BOAnalyzerConfig(high_correlation_threshold=2)


def test_constraints_observed_violations_and_unknowns():
    profile = analyze({"x": [1, 2, 3, 4], "cost": [10, 30, None, 10], "y": [1, 2, 3, 4]},
                      constraints=[{"variable": "x", "operator": ">=", "value": 2},
                                   {"variable": "cost", "operator": "<=", "value": 20}])
    assert profile.coverage.constraint_violation_samples == 2
    assert profile.coverage.constraint_unevaluable_samples == 1
    assert profile.coverage.constraint_observations[1].directly_controllable is False
    assert profile.dataset.n_complete_in_domain_samples == 4
    assert profile.dataset.n_training_samples == 1
    assert any("not directly controllable" in warning for warning in profile.warnings)
    absent = analyze({"x": [1, 2], "y": [1, 2]},
                     constraints=[{"variable": "missing", "operator": "<=", "value": 20}])
    assert absent.coverage.constraint_unevaluable_samples == 2
    assert absent.dataset.n_training_samples == 0
    assert any("cannot be evaluated" in warning for warning in absent.warnings)


def test_categorical_constraint_and_valid_boolean_categories():
    profile = analyze({"material": [True, False, None], "y": [1, 2, 3]},
                      variables=[categorical(categories=(True, False))],
                      constraints=[{"variable": "material", "operator": "==", "value": True}])
    assert profile.coverage.in_bounds_samples == 2
    assert profile.coverage.constraint_violation_samples == 1
    assert profile.dataset.n_training_samples == 1


def test_dirty_constraint_only_column_is_partially_evaluable():
    profile = analyze({"x": [1, 2, 3], "y": [2, 4, 6], "cost": [10, "bad", np.inf]},
                      constraints=[{"variable": "cost", "operator": "<=", "value": 20}])
    constraint = profile.coverage.constraint_observations[0]
    assert constraint.evaluated_samples == 1
    assert constraint.unevaluable_samples == 2
    assert constraint.violation_count == 0
    assert profile.dataset.n_training_samples == 1


def test_numeric_boolean_values_are_not_numeric_observations():
    profile = analyze({"x": [1, True, 2, None], "y": [1, 2, 3, 4]})
    assert profile.data_quality.invalid_value_counts["x"] == 1
    assert profile.dataset.n_complete_samples == 2


def test_fractional_integer_duplicate_count_and_missing_objective_repeats():
    profile = analyze({"i": [1, 1, 1.5, 2], "y": [1, None, 2, 4]}, variables=[integer()])
    assert profile.coverage.distinct_configurations == 2
    assert profile.data_quality.duplicate_configurations == 1
    assert profile.data_quality.repeated_objective.repeated_configurations == 0
    assert profile.dataset.n_training_samples == 2


def test_large_integer_configuration_identity_is_preserved():
    low = 2 ** 53
    profile = analyze({"i": [low, low + 1, low + 2], "y": [1, 2, 3]},
                      variables=[integer(bounds=(low, low + 2))])
    assert profile.coverage.distinct_configurations == 3
    assert profile.data_quality.duplicate_configurations == 0
    assert profile.search_space.numeric_domains["i"].bounds == (low, low + 2)


def test_partial_pairwise_correlations_and_configured_threshold():
    profile = analyze({"x": [1, 2, None, None], "z": [2, 4, 8, None], "y": [1, 2, 3, 4]},
                      variables=[continuous(), continuous("z")],
                      config=BOAnalyzerConfig(high_correlation_threshold=1))
    assert profile.data_quality.pairwise_correlations[0].correlation == pytest.approx(1)
    assert profile.data_quality.high_correlation


def test_generated_example_matches_fixture():
    directory = Path(__file__).resolve().parents[1] / "examples"
    frame = pd.read_csv(directory / "bo_profile_input.csv")
    config = BOProblem.model_validate_json((directory / "bo_problem.json").read_text())
    result = analyze_bo_problem(frame, {"dataset": {"row_count": len(frame)}}, config)
    assert result == BOProfile.model_validate_json((directory / "bo_profile.json").read_text())


def test_invalid_configurations_and_missing_columns_are_fatal():
    with pytest.raises(ValueError, match="Missing required"):
        analyze({"wrong": [1], "y": [1]})
    with pytest.raises(ValueError, match="numeric"):
        analyze({"x": [1], "y": ["not numeric"]})
    with pytest.raises(ValueError, match="minimum"):
        analyze({"x": [1], "y": [2]}, variables=[continuous(bounds=(2, 1))])
    with pytest.raises(ValueError, match="duplicated"):
        analyze({"x": [1], "y": [2]}, variables=[continuous(), continuous()])
    with pytest.raises(ValueError, match="objective"):
        analyze({"x": [1], "y": [2]}, variables=[continuous("y")])


def test_determinism_eda_authority_and_no_input_mutation():
    data = pd.DataFrame({"x": [1, 2, 2], "y": [2, 4, 6], "unused": [np.inf, "bad", None]}, index=[4, 4, 7])
    original = data.copy(deep=True)
    config = problem()
    eda = {"dataset": {"row_count": 999}, "target_recommendations": [{"column": "unused"}],
           "findings": [{"category": "statistics", "summary": "mean=1000"}]}
    first = analyze_bo_problem(data, eda, config)
    second = analyze_bo_problem(data, deepcopy(eda), deepcopy(config))
    assert first.model_dump_json() == second.model_dump_json()
    assert first.objective.column == "y" and first.objective.mean == 4
    assert "unused" not in first.data_quality.invalid_value_counts
    assert any("EDA row count" in warning for warning in first.warnings)
    pd.testing.assert_frame_equal(data, original)
    assert BOProfile.model_validate_json(first.model_dump_json()) == first
    json.dumps(first.model_dump(mode="json"), allow_nan=False)


def test_extreme_finite_values_are_json_safe():
    profile = analyze({"x": [-1e308, 0, 1e308, 1e308], "y": [-1e308, 0, 1e308, 1e308]},
                      variables=[continuous(bounds=(-1e308, 1e308))])
    json.dumps(profile.model_dump(mode="json"), allow_nan=False)
    assert profile.search_space.numeric_domains["x"].range is None


def write_inputs(directory: Path):
    directory.mkdir(parents=True)
    (directory / "dataset.csv").write_text("x,y\n1,2\n2,4\n3,6\n4,8\n", encoding="utf-8")
    (directory / "eda_report.json").write_text('{"dataset":{"row_count":4}}', encoding="utf-8")
    (directory / "bo_problem.json").write_text(json.dumps(problem()), encoding="utf-8")


def test_persisted_three_input_integration(tmp_path):
    directory = tmp_path / "session"
    write_inputs(directory)
    profile = generate_session_profile(directory)
    stored = BOProfile.model_validate_json((directory / "bo_profile.json").read_text())
    assert stored == profile
    assert stored.dataset.n_training_samples == 4
    assert stored.objective.mean == 5
    assert generate_session_profile(directory).model_dump_json() == stored.model_dump_json()


def test_api_automatic_generation_retrieval_regeneration_and_staleness(tmp_path, monkeypatch):
    monkeypatch.setenv("EDA_SESSION_DIR", str(tmp_path))
    session_id = "9e350c43-a542-4ae7-b818-cdcb410ee3ee"
    directory = tmp_path / session_id
    write_inputs(directory)
    client = TestClient(app)
    endpoint = f"/analyses/{session_id}/bo-profile"
    assert client.get(endpoint).status_code == 404
    response = client.post(f"/analyses/{session_id}/bo-problem", json=problem())
    assert response.status_code == 200, response.text
    stored = client.get(endpoint)
    assert stored.status_code == 200
    assert stored.json()["objective"]["mean"] == 5
    assert (directory / "bo_profile.json").exists()
    regenerated = client.post(endpoint, json={"high_dimensional_threshold": 1})
    assert regenerated.status_code == 200
    assert regenerated.json()["search_space"]["high_dimensional"]
    assert client.get(endpoint).json() == regenerated.json()
    assert client.post(endpoint, json={"high_correlation_threshold": 2}).status_code == 422
    changed = problem(direction="minimize")
    (directory / "bo_problem.json").write_text(json.dumps(changed))
    assert client.get(endpoint).status_code == 409
    assert client.post(endpoint).json()["objective"]["direction"] == "minimize"
    # Invalid replacement settings do not overwrite either saved artifact.
    previous_profile = (directory / "bo_profile.json").read_bytes()
    invalid = problem(variables=[continuous(bounds=(2, 1))])
    assert client.post(f"/analyses/{session_id}/bo-problem", json=invalid).status_code == 422
    assert (directory / "bo_profile.json").read_bytes() == previous_profile
    assert client.get("/analyses/missing/bo-profile").status_code == 404


def test_api_missing_problem_dataset_and_invalid_saved_problem(tmp_path, monkeypatch):
    monkeypatch.setenv("EDA_SESSION_DIR", str(tmp_path))
    directory = tmp_path / "9e350c43-a542-4ae7-b818-cdcb410ee3ee"
    directory.mkdir()
    (directory / "eda_report.json").write_text("{}")
    endpoint = f"/analyses/{directory.name}/bo-profile"
    client = TestClient(app)
    assert client.post(endpoint).status_code == 404
    (directory / "bo_problem.json").write_text(json.dumps(problem()))
    assert client.post(endpoint).status_code == 500
    (directory / "dataset.csv").write_text("wrong,y\n1,2\n")
    assert client.post(endpoint).status_code == 422
    (directory / "bo_problem.json").write_text("{}")
    assert client.post(endpoint).status_code == 422
