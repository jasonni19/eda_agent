import json
from copy import deepcopy

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from eda_agent.api import app
from eda_agent.bo import BOProblem, column_metadata, validate_problem


@pytest.fixture
def frame():
    return pd.DataFrame({"efficiency": [0.5, 0.8], "temperature": [20.5, 100.5],
                         "pressure": [1, 10], "material": ["Steel", "Aluminum"], "cost": [50, 120]})


@pytest.fixture
def config():
    return {"objective": {"column": "efficiency", "direction": "maximize"},
            "decision_variables": [
                {"name": "temperature", "type": "continuous", "bounds": [20, 100]},
                {"name": "pressure", "type": "integer", "bounds": [1, 10]},
                {"name": "material", "type": "categorical", "categories": ["Steel", "Aluminum"]}],
            "constraints": []}


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
@pytest.mark.parametrize("constraints", [[], [
    {"variable": "cost", "operator": "<=", "value": 100},
    {"variable": "pressure", "operator": ">=", "value": 2},
    {"variable": "material", "operator": "==", "value": "Steel"}]])
def test_valid_problem(frame, config, direction, constraints):
    config["objective"]["direction"] = direction
    config["constraints"] = constraints
    problem = validate_problem(BOProblem.model_validate(config), frame)
    assert json.loads(problem.model_dump_json()) == config


@pytest.mark.parametrize("mutation", [
    lambda c: c["objective"].update(column="missing"),
    lambda c: c["objective"].update(column="material"),
    lambda c: c.pop("objective"),
    lambda c: c["objective"].update(direction="up"),
    lambda c: c["decision_variables"].append(deepcopy(c["decision_variables"][0])),
    lambda c: c["decision_variables"][0].update(name="efficiency"),
    lambda c: c["decision_variables"][0].update(bounds=[100, 20]),
    lambda c: c["decision_variables"][0].update(bounds=[20, 20]),
    lambda c: c["decision_variables"][0].update(bounds=[0, float("inf")]),
    lambda c: c["decision_variables"][1].update(bounds=[1.5, 10]),
    lambda c: c["decision_variables"][1].update(bounds=[True, 10]),
    lambda c: c["decision_variables"][0].update(name="missing"),
    lambda c: c["decision_variables"][0].update(name="material"),
    lambda c: c["decision_variables"][2].update(categories=[]),
    lambda c: c["decision_variables"][2].update(categories=["Titanium"]),
    lambda c: c["decision_variables"][2].update(categories=["Steel", "Steel"]),
    lambda c: c.update(decision_variables=[]),
])
def test_invalid_problem(frame, config, mutation):
    mutation(config)
    with pytest.raises((ValueError, ValidationError)):
        validate_problem(BOProblem.model_validate(config), frame)


@pytest.mark.parametrize("constraint", [
    {"variable": "missing", "operator": "<=", "value": 1},
    {"variable": "cost", "operator": "<", "value": 1},
    {"variable": "cost", "operator": "<=", "value": "100"},
    {"variable": "cost", "operator": "<=", "value": True},
    {"variable": "cost", "operator": "<=", "value": float("nan")},
    {"variable": "material", "operator": "<=", "value": "Steel"},
    {"variable": "material", "operator": "==", "value": "Titanium"},
    {"variable": "pressure", "operator": "==", "value": 1.5},
    {"variable": "pressure", "operator": ">=", "value": 11},
])
def test_invalid_constraints(frame, config, constraint):
    config["constraints"] = [constraint]
    with pytest.raises((ValueError, ValidationError)):
        validate_problem(BOProblem.model_validate(config), frame)


def test_conflicting_constraints(frame, config):
    config["constraints"] = [{"variable": "pressure", "operator": ">=", "value": 8},
                             {"variable": "pressure", "operator": "<=", "value": 4}]
    with pytest.raises(ValueError, match="no feasible"):
        validate_problem(BOProblem.model_validate(config), frame)


def test_metadata():
    columns = column_metadata(pd.DataFrame({"integer": [1, 3], "float": [0.5, float("inf")],
                                           "category": ["A", None], "flag": [True, False]}))
    assert columns[0]["suggested_type"] == "integer"
    assert columns[0]["bounds"] == [1, 3]
    assert columns[1]["bounds"] == [0.5, 0.5]
    assert columns[2]["categories"] == ["A"]
    assert columns[3]["numeric"] is False
    json.dumps(columns, allow_nan=False)


def test_empty_numeric_objective(frame, config):
    frame["efficiency"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_problem(BOProblem.model_validate(config), frame)


def test_save_retrieve_and_session_isolation(tmp_path, monkeypatch, frame, config):
    monkeypatch.setenv("EDA_SESSION_DIR", str(tmp_path))
    session = "4b720f15-f7a1-4012-b510-a76de081f148"
    directory = tmp_path / session
    directory.mkdir()
    frame.to_csv(directory / "dataset.csv", index=False)
    (directory / "eda_report.json").write_text('{"target_recommendations": []}')
    client = TestClient(app)
    path = f"/analyses/{session}/bo-problem"
    assert client.get(path).status_code == 404
    assert client.get(f"/analyses/{session}/bo-setup").json()["dataset_columns"][0]["name"] == "efficiency"
    assert client.post(path, json=config).json() == config
    assert json.loads((directory / "bo_problem.json").read_text()) == config
    # Fresh client reads the on-disk artifact; there is no in-memory session dependency.
    assert TestClient(app).get(path).json() == config
    invalid = deepcopy(config)
    invalid["objective"]["column"] = "unknown"
    assert client.post(path, json=invalid).status_code == 422
    assert client.get(path).json() == config
    invalid["objective"]["column"] = "material"
    assert client.post(path, json=invalid).status_code == 422
    assert client.post("/analyses/00000000-0000-0000-0000-000000000000/bo-problem", json=config).status_code == 404
    assert client.get("/analyses/not-a-session/bo-setup").status_code == 404
    config["objective"]["direction"] = "minimize"
    assert client.post(path, json=config).status_code == 200
    assert client.get(path).json() == config


def test_analyze_returns_persistent_bo_session(tmp_path, monkeypatch, config):
    from datetime import UTC, datetime
    import eda_agent.api as api
    from eda_agent.schemas import EDAReport

    monkeypatch.setenv("EDA_SESSION_DIR", str(tmp_path))
    report = EDAReport(
        source_file="experiment.csv", generated_at=datetime.now(UTC),
        executive_summary="Example analysis.", dataset={"row_count": 2, "column_count": 5, "columns": []},
        target_recommendations=[], transformations=[], feature_exclusions=[], plot_plan=[],
        ml_recommendations=[], findings=[], caveats=[], tools_used=[],
    )
    monkeypatch.setattr(api, "run_eda_detailed", lambda *args: (report, []))
    monkeypatch.setattr(api, "render_plot_images", lambda *args: [])
    client = TestClient(app)
    csv = b"efficiency,temperature,pressure,material,cost\n0.5,20.5,1,Steel,50\n0.8,100.5,10,Aluminum,120\n"
    first = client.post("/analyze", files={"file": ("experiment.csv", csv, "text/csv")})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["report"]["executive_summary"] == "Example analysis."
    assert body["dataset_columns"][0]["name"] == "efficiency"
    directory = tmp_path / body["session_id"]
    assert (directory / "dataset.csv").read_bytes() == csv
    assert json.loads((directory / "eda_report.json").read_text())["source_file"] == "experiment.csv"
    assert client.post(f'/analyses/{body["session_id"]}/bo-problem', json=config).status_code == 200
    second = client.post("/analyze", files={"file": ("experiment.csv", csv, "text/csv")}).json()
    assert second["session_id"] != body["session_id"]
    assert client.get(f'/analyses/{second["session_id"]}/bo-problem').status_code == 404
