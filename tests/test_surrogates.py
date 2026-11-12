"""Selection contract and API tests use mocked advice; no provider requests."""
import json
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from eda_agent.api import app
from eda_agent.bo import BOProblem
from eda_agent.bo_analyzer import analyze_bo_problem
from eda_agent.surrogates import (SurrogateAdvice, SurrogateConfiguration, fallback,
                                  recommend_surrogate)


def inputs():
    problem = BOProblem.model_validate({"objective": {"column": "y", "direction": "maximize"},
        "decision_variables": [{"name": "x", "type": "continuous", "bounds": [0, 10]}]})
    profile = analyze_bo_problem(pd.DataFrame({"x": [0, 1, 2], "y": [1, 3, 2]}), {}, problem)
    return problem, profile


@pytest.mark.parametrize("patch", [
    {"model": "Invented"}, {"kernel": "RBF"}, {"n_estimators": 128},
    {"uncertainty": "ensemble_spread"}, {"normalize_inputs": False}, {"code": "print(1)"},
])
def test_rejects_unsupported_configuration(patch):
    config = fallback(inputs()[1]).configuration.model_dump()
    with pytest.raises(ValidationError):
        SurrogateConfiguration.model_validate({**config, **patch})


def test_mocked_valid_advice():
    problem, profile = inputs()
    result = recommend_surrogate(profile, problem, invoke=lambda *_: fallback(profile).model_dump())
    assert result.source == "llm"
    assert result.configuration.model == "GaussianProcess"
    assert result.execution_available is False


def test_bounded_retries_and_fallback():
    problem, profile = inputs()
    attempts = []
    def invalid(*_):
        attempts.append(1)
        return {"model": "arbitrary"}
    report = recommend_surrogate(profile, problem, invoke=invalid)
    assert len(attempts) == 2
    assert report.source == "deterministic_fallback"
    assert report.confidence == "low"


def test_retry_recovers_and_excludes_raw_data():
    problem, profile = inputs()
    attempts = []
    def invoke(received, received_problem):
        assert received is profile and received_problem is problem
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("malformed")
        return fallback(profile)
    assert recommend_surrogate(profile, problem, invoke=invoke).source == "llm"
    assert len(attempts) == 2


def test_tree_fallback():
    problem, profile = inputs()
    profile.search_space.categorical_variables = 1
    advice = fallback(profile)
    assert advice.configuration.model == "RandomForest"
    assert advice.configuration.uncertainty == "ensemble_spread"


def test_rejects_mismatched_problem():
    problem, profile = inputs()
    problem.objective.direction = "minimize"
    with pytest.raises(ValueError, match="match"):
        recommend_surrogate(profile, problem)


def test_alternatives_cannot_duplicate_recommendation():
    _, profile = inputs()
    advice = fallback(profile).model_dump()
    advice["alternatives"] = [{"model": "GaussianProcess", "reason": "duplicate"}]
    with pytest.raises(ValidationError):
        SurrogateAdvice.model_validate(advice)


def test_recommendation_pipeline_and_stale_report(tmp_path, monkeypatch):
    monkeypatch.setenv("EDA_SESSION_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    session = str(uuid4())
    directory = tmp_path / session
    directory.mkdir()
    (directory / "eda_report.json").write_text("{}")
    (directory / "dataset.csv").write_text("x,y\n0,1\n1,3\n2,2\n")
    client = TestClient(app)
    problem, _ = inputs()
    assert client.post(f"/analyses/{session}/bo-problem", json=problem.model_dump(mode="json")).status_code == 200
    endpoint = f"/analyses/{session}"
    assert client.get(endpoint + "/bo-report").status_code == 404
    response = client.post(endpoint + "/bo-recommend")
    assert response.status_code == 200
    assert response.json()["source"] == "deterministic_fallback"
    assert client.get(endpoint + "/bo-report").json() == response.json()
    assert json.loads((directory / "surrogate_report.json").read_text())["problem_sha256"]
    assert client.post(endpoint + "/bo-profile", json={"high_correlation_threshold": 0.5}).status_code == 200
    assert client.get(endpoint + "/bo-report").status_code == 409
    problem.objective.direction = "minimize"
    client.post(endpoint + "/bo-problem", json=problem.model_dump(mode="json"))
    assert not (directory / "surrogate_report.json").exists()
    assert client.post(f"/analyses/{uuid4()}/bo-recommend").status_code == 404


def test_langchain_agent_path_is_mocked(monkeypatch):
    from eda_agent import surrogates
    problem, profile = inputs()
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    model_calls = []
    def model_factory(name, **kwargs):
        model_calls.append(kwargs)
        return object()
    class Agent:
        def invoke(self, payload, config):
            assert config["recursion_limit"] == 4
            assert "profile" in payload["messages"][0]["content"]
            return {"structured_response": fallback(profile)}
    monkeypatch.setattr(surrogates, "init_chat_model", model_factory)
    monkeypatch.setattr(surrogates, "create_agent", lambda **_: Agent())
    assert recommend_surrogate(profile, problem).source == "llm"
    assert model_calls == [{"temperature": 0, "max_retries": 0, "timeout": 30}]
