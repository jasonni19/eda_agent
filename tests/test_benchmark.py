"""Fairness, actual artifact inference, optimizers, paired metrics and job API."""
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from eda_agent.api import app
from eda_agent.bo import BOProblem
from eda_agent.benchmark.bo_optimizer import BayesianOptimizer, DomainEncoder, FittedSurrogate, optimistic_acquisition
from eda_agent.benchmark.domain import candidate_key, sample_candidates, validate_candidate
from eda_agent.benchmark.llm_optimizer import LLMOptimizer, UsageTracker
from eda_agent.benchmark.metrics import summarize
from eda_agent.benchmark.pretrained import HousingEvaluator, model_problem, model_info, FEATURES
from eda_agent.benchmark.runner import execute
from eda_agent.benchmark.schemas import BenchmarkConfig, METHODS, Observation, RunStatus
from eda_agent.surrogates import fallback, recommend_surrogate, SurrogateConfiguration


def history(problem=None, count=5):
    problem = problem or model_problem()
    return [Observation(configuration=c, objective=float(i)) for i, c in
            enumerate(sample_candidates(problem, np.random.default_rng(1), count))]


def test_defaults_and_budget_limits():
    config = BenchmarkConfig()
    assert (config.initial_samples, config.optimization_iterations, config.random_seeds) == (20, 50, 10)
    for patch in ({"optimization_iterations": 101}, {"random_seeds": 11}, {"seed": -1}, {"model_id": "anything"}, {"initial_samples": True}):
        with pytest.raises(ValidationError):
            BenchmarkConfig(**patch)


def test_pinned_pretrained_model_loads_and_preserves_feature_order():
    assert model_info().available
    evaluator = HousingEvaluator()
    candidate = {"MedInc": 4.445, "HouseAge": 52, "AveRooms": 5.5346}
    matrix = evaluator.preprocess(candidate)
    assert matrix.shape == (1, len(FEATURES))
    assert matrix[0, :8].tolist() == [4.445, 52, 5.5346, 1.1, 1425, 3.07, 35.63, -119.57]
    assert matrix[0, 9] == 1.1 / 5.5346
    before = bytes(evaluator._booster.save_raw())
    result = evaluator.evaluate(candidate)
    assert np.isfinite(result)
    assert evaluator.evaluate(dict(reversed(list(candidate.items())))) == result
    assert bytes(evaluator._booster.save_raw()) == before
    with pytest.raises(ValueError):
        evaluator.evaluate({**candidate, "MedInc": 500})


def test_corrupt_artifact_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_MODEL_DIR", str(tmp_path))
    (tmp_path / "xgb.ubj").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        HousingEvaluator()


def test_candidates_seeded_unique_and_typed():
    problem = model_problem()
    left = sample_candidates(problem, np.random.default_rng(4), 20)
    assert left == sample_candidates(problem, np.random.default_rng(4), 20)
    assert left != sample_candidates(problem, np.random.default_rng(5), 20)
    assert len({candidate_key(item) for item in left}) == 20
    assert all(isinstance(item["HouseAge"], int) for item in left)
    original = Observation(configuration=left[0], objective=1)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_candidate(problem, {**left[0], "HouseAge": float(left[0]["HouseAge"])}, [original])
    for invalid in ({**left[0], "MedInc": True}, {**left[0], "HouseAge": 2.5}, {**left[0], "extra": 1}, {**left[0], "MedInc": float("nan")}):
        with pytest.raises(ValueError):
            validate_candidate(problem, invalid)


def test_sampler_and_validation_enforce_constraints():
    problem = model_problem()
    problem.constraints = BOProblem.model_validate({**problem.model_dump(mode="json"), "constraints": [
        {"variable": "MedInc", "operator": "<=", "value": 2},
        {"variable": "HouseAge", "operator": "==", "value": 12}]}).constraints
    values = sample_candidates(problem, np.random.default_rng(1), 20)
    assert all(item["MedInc"] <= 2 and item["HouseAge"] == 12 for item in values)
    with pytest.raises(ValueError):
        validate_candidate(problem, {**values[0], "MedInc": 10})


def test_llm_adapts_receives_only_own_history_and_remaining_budget():
    problem = model_problem()
    observations = history()
    received = []
    candidate = sample_candidates(problem, np.random.default_rng(22), 1, observations)[0]
    def invoke(payload):
        received.append(payload)
        return {"configuration": candidate}
    optimizer = LLMOptimizer(problem, np.random.default_rng(3), "mock", UsageTracker(), invoke=invoke)
    proposal, details = optimizer.propose(observations, 50)
    assert proposal == candidate and not details["fallback"]
    assert received[0]["observations"] == [o.model_dump(mode="json") for o in observations]
    assert received[0]["remaining_evaluations"] == 50
    assert set(received[0]) == {"problem", "observations", "remaining_evaluations"}
    assert len(observations) == 5


@pytest.mark.parametrize("response", [{"configuration": {"MedInc": 999}}, {"oops": True}, "not json"])
def test_llm_bounded_invalid_retries_and_reproducible_fallback(response):
    attempts = []
    def invoke(_):
        attempts.append(1)
        return response
    left = LLMOptimizer(model_problem(), np.random.default_rng(3), "mock", UsageTracker(), invoke=invoke)
    right = LLMOptimizer(model_problem(), np.random.default_rng(3), "mock", UsageTracker(), invoke=invoke)
    candidate, detail = left.propose(history(), 5)
    assert len(attempts) == 2
    assert detail["fallback"] and detail["invalid_proposals"] == 2
    assert right.propose(history(), 5)[0] == candidate
    validate_candidate(model_problem(), candidate, history())


@pytest.mark.parametrize("family", ["GaussianProcess", "RandomForest", "ExtraTrees"])
def test_bo_fits_supported_family_and_proposes_valid_unseen_candidate(family):
    config = SurrogateConfiguration(model=family, kernel="Matern" if family == "GaussianProcess" else None,
        n_estimators=None if family == "GaussianProcess" else 32,
        uncertainty="predictive_std" if family == "GaussianProcess" else "ensemble_spread")
    observations = history()
    engine = BayesianOptimizer(model_problem(), config, np.random.default_rng(7), 7, 64)
    candidate, details = engine.propose(observations)
    validate_candidate(model_problem(), candidate, observations)
    assert details["fitting_runtime"] > 0
    assert details["acquisition_runtime"] > 0
    engine2 = BayesianOptimizer(model_problem(), config, np.random.default_rng(7), 7, 64)
    assert engine2.propose(observations)[0] == candidate


def test_mixed_encoding_and_acquisition_direction():
    problem = BOProblem.model_validate({"objective": {"column": "y", "direction": "maximize"},
        "decision_variables": [{"name": "x", "type": "continuous", "bounds": [10, 20]},
                               {"name": "category", "type": "categorical", "categories": ["A", "B"]}]})
    assert DomainEncoder(problem).transform([{"x": 15, "category": "B"}]).tolist() == [[.5, 0, 1]]
    assert np.argmax(optimistic_acquisition(np.array([1, 3]), np.zeros(2), maximize=True)) == 1
    assert np.argmax(optimistic_acquisition(np.array([1, 3]), np.zeros(2), maximize=False)) == 0


class LightweightEvaluator:
    def __init__(self):
        self.calls = 0
    def evaluate(self, candidate):
        self.calls += 1
        return float(candidate["MedInc"] + candidate["HouseAge"] * .1 - candidate["AveRooms"] * .2)
    def metadata(self):
        return {"test_only": True}


class SamplingOptimizer:
    def __init__(self, problem, rng, seen=None):
        self.problem, self.rng, self.seen = problem, rng, seen
    def propose(self, observations, remaining):
        if self.seen is not None:
            self.seen.append(deepcopy(observations))
        candidate = sample_candidates(self.problem, self.rng, 1, observations)[0]
        return candidate, {"fallback": False, "invalid_proposals": 0, "proposal_attempts": 1,
            "fitting_runtime": 0., "acquisition_runtime": 0., "proposal_events": []}


def select(profile, problem):
    return recommend_surrogate(profile, problem, invoke=lambda *_: fallback(profile), execution_available=True)


def test_runner_exact_50_budgets_paired_initial_data_and_independent_feedback(tmp_path):
    config = BenchmarkConfig(random_seeds=2, reference_samples=100)
    seen = {name: [] for name in METHODS}
    evaluator = LightweightEvaluator()
    result = execute(tmp_path, config, lambda **_: None, {"eda": "mock", "selection": "mock", "llm_optimizer": "mock"},
        evaluator=evaluator, eda=lambda frame: {}, selector=select,
        llm_factory=lambda problem, rng, seed: SamplingOptimizer(problem, rng, seen[METHODS[0]]),
        bo_factory=lambda problem, cfg, rng, seed: SamplingOptimizer(problem, rng, seen[METHODS[1]]))
    assert evaluator.calls == 2 * (20 + 50 + 50) + 100
    for name in METHODS:
        assert len(seen[name]) == 100
        assert len(result["convergence"][name]["mean"]) == 51
        assert result["metrics"][name]["new_evaluations"] == 100
    for seed_offset in (0, 50):
        assert seen[METHODS[0]][seed_offset] == seen[METHODS[1]][seed_offset]
        assert seen[METHODS[0]][seed_offset + 1][20:] != seen[METHODS[1]][seed_offset + 1][20:]
        assert len(seen[METHODS[0]][seed_offset + 49]) == 69
    assert all(item["shared_initial_data_verified"] for item in result["seeds"])
    assert result["seeds"][0]["initial_dataset_sha256"] != result["seeds"][1]["initial_dataset_sha256"]
    assert result["llm_usage"]["llm_optimizer"]["estimated_cost"] is None
    assert all((tmp_path / name).exists() for name in ("dataset.csv", "eda_report.json", "bo_profile.json",
        "bo_problem.json", "surrogate_report.json", "llm_results.csv", "bo_results.csv", "comparison_summary.json"))
    for method in METHODS:
        paths = result["convergence"][method]["seeds"]
        assert all(np.all(np.diff(path) >= 0) for path in paths)


def test_real_pretrained_evaluator_end_to_end_with_mock_llm(tmp_path):
    evaluator = HousingEvaluator()
    result = execute(tmp_path, BenchmarkConfig(random_seeds=1, optimization_iterations=3, reference_samples=100, candidate_pool_size=64),
        lambda **_: None, {"eda": "mock", "selection": "mock", "llm_optimizer": "mock"},
        evaluator=evaluator, eda=lambda frame: {}, selector=select,
        llm_factory=lambda problem, rng, seed: SamplingOptimizer(problem, rng))
    assert result["evaluation_counts"]["total"] == evaluator.calls == 126
    assert result["metrics"]["agent_selected_bo"]["new_evaluations"] == 3
    assert result["paired_comparison"]["mean_difference_95_percent_t_interval"] is None


def test_metrics_minimization_reference_and_threshold():
    config = BenchmarkConfig(optimization_iterations=2, random_seeds=2, optimization_direction="minimize", threshold=3)
    trajectories = {METHODS[0]: [[5, 4, 3], [6, 4, 3]], METHODS[1]: [[5, 2, 1], [6, 3, 2]]}
    rows = [{"method": name, "fallback": False, "invalid_proposals": 0, "proposal_attempts": 1,
             "recommendation_runtime": 0.1, "evaluation_runtime": 0.2} for name in METHODS for _ in range(4)]
    result = summarize(trajectories, rows, config, reference=2)
    assert result["metrics"][METHODS[1]]["estimated_simple_regret"]["mean"] == -.5
    assert result["metrics"][METHODS[1]]["evaluations_to_threshold"] == [1, 1]
    assert result["paired_comparison"]["bo_win_rate"] == 1
    assert result["paired_comparison"]["oriented_bo_minus_llm"]["mean"] == 1.5


def test_job_api_errors_limits_and_path_safety(tmp_path, monkeypatch):
    from eda_agent.benchmark import api
    monkeypatch.setenv("BENCHMARK_RUN_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    assert client.get("/benchmark/models").status_code == 200
    assert client.post("/benchmark/run", json={}).status_code == 422
    assert client.get("/benchmark/not-a-uuid/status").status_code == 404
    assert client.get(f"/benchmark/{uuid4()}/results").status_code == 404
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert client.post("/benchmark/run", json={"optimization_iterations": 999}).status_code == 422
    assert api._capacity.acquire(blocking=False)
    try:
        assert client.post("/benchmark/run", json={}).status_code == 429
    finally:
        api._capacity.release()


def test_background_submission_persistence_restart_and_download(tmp_path, monkeypatch):
    from eda_agent.benchmark import api
    monkeypatch.setenv("BENCHMARK_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = []
    monkeypatch.setattr(api._executor, "submit", lambda *args: captured.append(args))
    client = TestClient(app)
    response = client.post("/benchmark/run", json={"random_seeds": 1, "optimization_iterations": 1})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    try:
        assert client.get(f"/benchmark/{run_id}/status").json()["status"] == "queued"
        assert client.get(f"/benchmark/{run_id}/results").status_code == 409
        assert client.get(f"/benchmark/{run_id}/download").status_code == 200
        assert client.get(f"/benchmark/{run_id}/download?artifact=../../.env").status_code == 422
        api._active.discard(run_id)
        assert client.get(f"/benchmark/{run_id}/status").json()["status"] == "interrupted"
        assert (tmp_path / run_id / "config.json").exists()
        assert captured
    finally:
        api._active.discard(run_id)
        api._capacity.release()


def test_completed_background_job_results_convergence_and_download(tmp_path, monkeypatch):
    from eda_agent.benchmark import api
    monkeypatch.setenv("BENCHMARK_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    def mock_execute(path, config, progress, names):
        return execute(path, config, progress, names, evaluator=LightweightEvaluator(), eda=lambda _: {}, selector=select,
            llm_factory=lambda p, r, s: SamplingOptimizer(p, r),
            bo_factory=lambda p, c, r, s: SamplingOptimizer(p, r))
    monkeypatch.setattr(api, "execute", mock_execute)
    monkeypatch.setattr(api._executor, "submit", lambda function, *args: function(*args))
    client = TestClient(app)
    response = client.post("/benchmark/run", json={"random_seeds": 1, "optimization_iterations": 2, "reference_samples": 100})
    assert response.status_code == 202
    endpoint = "/benchmark/" + response.json()["run_id"]
    status = client.get(endpoint + "/status").json()
    assert status["status"] == "completed"
    assert status["completed_evaluations"] == status["total_evaluations"] == 4
    results = client.get(endpoint + "/results")
    assert results.status_code == 200
    assert len(results.json()["iterations"]) == 4
    assert client.get(endpoint + "/convergence").status_code == 200
    for artifact in ("all", "summary", "llm", "bo"):
        download = client.get(endpoint + "/download", params={"artifact": artifact})
        assert download.status_code == 200
        assert "attachment" in download.headers["Content-Disposition"]


def test_background_failure_is_reported_and_capacity_released(tmp_path, monkeypatch):
    from eda_agent.benchmark import api
    monkeypatch.setenv("BENCHMARK_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    def fail(*_):
        raise RuntimeError("private credential must not be exposed")
    monkeypatch.setattr(api, "execute", fail)
    monkeypatch.setattr(api._executor, "submit", lambda function, *args: function(*args))
    client = TestClient(app)
    response = client.post("/benchmark/run", json={})
    endpoint = "/benchmark/" + response.json()["run_id"]
    status = client.get(endpoint + "/status").json()
    assert status["status"] == "failed"
    assert "private credential" not in status["error"]
    assert client.post("/benchmark/run", json={}).status_code == 202


def test_baseline_langchain_structured_output_and_usage_accounting(monkeypatch):
    from eda_agent.benchmark import llm_optimizer
    from eda_agent.benchmark.schemas import CandidateProposal
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import LLMResult, ChatGeneration
    tracker = UsageTracker()
    candidate = {"MedInc": 7, "HouseAge": 20, "AveRooms": 6}
    class Model:
        def with_structured_output(self, schema):
            assert schema is CandidateProposal
            return self
        def invoke(self, messages, config):
            assert config["callbacks"] == [tracker]
            payload = json.loads(messages[1]["content"])
            assert len(payload["observations"]) == 5
            call_id = uuid4()
            tracker.on_chat_model_start({}, messages, run_id=call_id)
            tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="valid",
                usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}))]]), run_id=call_id)
            return CandidateProposal(configuration=candidate)
    settings = []
    monkeypatch.setattr(llm_optimizer, "init_chat_model", lambda model_name, **kwargs: settings.append(kwargs) or Model())
    optimizer = LLMOptimizer(model_problem(), np.random.default_rng(1), "mock", tracker)
    assert optimizer.propose(history(), 50)[0] == candidate
    assert settings == [{"temperature": 0, "timeout": 45, "max_retries": 0}]
    usage = tracker.report(BenchmarkConfig(input_cost_per_million=1, output_cost_per_million=2))
    assert usage["model_calls"] == 1
    assert usage["estimated_cost"] == pytest.approx(.00014)
    assert usage["input_tokens"] == 100


def test_provider_failures_are_separate_from_invalid_proposals():
    def fail(_):
        raise ConnectionError("unavailable")
    optimizer = LLMOptimizer(model_problem(), np.random.default_rng(3), "mock", UsageTracker(), invoke=fail)
    _, details = optimizer.propose(history(), 5)
    assert details["fallback"]
    assert details["invalid_proposals"] == 0
    assert all(event["kind"] == "provider" for event in details["proposal_events"])


def test_shutdown_interrupts_job_without_claiming_completion(tmp_path, monkeypatch):
    from eda_agent.benchmark import api
    monkeypatch.setenv("BENCHMARK_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(api._executor, "submit", lambda function, *args: function(*args))
    def interrupted(path, config, progress, names):
        api.shutdown()
        progress(phase="stop")
    monkeypatch.setattr(api, "execute", interrupted)
    client = TestClient(app)
    try:
        response = client.post("/benchmark/run", json={})
        endpoint = "/benchmark/" + response.json()["run_id"]
        assert client.get(endpoint + "/status").json()["status"] == "interrupted"
        assert client.get(endpoint + "/results").status_code == 409
    finally:
        api.startup()


def test_app_lifespan_registers_cooperative_shutdown():
    from eda_agent.benchmark import api
    try:
        with TestClient(app) as client:
            assert not api._stopping.is_set()
            assert client.get("/health").status_code == 200
        assert api._stopping.is_set()
    finally:
        api.startup()


def test_atomic_artifact_retries_transient_windows_locks(tmp_path, monkeypatch):
    from eda_agent import bo_service
    target = tmp_path / "status.json"
    target.write_text("old")
    original = Path.replace
    attempts = []
    def transient(self, destination):
        attempts.append(1)
        if len(attempts) < 3:
            assert target.read_text() == "old"
            raise PermissionError(13, "locked", str(destination))
        return original(self, destination)
    monkeypatch.setattr(Path, "replace", transient)
    monkeypatch.setattr(bo_service.time, "sleep", lambda _: None)
    bo_service.write_artifact(target, "new")
    assert target.read_text() == "new"
    assert len(attempts) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_artifact_permanent_denial_preserves_previous_file(tmp_path, monkeypatch):
    from eda_agent import bo_service
    target = tmp_path / "status.json"
    target.write_text("old")
    attempts = []
    def denied(self, destination):
        attempts.append(1)
        raise PermissionError(13, "locked", str(destination))
    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(bo_service.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        bo_service.write_artifact(target, "new")
    assert len(attempts) == 6
    assert target.read_text() == "old"


def test_permission_diagnostic_includes_filename_without_private_error_message():
    from eda_agent.benchmark.api import failure_detail
    message = failure_detail(PermissionError(13, "secret must not appear", "status.json"))
    assert "status.json" in message
    assert "OneDrive" in message
    assert "secret must not appear" not in message
