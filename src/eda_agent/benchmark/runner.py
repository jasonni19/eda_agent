"""Equal-budget paired experiment orchestration. Optimizers receive no evaluator object."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import platform
from time import perf_counter
from pathlib import Path
from collections.abc import Callable

import numpy as np
import pandas as pd

from ..agent import run_eda_detailed
from ..bo_analyzer import analyze_bo_problem
from ..surrogates import recommend_surrogate
from .bo_optimizer import BayesianOptimizer
from .domain import sample_candidates, validate_candidate
from .llm_optimizer import LLMOptimizer, UsageTracker, PROMPT_VERSION, SYSTEM_PROMPT
from .metrics import summarize
from .pretrained import GroundTruthEvaluator, HousingEvaluator, model_problem
from .schemas import METHODS, BenchmarkConfig, Observation
from .storage import save_json, save_csv


def execute(directory: Path, config: BenchmarkConfig, progress: Callable, model_names: dict[str, str], *,
            evaluator: GroundTruthEvaluator | None = None, eda=None, selector=None,
            llm_factory=None, bo_factory=None) -> dict:
    """Test seams accept mocks; the production route always uses the pinned evaluator."""
    start = perf_counter()
    progress(phase="Loading verified pretrained model")
    evaluator = evaluator or HousingEvaluator()
    problem = model_problem(config.optimization_direction)
    save_json(directory / "ground_truth_metadata.json", evaluator.metadata())
    save_json(directory / "bo_problem.json", problem.model_dump(mode="json"))
    runtime = {"initial_evaluation_seconds": 0.0, "eda_seconds": 0.0, "selection_seconds": 0.0,
               "reference_evaluation_seconds": 0.0, "fitting_seconds": 0.0, "acquisition_seconds": 0.0,
               "ground_truth_new_evaluation_seconds": 0.0, "recommendation_seconds": 0.0}
    trackers = {phase: UsageTracker() for phase in ("eda", "selection", "llm_optimizer")}
    trajectories = {method: [] for method in METHODS}
    rows, selections, seed_metadata = [], [], []
    all_calls = 0
    for offset in range(config.random_seeds):
        seed = config.seed + offset
        rngs = [np.random.default_rng(stream) for stream in np.random.SeedSequence(seed).spawn(3)]
        seed_dir = directory / f"seed_{seed}"
        seed_dir.mkdir()
        progress(seed=seed, method=None, iteration=0, phase="Generating shared initial observations")
        initial = []
        t0 = perf_counter()
        for candidate in sample_candidates(problem, rngs[0], config.initial_samples):
            initial.append(Observation(configuration=candidate, objective=evaluator.evaluate(candidate)))
            all_calls += 1
        runtime["initial_evaluation_seconds"] += perf_counter() - t0
        frame = pd.DataFrame([{**item.configuration, problem.objective.column: item.objective} for item in initial])
        frame.to_csv(seed_dir / "dataset.csv", index=False)
        initial_digest = hashlib.sha256((seed_dir / "dataset.csv").read_bytes()).hexdigest()
        save_json(seed_dir / "bo_problem.json", problem.model_dump(mode="json"))
        progress(phase="EDA on initial observations only")
        t0 = perf_counter()
        if eda is None:
            report, _ = run_eda_detailed(seed_dir / "dataset.csv", seed_dir / "eda_report.json",
                                        model_names["eda"], callbacks=[trackers["eda"]])
        else:
            report = eda(frame.copy(deep=True))
        report_json = report.model_dump(mode="json") if hasattr(report, "model_dump") else report
        save_json(seed_dir / "eda_report.json", report_json)
        runtime["eda_seconds"] += perf_counter() - t0
        profile = analyze_bo_problem(frame, report_json, problem)
        save_json(seed_dir / "bo_profile.json", profile.model_dump(mode="json"))
        progress(phase="Selecting surrogate from initial profile")
        t0 = perf_counter()
        selection = (recommend_surrogate(profile, problem, callbacks=[trackers["selection"]],
                     model_name=model_names["selection"], execution_available=True)
                     if selector is None else selector(profile, problem))
        runtime["selection_seconds"] += perf_counter() - t0
        save_json(seed_dir / "surrogate_report.json", selection.model_dump(mode="json"))
        selections.append({"seed": seed, "report": selection.model_dump(mode="json"),
                           "profile": profile.model_dump(mode="json")})
        if offset == 0:
            for filename in ("dataset.csv", "eda_report.json", "bo_profile.json", "surrogate_report.json"):
                (directory / filename).write_bytes((seed_dir / filename).read_bytes())
        # Deep copies ensure methods cannot mutate their shared start or each other's feedback.
        histories = {method: deepcopy(initial) for method in METHODS}
        baseline = (LLMOptimizer(problem.model_copy(deep=True), rngs[1], model_names["llm_optimizer"], trackers["llm_optimizer"])
                    if llm_factory is None else llm_factory(problem.model_copy(deep=True), rngs[1], seed))
        bo = (BayesianOptimizer(problem.model_copy(deep=True), selection.configuration, rngs[2], seed, config.candidate_pool_size)
              if bo_factory is None else bo_factory(problem.model_copy(deep=True), selection.configuration, rngs[2], seed))
        best = max if config.optimization_direction == "maximize" else min
        paths = {method: [best(item.objective for item in initial)] for method in METHODS}
        for iteration in range(1, config.optimization_iterations + 1):
            # Alternate evaluation order across seeds; methods still see only their own history.
            order = METHODS if offset % 2 == 0 else tuple(reversed(METHODS))
            for method in order:
                progress(method=method, iteration=iteration, phase="Optimizing")
                optimizer = baseline if method == METHODS[0] else bo
                t0 = perf_counter()
                candidate, detail = optimizer.propose(deepcopy(histories[method]), config.optimization_iterations - iteration + 1)
                validate_candidate(problem, candidate, histories[method])
                recommendation_time = perf_counter() - t0
                t0 = perf_counter()
                objective = evaluator.evaluate(candidate)
                all_calls += 1
                evaluation_time = perf_counter() - t0
                observation = Observation(configuration=dict(candidate), objective=objective)
                histories[method].append(observation)
                paths[method].append(best(paths[method][-1], objective))
                row = {"seed": seed, "method": method, "iteration": iteration,
                    "configuration": candidate, "objective": objective, "best_objective": paths[method][-1],
                    "estimated_simple_regret": None, "evaluation_runtime": evaluation_time,
                    "recommendation_runtime": recommendation_time, "surrogate_warnings": [], **detail}
                rows.append(row)
                runtime["fitting_seconds"] += detail["fitting_runtime"]
                runtime["acquisition_seconds"] += detail["acquisition_runtime"]
                runtime["ground_truth_new_evaluation_seconds"] += evaluation_time
                runtime["recommendation_seconds"] += recommendation_time
                save_csv(seed_dir / ("llm_results.csv" if method == METHODS[0] else "bo_results.csv"),
                         [item for item in rows if item["seed"] == seed and item["method"] == method])
                progress(completed_method=method)
        assert hashlib.sha256((seed_dir / "dataset.csv").read_bytes()).hexdigest() == initial_digest
        seed_metadata.append({"seed": seed, "initial_dataset_sha256": initial_digest,
            "initial_observations": [item.model_dump(mode="json") for item in initial],
            "shared_initial_data_verified": all(histories[method][:config.initial_samples] == initial for method in METHODS)})
        for method in METHODS:
            trajectories[method].append(paths[method])
    # Separate RNG and post-experiment evaluation keep reference information out of both methods.
    progress(method=None, phase="Estimating independent reference and computing paired metrics")
    t0 = perf_counter()
    reference_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x524546]))
    values = [evaluator.evaluate(candidate) for candidate in sample_candidates(problem, reference_rng, config.reference_samples)]
    all_calls += len(values)
    reference = (max if config.optimization_direction == "maximize" else min)(values)
    runtime["reference_evaluation_seconds"] = perf_counter() - t0
    for row in rows:
        row["estimated_simple_regret"] = (reference - row["best_objective"] if config.optimization_direction == "maximize"
                                         else row["best_objective"] - reference)
    for method, filename in zip(METHODS, ("llm_results.csv", "bo_results.csv")):
        save_csv(directory / filename, [row for row in rows if row["method"] == method])
        for meta in seed_metadata:
            save_csv(directory / f"seed_{meta['seed']}" / filename,
                     [row for row in rows if row["method"] == method and row["seed"] == meta["seed"]])
    summary = summarize(trajectories, rows, config, reference)
    summary["runtime"] = {**runtime, "total_seconds": perf_counter() - start,
                          "note": "Fitting and acquisition are included in recommendation_seconds; do not sum overlapping fields."}
    summary["llm_usage"] = {name: tracker.report(config) for name, tracker in trackers.items()}
    summary["evaluation_counts"] = {"shared_initial": config.initial_samples * config.random_seeds,
        "llm_optimizer": config.optimization_iterations * config.random_seeds,
        "agent_selected_bo": config.optimization_iterations * config.random_seeds,
        "reference": config.reference_samples, "total": all_calls}
    summary["surrogate_selections"] = selections
    summary["seeds"] = seed_metadata
    summary["reproducibility"] = {"model_names": model_names, "prompt_version": PROMPT_VERSION,
        "baseline_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "python": platform.python_version(), "packages": {name: importlib.metadata.version(name)
             for name in ("numpy", "pandas", "scikit-learn", "xgboost", "langchain", "langchain-openai")},
        "acquisition": "normalized GP predictive std or tree ensemble spread; optimistic score kappa=2; random feasible pool",
        "note": "Seeds reproduce sampling and surrogate fitting, not hosted LLM responses or API latency. All observations are persisted."}
    save_json(directory / "comparison_summary.json", summary)
    save_json(directory / "iterations.json", rows)
    return summary
