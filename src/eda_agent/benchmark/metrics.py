"""Paired performance metrics. Reference estimates are never sent to optimizers."""
import numpy as np
from scipy.stats import t
from .schemas import METHODS, BenchmarkConfig


def describe(values) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else None,
            "median": float(np.median(array)), "minimum": float(array.min()), "maximum": float(array.max())}


def summarize(trajectories: dict[str, list[list[float]]], rows: list[dict],
              config: BenchmarkConfig, reference: float) -> dict:
    maximize = config.optimization_direction == "maximize"
    metrics, curves = {}, {}
    for method in METHODS:
        paths = np.asarray(trajectories[method], dtype=float)
        finals = paths[:, -1]
        regrets = reference - finals if maximize else finals - reference
        reached = ([next((i for i, value in enumerate(path) if
                          (value >= config.threshold if maximize else value <= config.threshold)), None)
                    for path in paths] if config.threshold is not None else [])
        data = [item for item in rows if item["method"] == method]
        metrics[method] = {"final_best": describe(finals), "estimated_simple_regret": describe(regrets),
            "auc": describe([np.sum((path[:-1] + path[1:]) * .5) for path in paths]),
            "evaluations_to_threshold": reached or None,
            "mean_evaluations_to_threshold_among_successes": float(np.mean([i for i in reached if i is not None]))
                if any(i is not None for i in reached) else None,
            "success_rate": sum(i is not None for i in reached) / len(reached) if reached else None,
            "mean_iterative_runtime_seconds": sum(item["recommendation_runtime"] + item["evaluation_runtime"] for item in data) / len(paths),
            "fallback_count": sum(item["fallback"] for item in data),
            "invalid_candidate_count": sum(item["invalid_proposals"] for item in data),
            "proposal_attempts": sum(item["proposal_attempts"] for item in data),
            "fallback_frequency": sum(item["fallback"] for item in data) / len(data) if data else 0,
            "invalid_candidate_rate": sum(item["invalid_proposals"] for item in data) / max(1, sum(item["proposal_attempts"] for item in data)),
            "new_evaluations": len(data)}
        curves[method] = {"mean": paths.mean(axis=0).tolist(),
            "std": paths.std(axis=0, ddof=1).tolist() if len(paths) > 1 else [None] * paths.shape[1],
            "seeds": paths.tolist()}
    # Positive oriented difference means BO achieved the better objective.
    diff = np.asarray(trajectories[METHODS[1]])[:, -1] - np.asarray(trajectories[METHODS[0]])[:, -1]
    if not maximize:
        diff = -diff
    ci = None
    if len(diff) > 1:
        half = float(t.ppf(.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff)))
        ci = [float(diff.mean() - half), float(diff.mean() + half)]
    return {"metrics": metrics, "convergence": curves, "paired_comparison": {
        "oriented_bo_minus_llm": describe(diff), "bo_win_rate": float(np.mean(diff > 0)),
        "tie_rate": float(np.mean(diff == 0)), "mean_difference_95_percent_t_interval": ci,
        "interval_note": "Paired Student-t interval; small-seed estimates are uncertain."},
        "reference": {"estimated_optimum": reference, "procedure": "Independent seeded uniform feasible search",
                      "evaluations": config.reference_samples, "regret_label": "estimated simple regret",
                      "note": "Reference may be beaten; signed estimated regret can be negative."}}
