"""Executable scaled/encoded surrogate adapters and optimistic acquisition search."""
from time import perf_counter
import warnings
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from ..surrogates import SurrogateConfiguration, SurrogateModel
from ..bo import _same_value
from .domain import sample_candidates


class DomainEncoder:
    def __init__(self, problem):
        self.problem = problem

    def transform(self, candidates):
        rows = []
        for candidate in candidates:
            row = []
            for variable in self.problem.decision_variables:
                value = candidate[variable.name]
                if variable.type == "categorical":
                    row.extend(float(_same_value(value, choice)) for choice in variable.categories)
                else:
                    low, high = variable.bounds
                    row.append((value - low) / (high - low))
            rows.append(row)
        return np.asarray(rows, dtype=float)


class FittedSurrogate:
    def __init__(self, configuration, seed):
        self.configuration = SurrogateConfiguration.model_validate(configuration)
        model = self.configuration.model
        if model == SurrogateModel.gaussian_process:
            self.estimator = GaussianProcessRegressor(kernel=ConstantKernel(1.0, (0.01, 100)) * Matern(nu=2.5),
                alpha=1e-6, normalize_y=False, n_restarts_optimizer=0, random_state=seed)
        else:
            cls = RandomForestRegressor if model == SurrogateModel.random_forest else ExtraTreesRegressor
            self.estimator = cls(n_estimators=self.configuration.n_estimators, random_state=seed, n_jobs=1)

    def fit(self, inputs, targets):
        self.mean = float(np.mean(targets))
        self.scale = float(np.std(targets)) or 1.0
        self.estimator.fit(inputs, (np.asarray(targets) - self.mean) / self.scale)

    def predict_with_uncertainty(self, inputs):
        if self.configuration.model == SurrogateModel.gaussian_process:
            means, std = self.estimator.predict(inputs, return_std=True)
        else:
            predictions = np.asarray([tree.predict(inputs) for tree in self.estimator.estimators_])
            means, std = predictions.mean(axis=0), predictions.std(axis=0, ddof=1)
        return means * self.scale + self.mean, std * self.scale


def optimistic_acquisition(means, uncertainty, *, maximize, exploration=2.0):
    """GP-UCB style score; tree spread is an explicitly heuristic optimism bonus."""
    return (means if maximize else -means) + exploration * uncertainty


class BayesianOptimizer:
    def __init__(self, problem, configuration, rng, seed, pool_size=1024):
        self.problem, self.configuration = problem, configuration
        self.rng, self.seed, self.pool_size = rng, seed, pool_size
        self.encoder = DomainEncoder(problem)

    def propose(self, history, remaining=None):
        start = perf_counter()
        surrogate = FittedSurrogate(self.configuration, self.seed)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            surrogate.fit(self.encoder.transform([item.configuration for item in history]), [item.objective for item in history])
        fitting = perf_counter() - start
        start = perf_counter()
        candidates = sample_candidates(self.problem, self.rng, self.pool_size, history, require_count=False)
        means, std = surrogate.predict_with_uncertainty(self.encoder.transform(candidates))
        scores = optimistic_acquisition(means, std, maximize=self.problem.objective.direction == "maximize")
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite acquisition scores.")
        candidate = candidates[int(np.argmax(scores))]
        acquisition = perf_counter() - start
        return candidate, {"fallback": False, "invalid_proposals": 0, "proposal_attempts": 1,
                           "fitting_runtime": fitting, "acquisition_runtime": acquisition, "proposal_events": [],
                           "surrogate_warnings": list(dict.fromkeys(str(item.message) for item in captured))}
