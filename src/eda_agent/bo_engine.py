"""Extension contracts. No evaluator or fitting adapter is implicitly supplied by a CSV.

Implementations must enforce domain and constraint feasibility, exclude previously
observed candidates, and explicitly handle constraints on non-controllable outputs.
Tree spread must never be represented as calibrated Gaussian uncertainty.
"""
from typing import Mapping, Protocol, Sequence

from .bo import BOProblem, Category
from .surrogates import SurrogateConfiguration

Candidate = Mapping[str, Category]


class ObjectiveEvaluator(Protocol):
    def evaluate(self, candidate: Candidate) -> Mapping[str, float]:
        """Return measured objective and any constrained output values."""
        ...


class Preprocessor(Protocol):
    def fit_transform(self, observations: Sequence[Candidate]) -> Sequence[Sequence[float]]: ...
    def transform(self, candidates: Sequence[Candidate]) -> Sequence[Sequence[float]]: ...


class SurrogateAdapter(Protocol):
    def fit(self, inputs: Sequence[Sequence[float]], targets: Sequence[float],
            configuration: SurrogateConfiguration) -> None: ...
    def predict(self, inputs: Sequence[Sequence[float]]) -> Sequence[float]: ...


class UncertaintyEstimator(Protocol):
    def estimate(self, surrogate: SurrogateAdapter, inputs: Sequence[Sequence[float]]) -> Sequence[float]: ...


class Acquisition(Protocol):
    def score(self, means: Sequence[float], uncertainty: Sequence[float], *,
              incumbent: float, maximize: bool) -> Sequence[float]: ...


class BOEngine(Protocol):
    def propose(self, problem: BOProblem, configuration: SurrogateConfiguration,
                observations: Sequence[Candidate]) -> Candidate: ...
    def run(self, problem: BOProblem, configuration: SurrogateConfiguration,
            observations: Sequence[Candidate], evaluator: ObjectiveEvaluator, *, budget: int) -> str:
        """Persist evaluations and refit between steps; return a durable run identifier."""
        ...
