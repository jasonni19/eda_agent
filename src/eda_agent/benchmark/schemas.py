"""Benchmark contracts; budgets count successful new evaluator calls only."""
from typing import Literal
from pydantic import Field, StrictInt
from ..bo import StrictModel, Category

METHODS = ("llm_optimizer", "agent_selected_bo")


class BenchmarkConfig(StrictModel):
    model_id: Literal["california-housing-xgb"] = "california-housing-xgb"
    name: str = Field(default="Optimization comparison", min_length=1, max_length=120)
    initial_samples: StrictInt = Field(default=20, ge=5, le=100)
    optimization_iterations: StrictInt = Field(default=50, ge=1, le=100)
    random_seeds: StrictInt = Field(default=10, ge=1, le=10)
    seed: StrictInt = Field(default=0, ge=0, le=1000000)
    optimization_direction: Literal["maximize", "minimize"] = "maximize"
    threshold: float | None = None
    reference_samples: StrictInt = Field(default=4096, ge=100, le=20000)
    candidate_pool_size: StrictInt = Field(default=1024, ge=64, le=4096)
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)


class CandidateProposal(StrictModel):
    configuration: dict[str, Category]


class Observation(StrictModel):
    configuration: dict[str, Category]
    objective: float


class RunStatus(StrictModel):
    run_id: str
    name: str
    status: Literal["queued", "running", "completed", "failed", "interrupted"]
    phase: str = "queued"
    seed: int | None = None
    method: str | None = None
    iteration: int = 0
    completed_evaluations: int = 0
    total_evaluations: int
    method_completed: dict[str, int] = Field(default_factory=lambda: {name: 0 for name in METHODS})
    error: str | None = None


class ModelInfo(StrictModel):
    id: str
    name: str
    available: bool
    source: str
    revision: str
    adjustable_features: list[str]
    limitation: str
    problem: dict


class RunCreated(StrictModel):
    run_id: str


class HistoryItem(StrictModel):
    run_id: str
    name: str
    status: str


class BenchmarkResults(StrictModel):
    summary: dict
    config: dict
    iterations: list[dict]


class ConvergenceResults(StrictModel):
    curves: dict
    reference: dict
