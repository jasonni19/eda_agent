"""Pinned native XGBoost evaluator. No pickle loading, remote imports, or training."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np
from ..bo import BOProblem
from .domain import validate_candidate
from .schemas import ModelInfo

REVISION = "3b04cdba5f8f9433bb135dfc3be2ef5c64ff5179"
SOURCE = "https://huggingface.co/shalev396/california-housing"
ARTIFACT_SHA256 = "cc2160311b6ff9b338997e541a632895549e12921ba40b73b040a3d8d0ee5a10"
CONFIG_SHA256 = "54123721e71f830ecf3891fcaffd85c68cf06e68fa66b5b4bb8179924093cf30"
RAW = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
FEATURES = RAW + ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
FIXED = {"AveBedrms": 1.10, "Population": 1425, "AveOccup": 3.07, "Latitude": 35.63, "Longitude": -119.57}


class GroundTruthEvaluator(Protocol):
    def evaluate(self, candidate: dict) -> float: ...
    def metadata(self) -> dict: ...


def model_directory() -> Path:
    return Path(os.getenv("BENCHMARK_MODEL_DIR", "benchmark_models/california-housing"))


def model_problem(direction: str = "maximize") -> BOProblem:
    # A 3D fixed-context slice of the author's documented public input sliders.
    return BOProblem.model_validate({"objective": {"column": "price_usd", "direction": direction},
        "decision_variables": [
            {"name": "MedInc", "type": "continuous", "bounds": [0.5, 15]},
            {"name": "HouseAge", "type": "integer", "bounds": [1, 52]},
            {"name": "AveRooms", "type": "continuous", "bounds": [1.1, 20]}], "constraints": []})


def model_info() -> ModelInfo:
    directory = model_directory()
    available = all((directory / name).is_file() for name in ("xgb.ubj", "config.json"))
    available = available and all(importlib.util.find_spec(name) is not None for name in ("xgboost", "sklearn"))
    return ModelInfo(id="california-housing-xgb", name="California housing · fixed-context 3D slice",
        available=available, source=SOURCE, revision=REVISION,
        adjustable_features=["MedInc", "HouseAge", "AveRooms"], problem=model_problem().model_dump(mode="json"),
        limitation="Pretrained 1990 census regressor; three inputs vary within author UI ranges, five inputs stay fixed. "
                   "A prediction benchmark, not causal housing advice. Joint combinations may be off the training manifold.")


class HousingEvaluator:
    def __init__(self):
        from xgboost import Booster
        directory = model_directory()
        for name, expected in (("xgb.ubj", ARTIFACT_SHA256), ("config.json", CONFIG_SHA256)):
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Pretrained artifact checksum mismatch; download the pinned version again.")
        self._config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if self._config["features"] != FEATURES or self._config["raw_features"] != RAW:
            raise ValueError("Unexpected pretrained model feature order.")
        self._booster = Booster()
        self._booster.load_model(str(directory / "xgb.ubj"))
        self._booster.set_param({"device": "cpu", "nthread": 1})
        if self._booster.num_features() != len(FEATURES):
            raise ValueError("Unexpected pretrained input dimension.")
        self.problem = model_problem()
        self.calls = 0

    @staticmethod
    def preprocess(candidate: dict) -> np.ndarray:
        validate_candidate(model_problem(), candidate)
        raw = {**FIXED, **candidate}
        # Exact author's float64, unscaled feature engineering and column order.
        values = [raw[name] for name in RAW]
        return np.asarray([values + [raw["AveRooms"], raw["AveBedrms"] / raw["AveRooms"], raw["AveOccup"]]], dtype=np.float64)

    def evaluate(self, candidate: dict) -> float:
        matrix = self.preprocess(candidate)
        value = float(self._booster.inplace_predict(matrix)[0])
        self.calls += 1
        result = round(max(value, 0.0) * 100000, 2)
        if not np.isfinite(result):
            raise ValueError("Evaluator returned a non-finite objective.")
        return result

    def metadata(self) -> dict:
        import xgboost
        return {**model_info().model_dump(mode="json"), "artifact_sha256": ARTIFACT_SHA256,
            "config_sha256": CONFIG_SHA256, "artifact_file": "xgb.ubj", "fixed_inputs": FIXED,
            "feature_order": FEATURES, "preprocessing": "unscaled float64; rooms, bedrooms/rooms, occupancy ratios",
            "output": "round(max(raw_prediction, 0) * 100000, 2)", "runtime_xgboost_version": xgboost.__version__,
            "domain_source": "https://huggingface.co/spaces/shalev396/california-housing/blob/main/app.py",
            "domain_source_sha256": hashlib.sha256((model_directory() / "source_app.py").read_bytes()).hexdigest()
                if (model_directory() / "source_app.py").exists() else None,
            "license": "MIT", "frozen": True, "trained_here": False}
