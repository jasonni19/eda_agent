"""California Housing Price Predictor: XGBoost + PyTorch MLP regressors for 1990 census districts.

Single source of truth for the input schema, the feature engineering, the MLP architecture and
inference. The training code (../training), the Hugging Face Space and `handler.py` all import this
file, so evaluation and serving run exactly the same code path.

Files next to this module (written by training/src/export.py):
    config.json         input schema + engineered features, exported models, default model, MLP init
                        args (read by PyTorchModelHubMixin), target units, library versions
    xgb.ubj             XGBoost booster (native format, `Booster.save_model`), trimmed to the best round
    model.safetensors   PyTorch MLP weights (`HousingMLP.save_pretrained`)
    scaler.joblib       StandardScaler fit on the training split (used by the MLP only)

    import model
    predictor = model.load("path/to/this/repo", device="cpu")
    predictor.predict({"MedInc": 8.33, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
                       "Population": 322, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23})
    # -> {"price_usd": 4xx,xxx.x, "model": "xgboost"}
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

REPO_ID = "shalev396/california-housing"
FRAMEWORK = "xgboost"

# The 8 census features of sklearn's fetch_california_housing, in dataset column order.
RAW_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
# Classic ratios of the raw-dataset recipe. sklearn's version is already per household, so
# rooms_per_household == AveRooms and population_per_household == AveOccup (kept to mirror the recipe);
# bedrooms_per_room = AveBedrms / AveRooms is the genuinely new signal.
ENGINEERED_FEATURES = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
FEATURES = RAW_FEATURES + ENGINEERED_FEATURES      # model input order (11 columns)
TARGET = "MedHouseVal"                             # district median house value in $100k
TARGET_UNIT_USD = 100_000

XGB_FILE = "xgb.ubj"
MLP_FILE = "model.safetensors"
SCALER_FILE = "scaler.joblib"


def cuda_available() -> bool:
    """The MLP can run on CUDA; XGBoost inference always runs on the CPU."""
    return torch.cuda.is_available()


# --------------------------------------------------------------------------- preprocessing
def engineer(columns: Mapping) -> dict[str, np.ndarray]:
    """Raw columns (dict of scalars/arrays or a DataFrame) -> float64 arrays for all 11 FEATURES."""
    missing = [c for c in RAW_FEATURES if c not in columns]
    if missing:
        raise ValueError(f"missing feature(s): {', '.join(missing)}; expected {RAW_FEATURES}")
    out = {}
    for name in RAW_FEATURES:
        try:
            values = np.atleast_1d(np.asarray(columns[name], dtype=np.float64))
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric, got {columns[name]!r}") from None
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite number")
        out[name] = values
    if (out["AveRooms"] <= 0).any():
        raise ValueError("AveRooms must be > 0")
    out["rooms_per_household"] = out["AveRooms"]
    out["bedrooms_per_room"] = out["AveBedrms"] / out["AveRooms"]
    out["population_per_household"] = out["AveOccup"]
    return out


def to_matrix(columns: Mapping) -> np.ndarray:
    """Raw columns -> (n, 11) float64 feature matrix in FEATURES order (unscaled; XGBoost input)."""
    feats = engineer(columns)
    return np.column_stack([feats[name] for name in FEATURES])


# --------------------------------------------------------------------------- models
class HousingMLP(
    nn.Module,
    PyTorchModelHubMixin,
    tags=["ml-lab", "tabular-regression"],
    repo_url="https://github.com/shalev396/ml-lab",
    pipeline_tag="tabular-regression",
    license="mit",
):
    """[Linear -> ReLU -> Dropout] per hidden layer -> Linear(1): standardized features -> $100k.

    Default 11 -> 256 -> 128 -> 64 -> 1 (44,289 parameters).
    """

    def __init__(self, in_features: int = len(FEATURES), hidden_units: list[int] = (256, 128, 64),
                 dropout: float = 0.15):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for units in hidden_units:
            layers += [nn.Linear(prev, units), nn.ReLU(), nn.Dropout(dropout)]
            prev = units
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, in_features) standardized features -> (N, 1) house value in $100k."""
        return self.net(x)


def load_booster(path: str | Path):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(path))
    booster.set_param({"device": "cpu", "nthread": 1})   # single rows: threads only add overhead
    return booster


# --------------------------------------------------------------------------- inference
class Predictor:
    """Loads every exported model once and prices census districts (default: the best on validation)."""

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.config = json.loads(self._require("config.json").read_text(encoding="utf-8"))
        if self.config.get("features", FEATURES) != FEATURES:
            raise ValueError("config.json feature order differs from model.FEATURES")
        self.best = self.config["best"]
        self.booster = load_booster(self._require(XGB_FILE))
        self._require(MLP_FILE)
        self.scaler = joblib.load(self._require(SCALER_FILE))
        self.mlp = HousingMLP.from_pretrained(self.model_dir, map_location="cpu", strict=True)
        self.mlp.to(self.device).eval()

    def _require(self, filename: str) -> Path:
        path = self.model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found: train and export the model first (training/notebook.ipynb) "
                                    f"or download it with huggingface_hub.snapshot_download('{REPO_ID}')")
        return path

    @property
    def choices(self) -> list[str]:
        """Valid values for `model`: "best" plus every exported model name."""
        return ["best", *self.config["models"]]

    def resolve(self, model: str | None = "best") -> str:
        name = self.best if model in (None, "", "best") else model
        if name not in self.config["models"]:
            raise ValueError(f"unknown model {model!r}; choose one of {self.choices}")
        return name

    @torch.inference_mode()
    def predict_value(self, columns: Mapping, model: str | None = "best") -> np.ndarray:
        """Batch path used for evaluation: raw columns -> predicted MedHouseVal ($100k), floored at 0."""
        name = self.resolve(model)
        x = to_matrix(columns)
        if name == "xgboost":
            pred = self.booster.inplace_predict(x)
        else:
            scaled = torch.as_tensor(self.scaler.transform(x), dtype=torch.float32, device=self.device)
            pred = self.mlp(scaled)[:, 0].float().cpu().numpy()
        return np.maximum(np.asarray(pred, dtype=np.float64), 0.0)

    def predict(self, features: Mapping, model: str | None = "best") -> dict:
        """One district {MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude}
        -> {"price_usd": predicted median house value in dollars, "model": model name}."""
        if not isinstance(features, Mapping):
            raise TypeError(f"features must be a dict with keys {RAW_FEATURES}")
        single = {name: features.get(name) for name in RAW_FEATURES if name in features}
        for name, value in single.items():
            if np.ndim(value) != 0:
                raise ValueError(f"{name} must be a single number (use predict_value for batches)")
        name = self.resolve(model)
        value = float(self.predict_value(single, name)[0])
        price = round(value * TARGET_UNIT_USD, 2)
        if not math.isfinite(price):
            raise ValueError("prediction is not finite; check the inputs")
        return {"price_usd": price, "model": name}


def load(model_dir: str | Path, device: str = "cpu") -> Predictor:
    return Predictor(model_dir, device)
