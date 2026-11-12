---
license: mit
library_name: xgboost
pipeline_tag: tabular-regression
tags:
- ml-lab
- xgboost
- pytorch
- scikit-learn
- tabular-regression
- house-prices
datasets: []
metrics:
- r2
- rmse
- mae
ml_lab:
  title: California Housing Price Predictor
  order: 14
  summary: Predicts the median house value of a California district from 8 census features.
  framework: XGBoost + PyTorch
  architecture: XGBoost (hist, early stopping) + PyTorch MLP 11-256-128-64-1
  dataset: California Housing (1990 census, sklearn)
  space: shalev396/california-housing
  runtime: cpu-basic
  ui_kind: custom
  colab: https://colab.research.google.com/github/shalev396/ml-lab/blob/main/california-housing/training/notebook.ipynb
  github: https://github.com/shalev396/ml-lab/tree/main/california-housing
  metric:
    name: r2
    value: 0.852486475534139
    split: test
  trained_at: '2026-09-25'
model-index:
- name: california-housing
  results:
  - task:
      type: tabular-regression
    dataset:
      name: California Housing (1990 census, sklearn)
      type: california-housing-1990-census-sklearn
      split: test
    metrics:
    - type: rmse
      value: 0.439662
      name: rmse
    - type: mae
      value: 0.285404
      name: mae
    - type: r2
      value: 0.852486
      name: r2
    - type: rmse_usd
      value: 43966.0
      name: rmse_usd
    - type: mae_usd
      value: 28540.0
      name: mae_usd
---
# California Housing Price Predictor

Predicts the **median house value** of a California census district (block group, 1990 census) from 8 features:
median income, house age, rooms, bedrooms, population, occupancy, latitude and longitude. The default model is
**XGBoost**, and a **PyTorch MLP** is included as an alternative. Both were trained on sklearn's
[California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) dataset.
On the held-out test split (4,128 districts), XGBoost reaches **R² 0.853** with an RMSE of **$43,966** and an MAE
of $28,540 (retrained 2026-09-25, `metrics.json`).

## Model
- **Input**: one district as a dict with the 8 raw features `MedInc` (median income in $10k), `HouseAge`,
  `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude` (same units as sklearn).
- **Features** (`model.engineer`): the 8 raw features plus 3 ratios from the classic recipe:
  `rooms_per_household`, `bedrooms_per_room = AveBedrms / AveRooms` and `population_per_household`. sklearn's
  version is already per household, so the first and last ratio equal `AveRooms` and `AveOccup`. They are kept
  to mirror the recipe; `bedrooms_per_room` is the only new signal. This gives 11 columns, in the order stored in
  `config.json["features"]`.
- **`xgboost`** (default, `xgb.ubj`): `XGBRegressor` with hist trees, depth 6, learning rate 0.05, subsample and
  column subsample 0.8. Early stopping on validation RMSE picked round 1,085 of at most 2,000, and the booster is
  saved trimmed to those 1,085 trees in XGBoost's native format. It uses the unscaled features.
- **`mlp`** (alternative, `model.safetensors` + `config.json`): `HousingMLP`, a `PyTorchModelHubMixin` module
  11-256-128-64-1 with ReLU and dropout 0.15 (44,289 parameters). Its inputs are standardized by `scaler.joblib`,
  a `StandardScaler` fit on the train split.
- **Output**: `predict(features, model="best")` returns
  `{"price_usd": 368791.17, "model": "xgboost"}`. The model predicts `MedHouseVal` in $100k; the value is floored
  at 0 and multiplied by 100,000. `model` can be `"best"`, `"xgboost"` or `"mlp"`.
- **`config.json`** holds the MLP's init arguments (read back by `HousingMLP.from_pretrained`), the feature order
  and formulas, the default model, each model's validation RMSE and the library versions.

## Usage
```python
from huggingface_hub import snapshot_download
import sys
path = snapshot_download("shalev396/california-housing")
sys.path.insert(0, path)
import model
predictor = model.load(path, device="cpu")   # the MLP can also run on "cuda"; XGBoost always runs on the CPU
print(predictor.predict({"MedInc": 4.445, "HouseAge": 52, "AveRooms": 5.5346, "AveBedrms": 1.1509,
                         "Population": 742, "AveOccup": 2.3333, "Latitude": 37.77, "Longitude": -122.43}))
# {'price_usd': 368791.17, 'model': 'xgboost'}   (actual value of this San Francisco district: $400,000)
```
- **Space API**: the [Space](https://huggingface.co/spaces/shalev396/california-housing) serves the same code at
  `POST /gradio_api/call/predict` (8 numbers + model name -> `[result, seconds, device]`).
- **Inference Endpoint**: `handler.py` makes this repo deployable as a
  [custom Inference Endpoint](https://huggingface.co/docs/inference-endpoints/guides/custom_handler). Request
  `{"inputs": {<8 features>}, "parameters": {"model": "best"}}`; a list of districts returns a list of results.

## Training
- **Data**: 20,640 California block groups from the 1990 US census (StatLib), loaded with
  `sklearn.datasets.fetch_california_housing`. The target `MedHouseVal` is capped at 5.00001 ($500,001) in the
  source data.
- **Split**: 80/20 train/test with seed 42, then 15 % of the train part as validation: 14,035 / 2,477 / 4,128
  districts.
- **Experiments**: LinearRegression, RandomForest (200 trees), XGBoost (early stopping, 50 rounds patience) and
  the MLP (AdamW lr 1e-3, weight decay 1e-4, batch 256, MSE loss, early stopping with patience 10: best epoch 81
  of 91). All are fit on the train split.
- **Selection**: the default model is the exported model with the lowest **validation** RMSE. The test split is
  scored once, after that choice.
- **Hardware / time**: CPU only. The run was measured on a shared 20-thread desktop CPU that several other
  training jobs were using at the same time (6 threads for this run), so its fit times are much slower than an
  idle machine would give: XGBoost 1,205 s, MLP 956 s, RandomForest 8 s.
- Full code: [training/](https://github.com/shalev396/ml-lab/tree/main/california-housing/training) ·
  [Colab](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/california-housing/training/notebook.ipynb)

## Evaluation
Default model (XGBoost) on the test split. RMSE and MAE are in $100k (the target unit); the `_usd` rows are in
dollars:

<!-- metrics:start -->
| metric (test) | value |
|---|---|
| rmse | 0.4397 |
| mae | 0.2854 |
| r2 | 0.8525 |
| rmse_usd | 43966.0000 |
| mae_usd | 28540.0000 |
<!-- metrics:end -->

## Experiments
Every experiment, ranked by validation RMSE. The default model is in bold:

<!-- comparison:start -->
| model | val RMSE | test RMSE | test MAE | test R² | test RMSE ($) | exported |
|---|---|---|---|---|---|---|
| **XGBoost** (default) | **0.4671** | **0.4397** | **0.2854** | **0.8525** | **$43,966** | **yes** |
| PyTorch MLP | 0.5269 | 0.5186 | 0.3487 | 0.7947 | $51,865 | yes |
| RandomForest | 0.5299 | 0.5096 | 0.3327 | 0.8018 | $50,964 | no |
| LinearRegression | 0.7241 | 0.7277 | 0.5252 | 0.5958 | $72,774 | no |
<!-- comparison:end -->

XGBoost wins: it beats the MLP and the RandomForest by $7-8k of test RMSE. The earlier version of this
project reported almost the same XGBoost result (test RMSE 0.4423, R² 0.851) with the same split and XGBoost settings.

![Model comparison](assets/model_comparison.png)
![Training curves](assets/training_curves.png)
![Predicted vs actual](assets/pred_vs_actual.png)
![Feature importance](assets/feature_importance.png)

Median income carries 42 % of XGBoost's total gain; location (longitude + latitude) carries another 25 %.

## Limitations
- **1990 data**: prices are 1990 district medians in 1990 dollars. They say nothing about today's market or about
  a single house.
- **Capped target**: the source data caps values at $500,001, so the model never learned what lies above it.
  Capped districts are often underestimated (the vertical column at $500k in the scatter plot), and predictions
  rarely go past about $540k.
- **District-level features**: the inputs are averages over a block group (hundreds to thousands of people), not
  properties of one home.
- **Extrapolation**: inputs far outside the training ranges (for example coordinates outside California) still
  return a number, but it is meaningless.
- **Duplicate ratios**: two of the three engineered features duplicate raw columns (see Model). This is harmless for
  the trees and the scaled MLP, but they add no information.
