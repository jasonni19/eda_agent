"""California Housing Price Predictor — Hugging Face Space. Code + weights come from the model repo shalev396/california-housing."""
from space_utils import GPU, LAUNCH_KWARGS, card, header, import_model, is_quota_error, links, read_json  # FIRST

import csv
import html
import time
from pathlib import Path

import gradio as gr

SLUG = "california-housing"
M, MODEL_DIR = import_model(f"shalev396/{SLUG}")
METRICS = read_json(MODEL_DIR / "metrics.json", {})
CONFIG = read_json(MODEL_DIR / "config.json", {})

CPU = M.load(MODEL_DIR, "cpu")
CUDA = M.load(MODEL_DIR, "cuda") if M.cuda_available() else None  # module level (required by ZeroGPU)

NAMES = {"xgboost": "XGBoost", "mlp": "PyTorch MLP"}


def _district(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude) -> dict:
    values = dict(zip(M.RAW_FEATURES, (med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup,
                                       latitude, longitude)))
    missing = [k for k, v in values.items() if v is None]
    if missing:
        raise gr.Error(f"Fill in every feature (missing: {', '.join(missing)}).")
    return {k: float(v) for k, v in values.items()}


@GPU(duration=10)
def _predict_gpu(district, model):
    return CUDA.predict(district, model=model)


def predict(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude, model="best"):
    """Public API `/predict`: 8 census features (+ model) -> ({"price_usd", "model"}, seconds, device)."""
    district = _district(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude)
    model = model or "best"
    if model not in CPU.choices:
        raise gr.Error(f"Unknown model {model!r}; choose one of {CPU.choices}.")
    start = time.perf_counter()
    try:
        if CUDA is not None:
            try:
                result, device = _predict_gpu(district, model), "gpu"
            except gr.Error as err:          # visitor out of ZeroGPU quota → CPU fallback
                if not is_quota_error(err):
                    raise
                result, device = CPU.predict(district, model=model), "cpu"
        else:
            result, device = CPU.predict(district, model=model), "cpu"
    except ValueError as err:                # invalid inputs (e.g. AveRooms <= 0)
        raise gr.Error(str(err)) from None
    return result, round(time.perf_counter() - start, 4), device


def render_price(result) -> str:
    """Big price card for the UI (chained after /predict, private)."""
    if not result:
        return '<div class="price-card"><div class="price">—</div></div>'
    name = NAMES.get(result["model"], result["model"])
    return (f'<div class="price-card"><div class="price">${result["price_usd"]:,.0f}</div>'
            f'<div class="price-sub">predicted median house value · {html.escape(name)}</div></div>')


def _examples() -> tuple[list[list], list[str]]:
    """Real test-split districts from examples/districts.csv -> (input rows, labels)."""
    with open(Path(__file__).resolve().parent / "examples" / "districts.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    examples = [[float(r[k]) for k in M.RAW_FEATURES] + ["best"] for r in rows]
    labels = [f"{r['city']} (actual ${float(r[M.TARGET]) * M.TARGET_UNIT_USD:,.0f})" for r in rows]
    return examples, labels


# --------------------------------------------------------------------------- info cards
SCORES = METRICS.get("metrics", {})
DATA = METRICS.get("data", {})
COMPARISON = METRICS.get("comparison", {})
BEST = CONFIG.get("best", "xgboost")


def _score(name: str) -> str:
    m = COMPARISON.get(name)
    return f"R² {m['r2']:.3f} · RMSE ${m['rmse_usd']:,.0f}" if m else "n/a"


MODEL_ROWS = [
    ("Default model", f"{NAMES.get(BEST, BEST)} (lowest validation RMSE)"),
    ("XGBoost", f"hist trees, depth 6, {CONFIG.get('models', {}).get('xgboost', {}).get('best_iteration', 0) + 1} "
                "rounds (early stopping)"),
    ("PyTorch MLP", f"{len(M.FEATURES)}-{'-'.join(map(str, CONFIG.get('hidden_units', [])))}-1, ReLU, dropout "
                    f"{CONFIG.get('dropout', '?')} · {METRICS.get('model_size', {}).get('mlp_params', 0):,} params"),
    ("Features", "8 census features + 3 ratios (rooms/household, bedrooms/room, people/household)"),
    ("Test XGBoost", _score("xgboost")),
    ("Test MLP", _score("mlp")),
]
TRAIN_ROWS = [
    ("Dataset", "California Housing, 1990 US census (sklearn), 20,640 districts"),
    ("Split", f"{DATA.get('n_train', '?'):,} train / {DATA.get('n_val', '?'):,} val / {DATA.get('n_test', '?'):,} test"
              if DATA else "n/a"),
    ("Baselines (test)", f"LinearRegression {_score('linear_regression')} · RandomForest {_score('random_forest')}"),
    ("Target", "median house value, capped at $500k in the source data"),
    ("Weights", METRICS.get("source", "n/a")),
    ("Trained", f"{METRICS.get('trained_at', '?')} on {METRICS.get('device', '?')}"),
]

PRICE_CSS = """
.price-card { text-align: center; padding: 18px 8px; }
.price-card .price { font-size: 2.4em; font-weight: 700; letter-spacing: .01em; }
.price-card .price-sub { opacity: .8; margin-top: 4px; }
"""
EXAMPLES, EXAMPLE_LABELS = _examples()

with gr.Blocks(title="California Housing Price Predictor") as demo:
    gr.HTML(header("California Housing Price Predictor", "🏠",
                   "Describe a California census district (1990) and get its predicted <b>median house value</b> "
                   "from an XGBoost model or a PyTorch MLP."))
    with gr.Row(equal_height=True):
        with gr.Column():
            med_inc = gr.Slider(0.5, 15.0, value=3.87, step=0.01, label="MedInc: median income (in $10k)")
            house_age = gr.Slider(1, 52, value=29, step=1, label="HouseAge: median house age (years)")
            ave_rooms = gr.Slider(1.0, 20.0, value=5.43, step=0.01, label="AveRooms: rooms per household")
            ave_bedrms = gr.Slider(0.3, 5.0, value=1.10, step=0.01, label="AveBedrms: bedrooms per household")
            population = gr.Number(value=1425, minimum=1, precision=0, label="Population of the district")
            ave_occup = gr.Slider(0.7, 10.0, value=3.07, step=0.01, label="AveOccup: people per household")
            latitude = gr.Slider(32.5, 42.0, value=35.63, step=0.01, label="Latitude")
            longitude = gr.Slider(-124.4, -114.3, value=-119.57, step=0.01, label="Longitude")
            model = gr.Dropdown(CPU.choices, value="best", label="Model",
                                info=f"best = {NAMES.get(BEST, BEST)} (lowest validation RMSE)")
            btn = gr.Button("Predict price", variant="primary")
        with gr.Column():
            price = gr.HTML(render_price(None))
            out = gr.JSON(label="Result")
            sec = gr.Number(label="Inference time (s)", precision=4)
            dev = gr.Textbox(label="Device", max_lines=1)
    inputs = [med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude, model]
    gr.Examples(examples=EXAMPLES, example_labels=EXAMPLE_LABELS, inputs=inputs, cache_examples=False,
                label="Real test-split districts")
    btn.click(predict, inputs=inputs, outputs=[out, sec, dev], api_name="predict").then(
        render_price, inputs=[out], outputs=[price], api_visibility="private")
    with gr.Row(equal_height=True):
        with gr.Column():
            with gr.Group():
                gr.HTML(card("Model", MODEL_ROWS))
        with gr.Column():
            with gr.Group():
                gr.HTML(card("Training", TRAIN_ROWS))
    gr.HTML(links(SLUG))

if __name__ == "__main__":
    demo.launch(**{**LAUNCH_KWARGS, "css": LAUNCH_KWARGS["css"] + PRICE_CSS})
