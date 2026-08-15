"""Deterministically render validated EDA plot plans as downloadable PNG images."""

from __future__ import annotations

import base64
import re
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .schemas import PlotPlan


MAX_PLOT_COLUMNS = 30
COLOR = "#f2d600"
EDGE_COLOR = "#111111"


def _safe_filename(index: int, title: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "eda_plot"
    return f"{index + 1:02d}_{stem[:70]}.png"


def _style(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)


def _render_one(df: pd.DataFrame, plan: PlotPlan) -> bytes | None:
    columns = [column for column in plan.columns if column in df.columns]
    if not columns:
        return None

    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    try:
        if plan.plot_type == "missingness_bar":
            columns = columns[:MAX_PLOT_COLUMNS]
            values = df[columns].isna().mean().mul(100)
            axis.bar(values.index.astype(str), values.values, color=COLOR, edgecolor=EDGE_COLOR)
            axis.set_ylabel("Missing values (%)")
            axis.tick_params(axis="x", rotation=55)
        elif plan.plot_type == "histogram":
            series = pd.to_numeric(df[columns[0]], errors="coerce").dropna()
            if series.empty:
                return None
            axis.hist(series, bins=min(30, max(5, int(np.sqrt(len(series))))), color=COLOR, edgecolor=EDGE_COLOR)
            axis.set_xlabel(columns[0])
            axis.set_ylabel("Count")
        elif plan.plot_type == "box_plot":
            series = pd.to_numeric(df[columns[0]], errors="coerce").dropna()
            if series.empty:
                return None
            axis.boxplot(series, vert=False, patch_artist=True, boxprops={"facecolor": COLOR}, medianprops={"color": EDGE_COLOR, "linewidth": 2})
            axis.set_xlabel(columns[0])
            axis.set_yticks([])
        elif plan.plot_type == "count_plot":
            counts = df[columns[0]].fillna("<MISSING>").astype(str).value_counts().head(10).sort_values()
            if counts.empty:
                return None
            axis.barh(counts.index, counts.values, color=COLOR, edgecolor=EDGE_COLOR)
            axis.set_xlabel("Count")
        elif plan.plot_type == "correlation_heatmap":
            numeric = df[columns].select_dtypes(include="number").iloc[:, :20]
            if numeric.shape[1] < 2:
                return None
            matrix = numeric.corr()
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=55, ha="right")
            axis.set_yticks(range(len(matrix.index)), matrix.index)
            figure.colorbar(image, ax=axis, label="Pearson correlation")
        elif plan.plot_type == "target_relationship" and len(columns) >= 2:
            left, right = columns[:2]
            if pd.api.types.is_numeric_dtype(df[left]) and pd.api.types.is_numeric_dtype(df[right]):
                axis.scatter(df[left], df[right], color=COLOR, edgecolor=EDGE_COLOR, alpha=0.7)
                axis.set_xlabel(left)
                axis.set_ylabel(right)
            else:
                return None
        else:
            return None

        _style(axis, plan.title)
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=110, facecolor="white")
        return buffer.getvalue()
    finally:
        plt.close(figure)


def render_plot_images(df: pd.DataFrame, plans: list[PlotPlan]) -> list[dict[str, str]]:
    """Return API-only PNG artifacts without modifying the validated report."""
    artifacts: list[dict[str, str]] = []
    for index, plan in enumerate(plans):
        image = _render_one(df, plan)
        if image is None:
            continue
        artifacts.append({
            "filename": _safe_filename(index, plan.title),
            "title": plan.title,
            "rationale": plan.rationale,
            "image_data_uri": f"data:image/png;base64,{base64.b64encode(image).decode('ascii')}",
        })
    return artifacts
