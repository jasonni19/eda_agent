import base64

import pandas as pd

from eda_agent.plots import render_plot_images
from eda_agent.schemas import PlotPlan


def test_render_plot_images_adds_downloadable_png_data():
    plans = [
        PlotPlan(
            plot_type="histogram",
            columns=["amount"],
            title="Distribution of amount",
            rationale="Inspect the distribution.",
        )
    ]

    artifacts = render_plot_images(pd.DataFrame({"amount": [1, 2, 2, 3, 100]}), plans)

    assert artifacts[0]["filename"] == "01_distribution_of_amount.png"
    assert "filename" not in plans[0].model_dump()
    assert "image_data_uri" not in plans[0].model_dump()
    encoded = artifacts[0]["image_data_uri"].removeprefix("data:image/png;base64,")
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")


def test_render_plot_images_ignores_unknown_columns_safely():
    plan = PlotPlan(
        plot_type="histogram",
        columns=["missing_column"],
        title="Unavailable plot",
        rationale="Test invalid input handling.",
    )

    artifacts = render_plot_images(pd.DataFrame({"amount": [1, 2]}), [plan])

    assert artifacts == []
