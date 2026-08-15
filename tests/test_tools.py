import pandas as pd

from eda_agent.tools import build_eda_tools

#basic unit tests for the tools


def call(tools, name):
    return next(tool for tool in tools if tool.name == name).invoke({})


def test_tools_compute_expected_evidence():
    frame = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 100.0, 1.0],
            "group": ["a", "a", None, "b", "a"],
        }
    )
    tools = build_eda_tools(frame)

    assert call(tools, "overview")["row_count"] == 5
    assert call(tools, "missing_values")["group"] == {"count": 1, "percent": 20.0}
    assert call(tools, "outliers")["amount"]["count"] == 1
    assert call(tools, "duplicates")["duplicate_row_count"] == 1
    assert call(tools, "value_counts")["group"][0] == {"value": "a", "count": 3}


def test_correlations_gracefully_handles_one_numeric_column():
    tools = build_eda_tools(pd.DataFrame({"number": [1, 2], "label": ["x", "y"]}))
    assert call(tools, "correlations")["pairs"] == []


def test_feature_analysis_recommends_named_target_with_task_type():
    tools = build_eda_tools(
        pd.DataFrame({"customer_id": ["a", "b", "c", "d"], "age": [20, 30, 40, 50], "churn": [0, 1, 0, 1]})
    )
    result = call(tools, "feature_analysis")

    assert result["target_candidates"][0]["column"] == "churn"
    assert result["target_candidates"][0]["likely_task"] == "classification"
    assert next(item for item in result["features"] if item["column"] == "customer_id")["inferred_type"] == "identifier"


def test_distribution_redundancy_and_preprocessing_evidence():
    frame = pd.DataFrame(
        {
            "constant": [1, 1, 1, 1],
            "x": [1.0, 2.0, 3.0, 100.0],
            "x_copy": [1.0, 2.0, 3.0, 100.0],
            "group": ["a", "a", "b", None],
        }
    )
    tools = build_eda_tools(frame)

    assert call(tools, "distributions")["x"]["skewness"] > 1
    redundancy = call(tools, "redundancy_analysis")
    assert "constant" in redundancy["constant_columns"]
    assert redundancy["high_correlations"][0]["correlation"] == 1.0
    advice = call(tools, "preprocessing_advice")
    assert any(item["column"] == "group" and "missing" in item["evidence"] for item in advice["transformations"])
    assert any(item["column"] == "constant" for item in advice["exclusion_candidates"])


def test_plot_plan_is_bounded_and_uses_existing_columns():
    frame = pd.DataFrame({"amount": [1, 2], "group": ["a", "b"]})
    plots = call(build_eda_tools(frame), "plot_plan")["plots"]

    assert len(plots) <= 12
    assert {column for plot in plots for column in plot["columns"]} <= {"amount", "group"}
