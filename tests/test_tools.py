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

