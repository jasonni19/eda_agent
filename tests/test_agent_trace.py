from langchain.messages import AIMessage, ToolMessage

from eda_agent.agent import _build_tool_trace, _trace_preview


def test_build_tool_trace_matches_calls_to_results():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "overview", "args": {}, "id": "call-1", "type": "tool_call"}],
        ),
        ToolMessage(
            content='{"row_count": 2, "column_count": 1}',
            name="overview",
            tool_call_id="call-1",
        ),
    ]

    trace = _build_tool_trace(messages)

    assert trace == [
        {
            "tool": "overview",
            "label": "Inspecting dataset",
            "operation": "df.info() and column cardinality",
            "arguments": {},
            "status": "completed",
            "result_preview": '{"row_count": 2, "column_count": 1}',
        }
    ]


def test_trace_preview_is_bounded():
    preview = _trace_preview("x" * 3000, limit=100)

    assert len(preview) < 140
    assert preview.endswith("result preview truncated")
