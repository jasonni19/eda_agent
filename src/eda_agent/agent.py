"""Agent assembly and report persistence."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, ToolMessage

from .schemas import EDAReport
from .tools import build_eda_tools


SYSTEM_PROMPT = """You are a careful exploratory data analysis agent.
Call every available analysis tool exactly once before producing the report. Never invent a
measurement: every number in a finding must come from a tool result. Explain meaningful data
quality issues and patterns, but do not claim causation from correlation. An IQR outlier is a
screening flag, not proof that a value is erroneous. Keep findings concise and actionable.
The dataset field must reproduce the overview tool's facts exactly. Include every tool name in
tools_used. Recommend and rank plausible target columns from feature_analysis evidence, but never
claim the true target is known without a business prediction objective. Confidence must reflect
that limitation. Use preprocessing_advice and redundancy_analysis for transformations and feature
exclusions. Copy the deterministic plot_plan tool specifications into plot_plan; do not invent
plots. Give downstream ML advice covering problem framing, preprocessing, validation, metrics,
and modeling when supported. Return only the requested structured report.
"""

TOOL_TRACE_METADATA = {
    "overview": ("Inspecting dataset", "df.info() and column cardinality"),
    "missing_values": ("Checking missing values", "df.isna().sum()"),
    "statistics": ("Calculating descriptive statistics", "df.describe(include='all')"),
    "correlations": ("Analyzing correlations", "numeric_df.corr(method='pearson')"),
    "outliers": ("Detecting outliers", "1.5 × IQR screening"),
    "duplicates": ("Checking duplicate rows", "df.duplicated().sum()"),
    "value_counts": ("Inspecting categorical values", "series.value_counts().head(10)"),
    "feature_analysis": ("Classifying features and target candidates", "dtype and cardinality heuristics"),
    "distributions": ("Analyzing distributions", "skewness, zeros, ranges, and concentration"),
    "redundancy_analysis": ("Checking feature redundancy", "constants, identifiers, and |correlation| ≥ 0.90"),
    "preprocessing_advice": ("Planning preprocessing", "measured transformation and exclusion rules"),
    "plot_plan": ("Selecting EDA plots", "plot rules based on inferred column types"),
}


def _trace_preview(content: Any, limit: int = 2500) -> str:
    """Create a bounded, display-safe preview of a tool result."""
    if isinstance(content, str):
        preview = content
    else:
        preview = json.dumps(content, indent=2, default=str)
    if len(preview) > limit:
        return preview[:limit].rstrip() + "\n… result preview truncated"
    return preview


def _build_tool_trace(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract auditable tool activity without exposing private model reasoning."""
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls[str(call.get("id", ""))] = call

    trace: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        call = calls.get(str(message.tool_call_id), {})
        name = str(message.name or call.get("name") or "tool")
        label, operation = TOOL_TRACE_METADATA.get(name, (name.replace("_", " ").title(), name))
        trace.append({
            "tool": name,
            "label": label,
            "operation": operation,
            "arguments": call.get("args", {}),
            "status": "error" if getattr(message, "status", None) == "error" else "completed",
            "result_preview": _trace_preview(message.content),
        })
    return trace


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with conservative defaults and a useful error for empty input."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def run_eda_detailed(
    csv_path: Path, output_path: Path, model_name: str
) -> tuple[EDAReport, list[dict[str, Any]]]:
    """Run the agent and return both its validated report and auditable tool trace."""
    csv_path = csv_path.resolve()
    df = load_csv(csv_path)
    model = init_chat_model(model_name, temperature=0)
    agent = create_agent(
        model=model,
        tools=build_eda_tools(df), # this is the list of tools that the agent will use to analyze the data
        system_prompt=SYSTEM_PROMPT,
        response_format=EDAReport, # this is the format of the response that the agent will return
    )
    timestamp = datetime.now(UTC).isoformat() # this is the timestamp of the analysis
    result = agent.invoke( # this is the invocation of the agent. 
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Analyze {csv_path.name}. Use source_file={csv_path.name!r} and "
                        f"generated_at={timestamp!r} in the final report."
                    ),
                }
            ]
        }
    )
    
    
    report = EDAReport.model_validate(result["structured_response"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report, _build_tool_trace(result.get("messages", []))


def run_eda(csv_path: Path, output_path: Path, model_name: str) -> EDAReport:
    """Run the tool-calling agent, validate its response, then save JSON."""
    report, _ = run_eda_detailed(csv_path, output_path, model_name)
    return report
