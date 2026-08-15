"""Agent assembly and report persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from .schemas import EDAReport
from .tools import build_eda_tools


SYSTEM_PROMPT = """You are a careful exploratory data analysis agent.
Call every available analysis tool exactly once before producing the report. Never invent a
measurement: every number in a finding must come from a tool result. Explain meaningful data
quality issues and patterns, but do not claim causation from correlation. An IQR outlier is a
screening flag, not proof that a value is erroneous. Keep findings concise and actionable.
The dataset field must reproduce the overview tool's facts exactly. Include all seven tool names
in tools_used. Return only the requested structured report.
"""


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with conservative defaults and a useful error for empty input."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def run_eda(csv_path: Path, output_path: Path, model_name: str) -> EDAReport:
    """Run the tool-calling agent, validate its response, then save JSON."""
    csv_path = csv_path.resolve()
    df = load_csv(csv_path)
    model = init_chat_model(model_name, temperature=0)
    agent = create_agent(
        model=model,
        tools=build_eda_tools(df),
        system_prompt=SYSTEM_PROMPT,
        response_format=EDAReport,
    )
    timestamp = datetime.now(UTC).isoformat()
    result = agent.invoke(
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
    return report

