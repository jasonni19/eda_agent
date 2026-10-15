"""Regenerate the synthetic example: python examples/generate_bo_profile.py."""

from pathlib import Path

import pandas as pd

from eda_agent.bo import BOProblem
from eda_agent.bo_analyzer import analyze_bo_problem


def main():
    directory = Path(__file__).resolve().parent
    frame = pd.read_csv(directory / "bo_profile_input.csv")
    problem = BOProblem.model_validate_json((directory / "bo_problem.json").read_text(encoding="utf-8"))
    # The analyzer accepts absent optional EDA information. This fixture uses only
    # the compatible dataset metadata subset; production loads eda_report.json.
    eda_metadata = {"dataset": {"row_count": len(frame)}}
    profile = analyze_bo_problem(frame, eda_metadata, problem)
    (directory / "bo_profile.json").write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
