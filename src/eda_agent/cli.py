"""Command-line interface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import run_eda


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a schema-validated EDA report from a CSV.")
    parser.add_argument("csv", type=Path, help="Input CSV path")
    parser.add_argument("--output", type=Path, default=Path("eda_report.json"))
    parser.add_argument("--model", default=None, help="LangChain model id; defaults to EDA_MODEL")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    model_name = args.model or os.getenv("EDA_MODEL", "openai:gpt-4.1-mini")
    report = run_eda(args.csv, args.output, model_name)
    print(f"Saved {len(report.findings)} findings to {args.output}")


if __name__ == "__main__":
    main()

