"""Necessary for the CLI to be used. Command-line interface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import load_csv, run_eda
from .outputs import markdown_filename, render_markdown_report, save_plot_images
from .plots import render_plot_images


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
    markdown_path = args.output.parent / markdown_filename(report.source_file)
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    plot_images = render_plot_images(load_csv(args.csv.resolve()), report.plot_plan)
    plots_directory = args.output.parent / "eda_plots"
    save_plot_images(plot_images, plots_directory)
    print(f"Saved {len(report.findings)} findings to {args.output}")
    print(f"Saved Markdown report to {markdown_path}")
    print(f"Saved {len(plot_images)} plots to {plots_directory}")


if __name__ == "__main__":
    main()
