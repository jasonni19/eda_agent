"""Create human-readable and bundled downloadable EDA outputs."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .schemas import EDAReport


def markdown_filename(source_file: str) -> str:
    """Return a safe report filename based on the uploaded CSV name."""
    stem = Path(source_file).stem or "dataset"
    safe_stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)
    return f"{safe_stem}_eda_report.md"


def render_markdown_report(report: EDAReport) -> str:
    """Convert the validated report into concise, human-readable Markdown."""
    lines = [
        f"# EDA Report: {report.source_file}",
        "",
        f"Generated: {report.generated_at.isoformat()}",
        "",
        "## Executive summary",
        "",
        report.executive_summary,
        "",
        "## Dataset overview",
        "",
        f"- Rows: {report.dataset.row_count:,}",
        f"- Columns: {report.dataset.column_count:,}",
        "",
        "| Column | Type | Non-null | Unique |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {column.name} | {column.dtype} | {column.non_null_count:,} | {column.unique_count:,} |"
        for column in report.dataset.columns
    )

    lines.extend(["", "## Recommended target columns", ""])
    if report.target_recommendations:
        for target in report.target_recommendations:
            lines.extend([
                f"### {target.column}",
                "",
                f"- Confidence: {target.confidence}",
                f"- Likely task: {target.likely_task}",
                f"- Rationale: {target.rationale}",
            ])
            lines.extend(f"- Caution: {caution}" for caution in target.cautions)
            lines.append("")
    else:
        lines.extend(["No reliable target candidates were identified.", ""])

    lines.extend(["## Key findings", ""])
    for finding in report.findings:
        lines.extend([
            f"### {finding.summary}",
            "",
            f"- Category: {finding.category}",
            f"- Severity: {finding.severity}",
            f"- Evidence: {finding.evidence}",
            f"- Recommendation: {finding.recommendation}",
            "",
        ])

    lines.extend(["## Suggested transformations", ""])
    lines.extend(
        f"- **{item.column}:** {item.transformation} — {item.rationale}"
        for item in report.transformations
    )
    if not report.transformations:
        lines.append("- No transformations were recommended.")

    lines.extend(["", "## Feature exclusions", ""])
    lines.extend(
        f"- **{item.column}** ({item.action}): {item.reason}"
        for item in report.feature_exclusions
    )
    if not report.feature_exclusions:
        lines.append("- No feature exclusions were recommended.")

    lines.extend(["", "## Downstream ML recommendations", ""])
    lines.extend(
        f"- **{item.area}:** {item.recommendation} — {item.rationale}"
        for item in report.ml_recommendations
    )

    lines.extend(["", "## Generated plots", ""])
    lines.extend(f"- **{plot.title}:** {plot.rationale}" for plot in report.plot_plan)

    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report.caveats)
    return "\n".join(lines).rstrip() + "\n"


def plots_zip_data_uri(plot_images: list[dict[str, str]]) -> str:
    """Bundle rendered PNG artifacts into an API-only downloadable ZIP data URI."""
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for plot in plot_images:
            encoded = plot["image_data_uri"].partition(",")[2]
            archive.writestr(f"eda_plots/{plot['filename']}", base64.b64decode(encoded))
    return f"data:application/zip;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def save_plot_images(plot_images: list[dict[str, str]], directory: Path) -> None:
    """Save rendered PNG artifacts into a real directory for CLI usage."""
    directory.mkdir(parents=True, exist_ok=True)
    for plot in plot_images:
        encoded = plot["image_data_uri"].partition(",")[2]
        (directory / plot["filename"]).write_bytes(base64.b64decode(encoded))
