import base64
from datetime import UTC, datetime
from io import BytesIO
from zipfile import ZipFile

from eda_agent.outputs import markdown_filename, plots_zip_data_uri, render_markdown_report
from eda_agent.schemas import ColumnSummary, DatasetSummary, EDAReport


def sample_report() -> EDAReport:
    return EDAReport(
        source_file="sales data.csv",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        executive_summary="Sales data is ready for further review.",
        dataset=DatasetSummary(
            row_count=2,
            column_count=1,
            columns=[ColumnSummary(name="sales", dtype="int64", non_null_count=2, unique_count=2)],
        ),
        target_recommendations=[],
        transformations=[],
        feature_exclusions=[],
        plot_plan=[],
        ml_recommendations=[],
        findings=[],
        caveats=["Confirm the business target."],
        tools_used=[],
    )


def test_markdown_report_uses_input_filename_and_human_readable_sections():
    report = sample_report()
    markdown = render_markdown_report(report)

    assert markdown_filename(report.source_file) == "sales_data_eda_report.md"
    assert "# EDA Report: sales data.csv" in markdown
    assert "## Executive summary" in markdown
    assert "Sales data is ready for further review." in markdown


def test_plots_zip_contains_downloadable_eda_plots_folder():
    image = b"\x89PNG\r\n\x1a\nexample"
    data_uri = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    archive_uri = plots_zip_data_uri([{"filename": "chart.png", "image_data_uri": data_uri}])
    archive_bytes = base64.b64decode(archive_uri.partition(",")[2])

    with ZipFile(BytesIO(archive_bytes)) as archive:
        assert archive.namelist() == ["eda_plots/chart.png"]
        assert archive.read("eda_plots/chart.png") == image
