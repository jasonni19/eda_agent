"""These are the deterministic pandas tools exposed to the language model."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import pandas as pd
from langchain.tools import tool


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values and non-finite floats to JSON-safe values."""
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_safe(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def build_eda_tools(df: pd.DataFrame) -> list[Any]:
    """Bind one DataFrame to read-only analysis tools. There are currently 7 tools available. 
    They are the following:
    - overview: Inspect dataset dimensions, column types, non-null counts, and cardinality.
    - missing_values: Count and calculate the percentage of missing values in every column.
    - statistics: Calculate descriptive statistics for numeric and non-numeric columns.
    - correlations: Find the strongest Pearson correlations between pairs of numeric columns.
    - outliers: Detect numeric outliers with the 1.5 times IQR rule and report boundaries/counts.
    - duplicates: Count fully duplicated rows, excluding the first occurrence.
    - value_counts: Get the ten most common values for categorical-like columns."""

    @tool("overview")
    def overview() -> dict[str, Any]:
        """Inspect dataset dimensions, column types, non-null counts, and cardinality."""
        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": [
                {
                    "name": str(column),
                    "dtype": str(df[column].dtype),
                    "non_null_count": int(df[column].notna().sum()),
                    "unique_count": int(df[column].nunique(dropna=True)),
                }
                for column in df.columns
            ],
        }

    @tool("missing_values")
    def missing_values() -> dict[str, Any]:
        """Count and calculate the percentage of missing values in every column."""
        counts = df.isna().sum()
        denominator = len(df)
        return {
            str(column): {
                "count": int(count),
                "percent": round(float(count / denominator * 100), 2) if denominator else 0.0,
            }
            for column, count in counts.items()
        }

    @tool("statistics")
    def statistics() -> dict[str, Any]:
        """Calculate descriptive statistics for numeric and non-numeric columns."""
        if df.empty and not len(df.columns):
            return {"columns": []}
        summary = df.describe(include="all").transpose().reset_index()
        summary = summary.rename(columns={summary.columns[0]: "column"})
        return {"columns": _records(summary)}

    @tool("correlations")
    def correlations() -> dict[str, Any]:
        """Find the strongest Pearson correlations between pairs of numeric columns."""
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return {"method": "pearson", "pairs": [], "note": "Fewer than two numeric columns."}
        matrix = numeric.corr(method="pearson")
        pairs: list[dict[str, Any]] = []
        for left_index, left in enumerate(matrix.columns):
            for right in matrix.columns[left_index + 1 :]:
                value = matrix.loc[left, right]
                if pd.notna(value):
                    pairs.append({"left": str(left), "right": str(right), "correlation": round(float(value), 4)})
        pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
        return {"method": "pearson", "pairs": pairs[:25]}

    @tool("outliers")
    def outliers() -> dict[str, Any]:
        """Detect numeric outliers with the 1.5 times IQR rule and report boundaries/counts."""
        results: dict[str, Any] = {}
        for column in df.select_dtypes(include="number").columns:
            series = df[column].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (series < lower) | (series > upper)
            results[str(column)] = {
                "lower_bound": _json_safe(float(lower)),
                "upper_bound": _json_safe(float(upper)),
                "count": int(mask.sum()),
                "percent_of_non_null": round(float(mask.mean() * 100), 2),
            }
        return results

    @tool("duplicates")
    def duplicates() -> dict[str, Any]:
        """Count fully duplicated rows, excluding the first occurrence."""
        count = int(df.duplicated().sum())
        return {
            "duplicate_row_count": count,
            "percent": round(float(count / len(df) * 100), 2) if len(df) else 0.0,
        }

    @tool("value_counts")
    def value_counts() -> dict[str, Any]:
        """Get the ten most common values for categorical-like columns."""
        result: dict[str, Any] = {}
        candidates = df.select_dtypes(include=["object", "category", "bool", "string"]).columns
        for column in candidates:
            counts = df[column].value_counts(dropna=False).head(10)
            result[str(column)] = [
                {"value": "<MISSING>" if pd.isna(value) else str(value), "count": int(count)}
                for value, count in counts.items()
            ]
        return result

    # return the list of tools
    return [overview, missing_values, statistics, correlations, outliers, duplicates, value_counts]
