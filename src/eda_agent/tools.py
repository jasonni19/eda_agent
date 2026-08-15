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

    @tool("feature_analysis")
    def feature_analysis() -> dict[str, Any]:
        """Classify feature types and rank plausible target columns with transparent heuristics."""
        rows = len(df)
        target_words = {"target", "label", "outcome", "response", "class", "churn", "default", "fraud", "price", "score"}
        id_words = {"id", "uuid", "guid", "identifier", "index"}
        features: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        for position, column in enumerate(df.columns):
            series = df[column]
            name = str(column)
            words = set(name.lower().replace("-", "_").replace(" ", "_").split("_"))
            unique = int(series.nunique(dropna=True))
            non_null = int(series.notna().sum())
            unique_ratio = unique / max(non_null, 1)
            if pd.api.types.is_bool_dtype(series):
                kind = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(series):
                kind = "datetime"
            elif pd.api.types.is_numeric_dtype(series):
                kind = "numeric"
            elif unique_ratio >= 0.98 and unique > 1:
                kind = "identifier"
            elif unique <= max(50, int(rows * 0.05)):
                kind = "categorical"
            else:
                kind = "text"
            features.append({"column": name, "inferred_type": kind, "unique_count": unique, "unique_ratio": round(unique_ratio, 4)})

            score = 0
            reasons: list[str] = []
            if words & target_words:
                score += 4
                reasons.append("name resembles a common outcome")
            if position == len(df.columns) - 1:
                score += 1
                reasons.append("final-column dataset convention")
            if kind in {"categorical", "boolean", "numeric"} and 2 <= unique <= min(20, max(rows // 2, 2)):
                score += 2
                likely_task = "classification"
                reasons.append("low cardinality is compatible with classification")
            elif kind == "numeric" and unique > 10:
                score += 1
                likely_task = "regression"
                reasons.append("continuous values are compatible with regression")
            else:
                likely_task = "unknown"
            if kind == "identifier" or words & id_words:
                score -= 5
                reasons.append("identifier-like uniqueness")
            if score > 0:
                candidates.append({"column": name, "heuristic_score": score, "likely_task": likely_task, "evidence": reasons})
        candidates.sort(key=lambda item: item["heuristic_score"], reverse=True)
        return {"features": features, "target_candidates": candidates[:5], "warning": "Business intent is required to confirm the true target."}

    @tool("distributions")
    def distributions() -> dict[str, Any]:
        """Measure numeric skew and zeros plus categorical concentration."""
        result: dict[str, Any] = {}
        for column in df.columns:
            series = df[column].dropna()
            if pd.api.types.is_numeric_dtype(series):
                result[str(column)] = {
                    "kind": "numeric",
                    "skewness": _json_safe(float(series.skew())) if len(series) > 2 else None,
                    "zero_percent": round(float(series.eq(0).mean() * 100), 2) if len(series) else 0.0,
                    "min": _json_safe(series.min()) if len(series) else None,
                    "median": _json_safe(series.median()) if len(series) else None,
                    "max": _json_safe(series.max()) if len(series) else None,
                }
            else:
                counts = series.value_counts()
                result[str(column)] = {
                    "kind": "categorical_or_text",
                    "unique_count": int(series.nunique()),
                    "top_value_percent": round(float(counts.iloc[0] / len(series) * 100), 2) if len(counts) else 0.0,
                }
        return result

    @tool("redundancy_analysis")
    def redundancy_analysis() -> dict[str, Any]:
        """Find constant, near-constant, identifier-like, and highly correlated features."""
        constant: list[str] = []
        near_constant: list[dict[str, Any]] = []
        identifiers: list[str] = []
        for column in df.columns:
            series = df[column].dropna()
            unique = int(series.nunique())
            if unique <= 1:
                constant.append(str(column))
            elif len(series):
                top_percent = float(series.value_counts().iloc[0] / len(series) * 100)
                if top_percent >= 95:
                    near_constant.append({"column": str(column), "top_value_percent": round(top_percent, 2)})
                if unique / len(series) >= 0.98:
                    identifiers.append(str(column))
        numeric = df.select_dtypes(include="number")
        high_correlations: list[dict[str, Any]] = []
        if numeric.shape[1] >= 2:
            matrix = numeric.corr()
            for index, left in enumerate(matrix.columns):
                for right in matrix.columns[index + 1:]:
                    value = matrix.loc[left, right]
                    if pd.notna(value) and abs(float(value)) >= 0.9:
                        high_correlations.append({"left": str(left), "right": str(right), "correlation": round(float(value), 4)})
        return {"constant_columns": constant, "near_constant_columns": near_constant, "identifier_candidates": identifiers, "high_correlations": high_correlations[:25]}

    # this tool is used to generate preprocessing advice for the data.
    @tool("preprocessing_advice")
    def preprocessing_advice() -> dict[str, Any]:
        """Generate transformation and exclusion evidence from measured column properties."""
        transformations: list[dict[str, str]] = [] # this is the list of transformations that will be applied to the data
        exclusions: list[dict[str, str]] = [] # this is the list of columns that will be excluded from the analysis
        for column in df.columns:
            series = df[column]
            name = str(column)
            non_null = series.dropna()
            missing_percent = float(series.isna().mean() * 100)
            if missing_percent > 0:
                transformations.append({"column": name, "suggestion": "impute or add a missingness indicator", "evidence": f"{missing_percent:.2f}% missing"})
            if pd.api.types.is_numeric_dtype(non_null) and len(non_null) > 2:
                skew = float(non_null.skew())
                if math.isfinite(skew) and abs(skew) >= 1:
                    transformations.append({"column": name, "suggestion": "review a power or log-like transformation", "evidence": f"skewness={skew:.4f}"})
            elif int(non_null.nunique()) <= 50:
                transformations.append({"column": name, "suggestion": "encode as categorical", "evidence": f"{int(non_null.nunique())} observed categories"})
            unique_ratio = int(non_null.nunique()) / max(len(non_null), 1)
            if int(non_null.nunique()) <= 1:
                exclusions.append({"column": name, "reason": "constant or entirely missing"})
            elif unique_ratio >= 0.98 and not pd.api.types.is_float_dtype(non_null):
                exclusions.append({"column": name, "reason": "identifier-like uniqueness"})
        return {"transformations": transformations, "exclusion_candidates": exclusions}

    # this tool is used to plan the plots that will be generated by the agent.
    @tool("plot_plan")
    def plot_plan() -> dict[str, Any]:
        """Select bounded EDA plot specifications appropriate for this dataset."""
        plots: list[dict[str, Any]] = []

        # Every dataset gets a missingness chart so users can quickly compare
        # which columns may require removal, imputation, or further review.
        if len(df.columns):
            plots.append({
                "plot_type": "missingness_bar",
                "columns": [str(c) for c in df.columns],
                "title": "Missing values by column",
                "rationale": "Compare missingness across features.",
            })

        # Histograms show distribution shape, while box plots make spread and
        # possible IQR outliers easy to inspect. Limit this to four numeric
        # columns so wide datasets do not generate hundreds of images.
        numeric = [str(c) for c in df.select_dtypes(include="number").columns]
        for column in numeric[:4]:
            plots.extend([
                {"plot_type": "histogram", "columns": [column], "title": f"Distribution of {column}", "rationale": "Inspect shape, skew, and multiple modes."},
                {"plot_type": "box_plot", "columns": [column], "title": f"Outliers in {column}", "rationale": "Inspect spread and IQR outlier candidates."},
            ])

        # A correlation heatmap is meaningful only when AT LEAST 2 numeric
        # columns exist. Twenty columns keeps the chart readable and bounded.
        if len(numeric) >= 2:
            plots.append({"plot_type": "correlation_heatmap", "columns": numeric[:20], "title": "Numeric feature correlations", "rationale": "Inspect linear relationships and redundancy."})

        # Count plots summarize balance and rare values for the first three
        # categorical-like columns. The renderer displays their top values.
        categorical = [str(c) for c in df.select_dtypes(include=["object", "category", "bool", "string"]).columns]
        for column in categorical[:3]:
            plots.append({"plot_type": "count_plot", "columns": [column], "title": f"Most common values of {column}", "rationale": "Inspect category balance and rare levels."})

        # This final cap protects response size and browser rendering time even
        # if more plot-selection rules are added later.
        return {"plots": plots[:12]}

    # return the list of tools
    return [
        overview, missing_values, statistics, correlations, outliers, duplicates,
        value_counts, feature_analysis, distributions, redundancy_analysis,
        preprocessing_advice, plot_plan,
    ]
