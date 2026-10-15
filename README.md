# Object EDA Agent

Creates a schema-validated EDA report from a CSV:

```text
CSV → pandas → LangChain tools → LLM summary → Pydantic → eda_report.json
```

- **pandas** calculates the results.
- **the LLM** interprets the results.
- **Pydantic** validates the final report.

## Setup

Requires Python 3.11+.

```powershell
conda activate eda_agent
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Add your API key to `.env`, then run:

```powershell
eda-agent path\to\dataset.csv
```

### How to start the Chat interface

Start the analysis API:

```powershell
eda-api
```

In a second terminal, start the web interface:

```powershell
cd web
npm install
npm run dev
```

Open `http://localhost:3000`, upload a CSV, review the executive summary and generated charts,
then download individual PNG files or these three outputs:

- `<input_file>_eda_report.json` — validated machine-readable report
- `<input_file>_eda_report.md` — human-readable report
- `eda_plots.zip` — downloadable `eda_plots/` folder containing all PNG charts

CLI runs create the same JSON and Markdown reports plus a real `eda_plots/` directory.

### Bayesian Optimization setup

After each web analysis, a BO setup form appears below the report. Choose a numeric
objective and maximize/minimize, explicitly select the columns you can control,
and edit continuous/integer bounds or allowed observed categories. Observed ranges
are suggestions, not guaranteed feasible operating limits. Optionally add simple
`<=`, `>=`, or `==` constraints (categorical constraints support equality).
Constraints on columns outside the selected decision variables are labeled as
requirements that cannot be directly enforced by changing that column alone.

Review and confirm the settings to save and download `bo_problem.json`. This is a
deterministic step; the LLM only supplies optional objective suggestions. Saving
also generates `bo_profile.json` with deterministic BO-specific statistics.
The BO Problem Analyzer section below the setup displays training counts, domain
coverage, data quality, objective statistics, and warnings, with a profile JSON download.
Results appear after confirming and saving the setup; edits clear the displayed
profile until the updated settings are saved again.
Surrogate selection and the optimization engine are not implemented.

Successful web analyses retain `dataset.csv` and `eda_report.json` under
`analysis_sessions/<session_id>/`; confirmed settings are saved as
`bo_problem.json` in that same directory. Set `EDA_SESSION_DIR` in `.env` to choose
another storage location. This local storage persists across API restarts and has
no automatic expiry; remove session directories when their artifacts are no longer needed.
The original upload filename remains in the EDA report.

API integration points:

- `POST /analyze`: existing response plus `session_id` and deterministic `dataset_columns`.
- `GET /analyses/{session_id}/bo-setup`: column metadata and EDA target suggestions.
- `POST /analyses/{session_id}/bo-problem`: validates against the saved CSV and atomically saves the problem.
- `GET /analyses/{session_id}/bo-problem`: retrieves the saved problem (404 until saved).
- `POST /analyses/{session_id}/bo-profile`: regenerates the profile from the three stored inputs;
  accepts an optional JSON object containing analyzer threshold overrides.
- `GET /analyses/{session_id}/bo-profile`: retrieves the saved profile (404 until generated;
  409 if its problem fingerprint differs from the current saved problem).

The Pydantic `BOProblem` model in `src/eda_agent/bo.py` defines the JSON contract
and is also exposed in FastAPI's OpenAPI schema at `http://127.0.0.1:8000/docs`.
The next pipeline stage can read the session's `bo_problem.json` or use the GET endpoint.
See `examples/bo_problem.json` for an example. Validation rejects duplicate variables,
non-numeric objectives, invalid categories/bounds/constraints, and conflicting constraints
on decision variables; numeric bounds may extend beyond observed ranges.

Run backend validation/integration tests with `python -m pytest`. Run the BO form's
component-rendering tests from `web/` with `npm run test:bo`.

### Deterministic BO problem profile

`bo_analyzer.py` profiles only the selected decision variables, objective, and columns
referenced by constraints. It never invokes an LLM. It uses the CSV as the numerical
authority and `bo_problem.json` as the definition of the optimization goal and domain.
The existing EDA schema stores statistics primarily as narrative findings, so the
analyzer does not parse numbers from those strings. It checks compatible structured
EDA row/column metadata for mismatches and tolerates absent optional EDA information.

The versioned `BOProfile` Pydantic contract is in `bo_profile.py` and is exposed in
FastAPI's OpenAPI schema. Profiles include the problem fingerprint, thresholds,
dimension/domain counts, missing and invalid data, numeric correlations, IQR outliers,
objective distribution, repeated observations, coverage, and warnings. They contain
no timestamps or randomness, so identical inputs and thresholds produce identical JSON.
The session's `bo_profile.json` is refreshed whenever its BO settings are saved.
Each artifact is replaced atomically; BO file operations are serialized within the
single API process. This local storage design is not a multi-process transaction store.

Count and statistic definitions:

- Each selected variable counts as one dimension. Raw and training sample/dimension
  ratios are both reported. Categorical combinations use the configured category
  counts; the discrete component also multiplies inclusive integer-domain sizes.
  A space containing continuous variables has `full_cardinality: "non_finite"` and
  `full_search_space_size: null`.
- Complete rows have finite objective/numeric values, integral integer variables,
  and nonmissing finite categorical values. Numeric strings can be parsed; unparseable
  values, booleans in numeric columns, and infinities are invalid, not missing.
- Domain membership additionally requires configured bounds/categories. Coverage
  percentages use **all rows** as the denominator. Out-of-bounds rows have evaluable
  decision values but fail a domain check; missing/non-finite decision values are
  counted separately as unevaluable. Distinct/duplicate configurations use valid
  complete decision vectors, regardless of objective availability or domain membership;
  duplicates count rows beyond the first instance of each configuration.
- `n_complete_in_domain_samples` counts complete in-domain rows before constraints.
  `n_training_samples` additionally requires every constraint to be evaluable and
  satisfied. Constraint violations and unknown values are separate counts. A missing
  constraint-only column produces a warning and unknown rows; missing objective or
  decision columns are fatal. Constraints on non-decision columns remain observational
  requirements and are explicitly labeled as not directly controllable.
- Pearson correlations use pairwise finite numeric decision observations (at least
  two), without the objective or categorical variables. Undefined correlations are null.
  IQR fences use pandas linear-interpolated quartiles and require four finite values;
  outlier percentages use finite values in that column as the denominator. Outliers
  remain in training counts and are not automatically treated as errors.
- Objective standard deviation uses `ddof=1` (two observations minimum); pandas sample
  skewness requires three observations. Constant objectives have zero skewness when
  at least three values exist. Undefined/non-finite statistics are null. Repeated-objective
  summaries use complete observations, include out-of-domain configurations, and describe
  ranges and distinct values without estimating a noise model.

`BOAnalyzerConfig` centralizes configurable heuristics (inclusive thresholds unless noted):

| Setting | Default | Meaning |
| --- | --- | --- |
| `high_correlation_threshold` | 0.85 | Absolute pairwise Pearson correlation |
| `outlier_iqr_multiplier` | 1.5 | IQR fence multiplier |
| `high_outlier_fraction` | 0.05 | Fraction of finite values flagged, requiring at least one outlier |
| `strong_skewness_threshold` | 1.0 | Absolute objective skewness |
| `high_dimensional_threshold` | 20 | Selected dimensions |
| `low_sample_dimension_ratio` | 10 | Training ratio below this value is small |
| `high_categorical_cardinality` | 10 | Allowed categories in any variable |
| `near_constant_fraction` | 0.95 | Fraction equal to the most frequent observed value |
| `near_constant_absolute_tolerance` | 1e-12 | Absolute numeric range tolerance |
| `near_constant_relative_tolerance` | 1e-8 | Numeric range tolerance relative to maximum absolute observed value |

A nonempty variable/objective is near-constant if its most frequent value meets the
fraction threshold or its numeric range is at most absolute tolerance plus relative
tolerance times maximum absolute observed value (with a scale floor of 1). Constant
decision variables are listed separately. These are descriptive heuristics, not model recommendations.

Independent Python use (no API key or model call required):

```python
import json
from pathlib import Path
import pandas as pd
from eda_agent.bo import BOProblem
from eda_agent.bo_analyzer import analyze_bo_problem
from eda_agent.bo_profile import BOAnalyzerConfig

session = Path("analysis_sessions/<session_id>")
profile = analyze_bo_problem(
    pd.read_csv(session / "dataset.csv"),
    json.loads((session / "eda_report.json").read_text(encoding="utf-8")),
    BOProblem.model_validate_json((session / "bo_problem.json").read_text(encoding="utf-8")),
    BOAnalyzerConfig(high_dimensional_threshold=15),
)
print(profile.model_dump_json(indent=2))
```

`generate_session_profile(session_directory, config)` in `bo_service.py` loads all
three persisted inputs and atomically saves the resulting profile. Missing/invalid
data produces warnings, while malformed BO settings or missing required columns
raise validation errors. The analyzer can describe an empty dataset even though the
setup UI requires numeric observations before initially saving a problem.

See `examples/bo_profile.json` and `examples/bo_profile_input.csv`. Regenerate the
synthetic example after installing the package with `python examples/generate_bo_profile.py`.

Optional arguments: You can specify what model for the agent to use.

```powershell
eda-agent data.csv --model openai:gpt-4.1-mini --output reports\eda_report.json
```

Run tests without calling an LLM:

```powershell
pytest
```

## Code map

- `schemas.py` — defines and validates the JSON report.
- `tools.py` — contains deterministic quality, feature, target-candidate, distribution,
  redundancy, preprocessing, and plot-planning tools.
- `agent.py` — connects the model, tools, and schema.
- `cli.py` — provides the `eda-agent` command.

## Limitations

- Pearson correlation does not prove causation.
- IQR outliers are review flags, not confirmed errors.
- Value counts are limited to 10 values and correlations to 25 pairs.
- pandas infers CSV data types automatically.
- Target recommendations are evidence-based candidates; business intent is required to confirm
  the actual prediction target.

For production use, consider input schemas, size limits, PII protection, retries, monitoring, and
human review.
