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
deterministic step; the LLM only supplies optional objective suggestions. No BO
analyzer, surrogate selection, or optimization engine runs yet.

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

The Pydantic `BOProblem` model in `src/eda_agent/bo.py` defines the JSON contract
and is also exposed in FastAPI's OpenAPI schema at `http://127.0.0.1:8000/docs`.
The next pipeline stage can read the session's `bo_problem.json` or use the GET endpoint.
See `examples/bo_problem.json` for an example. Validation rejects duplicate variables,
non-numeric objectives, invalid categories/bounds/constraints, and conflicting constraints
on decision variables; numeric bounds may extend beyond observed ranges.

Run backend validation/integration tests with `python -m pytest`. Run the BO form's
component-rendering tests from `web/` with `npm run test:bo`.

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
