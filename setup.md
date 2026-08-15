# LangChain EDA Agent

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

Open `http://localhost:3000`, upload a CSV, and download the JSON report.

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
- `tools.py` — contains the seven pandas analysis tools.
- `agent.py` — connects the model, tools, and schema.
- `cli.py` — provides the `eda-agent` command.

## Limitations

- Pearson correlation does not prove causation.
- IQR outliers are review flags, not confirmed errors.
- Value counts are limited to 10 values and correlations to 25 pairs.
- pandas infers CSV data types automatically.

For production use, consider input schemas, size limits, PII protection, retries, monitoring, and
human review.
