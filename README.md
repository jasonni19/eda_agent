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
Surrogate recommendations are available after profiling. The separate Optimization
Benchmark page executes direct LLM optimization and agent-selected BO against
the bundled fixed pretrained evaluator (see the benchmark section below).

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

## Surrogate selection extension

The existing upload, EDA, BO setup and deterministic analyzer are retained. After
saving BO setup and reviewing the profile, click **Recommend a surrogate**.
The recommendation section displays confidence, rationale, compatible alternatives,
warnings and a downloadable `surrogate_report.json`. Edit setup to change the problem;
regenerate to request fresh advice.

New session-scoped endpoints:

- `POST /analyses/{session_id}/bo-recommend`: validate the saved profile, select a
  surrogate and atomically persist the report.
- `GET /analyses/{session_id}/bo-report`: retrieve the report, rejecting stale
  problem or profile fingerprints.

`surrogates.py` owns strict Pydantic contracts and selection. It reuses the EDA
LangChain model factory and agent structured-output convention. Set `BO_MODEL` to
override `EDA_MODEL`. Two bounded attempts use a 30-second provider timeout and
no transport retries. Missing credentials, provider errors or invalid structured
advice produce a labeled, low-confidence deterministic fallback. The model sees
only the BO profile and configuration, never raw observations. Unsupported model
names, extra parameters, incompatible kernels and uncertainty strategies are
rejected. Provider failures are logged without prompts or dataset values.

The planned executable configurations are GaussianProcess/Matern with predictive
standard deviation, or RandomForest/ExtraTrees with ensemble spread. Tree spread
is heuristic uncertainty, not calibrated Gaussian uncertainty. Recommendations
are heuristics, not cross-validated rankings. Noise and smoothness remain unknown.

`bo_engine.py` defines separate preprocessing, fitting, uncertainty, acquisition,
evaluator and engine protocols. **Execution for arbitrary uploaded datasets is
not enabled.** The benchmark supplies fitting and candidate proposal with its
verified evaluator. To extend execution to other datasets, supply adapters (including categorical
encoding/scaling), a feasible candidate optimizer, durable run/feedback storage,
and a real evaluator or external experimental feedback. Non-controllable output
constraints need a feasibility strategy. Do not infer new objective measurements
from the uploaded CSV. No execution endpoint is advertised as working.

The current local application uses opaque UUID session identifiers as capability
references, matching existing EDA storage. It has no authenticated multi-user
ownership boundary; production multi-user hosting requires authentication,
authorization and retention controls. Do not share session identifiers.

Run backend checks with `python -m pytest -q`. Run frontend checks from `web` with
`npm run test:bo`, `npm run lint`, and `npm test`. No new dependencies are required.

## Optimization Benchmark

Open **Optimization Benchmark** from the EDA header, or visit
`http://localhost:3000/benchmark`. The existing upload and EDA workflow remains
available on the home page.

From the project root, install the added runtime dependencies and verify the
bundled pretrained artifact:

```powershell
conda activate eda_agent
python -m pip install -e ".[dev]"
python scripts/prepare_benchmark_model.py
eda-api
```

In a second terminal, run `cd web` and `npm run dev`. Use your existing
`OPENAI_API_KEY` in `.env`. `BENCHMARK_LLM_MODEL` selects the direct optimizer;
`EDA_MODEL` and `BO_MODEL` select the EDA and surrogate-selection models. All
resolved model names are captured when a job is submitted. Optional
`BENCHMARK_MODEL_DIR` and `BENCHMARK_RUN_DIR` override the local model/run folders.

The default comparison uses 20 initial observations, 50 new evaluations per
method, and seeds 0–9. Both methods receive exactly the same initial observations
within each seed and see only their own later observations. Each seed runs EDA
and surrogate selection once, on the initial dataset only. The surrogate family
is fixed for that seed; its parameters are refitted each iteration. Default runs
make paid API calls; click **Run Comparison** to start one. No paid 10-seed run is
executed automatically during installation or tests.

### Verified fixed pretrained evaluator

The bundled model is the author's native `xgb.ubj` from
[shalev396/california-housing](https://huggingface.co/shalev396/california-housing),
revision `3b04cdba5f8f9433bb135dfc3be2ef5c64ff5179`, SHA-256
`cc2160311b6ff9b338997e541a632895549e12921ba40b73b040a3d8d0ee5a10`.
The author's MIT license, model card, preprocessing source and config are retained
under `benchmark_models/california-housing/`. The loader checks both artifact and
config hashes before any experiment. It does not train, import downloaded Python,
load pickle, or download training data. The preparation script verifies cached
files or downloads exact pinned files; changed checksums fail explicitly.

The model has eight raw inputs plus three engineered ratios. To keep the
optimization small, the benchmark varies three inputs within the author's public
[demo input limits](https://huggingface.co/spaces/shalev396/california-housing/blob/main/app.py):

| Adjustable input | Type | Domain |
|---|---|---|
| MedInc | Continuous | 0.5–15.0 |
| HouseAge | Integer | 1–52 |
| AveRooms | Continuous | 1.1–20.0 |

The rooms interval is a subset of the author's 1–20 slider. Five inputs remain at
the author's defaults: AveBedrms=1.10, Population=1425, AveOccup=3.07,
Latitude=35.63, Longitude=-119.57. The source UI snapshot and its checksum are
retained. Input ordering, float64 unscaled ratios and USD output flooring/rounding
match the exported preprocessing contract. These are supported demo input
ranges, not a guarantee that every joint configuration resembles training data.
This benchmarks predictions of a fixed historical census model; it is not causal
housing advice or a current valuation model.

### Components and experimental accounting

`src/eda_agent/benchmark/` separates validated schemas, the pretrained loader,
feasible sampling, direct LLM optimization, BO adapters, runner, metrics, storage
and API routes. It reuses `agent.py`, `bo.py`, `bo_analyzer.py`, `surrogates.py`, and
atomic artifact writing. The direct LLM receives the full history, problem and
remaining budget each step. It has no surrogate, acquisition function, evaluator
object or model internals. Invalid/duplicate outputs get at most two attempts;
then a seeded feasible fallback is recorded. Provider failures, invalid proposal
previews and fallback events remain distinguishable in the CSV artifacts.

BO supports GaussianProcess/Matern (nu=2.5), RandomForest and ExtraTrees. Numeric
inputs are scaled by domain bounds; categorical inputs use fixed one-hot
encoding; targets are standardized. GPs return predictive standard deviation.
Tree uncertainty is the sample spread of individual-tree predictions, an
uncalibrated heuristic. Acquisition maximizes oriented mean + 2 × uncertainty over
a fresh feasible candidate pool (default 1,024), excludes observed candidates,
and respects input constraints. This is approximate acquisition search, not a
claim of an exact acquisition optimum. GP fitting warnings are saved per
iteration. Output constraints require a separate feasibility strategy and are
rejected by these adapters; the predefined benchmark has no output constraints.

Reference optimization is an independent 4,096-sample feasible search, executed
only after both methods finish. Its evaluations are counted separately from the
50-per-method budgets. The reference estimate may be beaten: estimated simple
regret remains signed, and is not represented as a known global regret.

The dashboard provides seed/aggregate trajectories, ±1 sample standard deviation
bands, paired final differences and a 95% Student-t interval when multiple seeds
exist. It also reports final mean/std/median/min/max, threshold crossing and
success rate, convergence area, fallback/invalid rates, setup/iterative timings,
provider calls, token usage and evaluator counts. API cost requires user-supplied
USD-per-million rates and complete token accounting; unavailable costs are null.
BO API cost includes its EDA and selection setup, and these phases are also shown
separately. Hosted LLM outputs are not guaranteed reproducible from a random seed.

### API and persisted artifacts

- `GET /benchmark/models`
- `POST /benchmark/run` (202 with an opaque run ID)
- `GET /benchmark/{run_id}/status`
- `GET /benchmark/{run_id}/results`
- `GET /benchmark/{run_id}/convergence`
- `GET /benchmark/{run_id}/download?artifact=all|summary|llm|bo`

Each `benchmark_runs/<run_id>/` holds config, ground-truth metadata, shared initial
CSV, EDA/problem/profile/selection reports, both iteration CSVs, summary and
progress state. `seed_<seed>/` holds the separate setup and CSV artifacts for each
seed. Top-level setup files mirror the first seed; top-level result CSVs aggregate
all seeds. Every convergence curve starts with the best initial value at x=0.
Failed/interrupted jobs retain partial per-seed artifacts; they do not receive a
completed comparison summary. Browser history remembers opaque IDs locally.

Only one background job is admitted per API process (429 while busy), with at
most 10 seeds, 100 initial observations, and 100 new evaluations per method/seed.
Use the existing single-process `eda-api` command. Jobs cooperate with shutdown
between bounded API calls; restarting does not silently resume or duplicate a
run. This local application has capability-ID isolation and no authenticated
multi-user ownership. Multi-worker or hosted deployments need a durable job queue
and authenticated storage authorization.

Validation commands:

```powershell
python -m pytest -q
cd web
npm run lint
npm test
```

Automated tests use mocked LLMs and lightweight test objectives for fairness
checks. A separate smoke test loads the actual pretrained artifact and runs real
BO steps against it. These tests are not scientific evidence that either method
outperforms the other.
