"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import ConvergenceChart, { Curves, labels, methods } from "./convergence-chart";

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
type Model = { id: string; name: string; available: boolean; limitation: string; problem: unknown; source: string; revision: string };
type Status = { status: string; phase: string; seed: number | null; iteration: number; method: string | null;
  completed_evaluations: number; total_evaluations: number; method_completed: Record<string, number>; error: string | null };
type Stats = { mean: number; std: number | null; median: number; minimum: number; maximum: number };
type Metric = { final_best: Stats; estimated_simple_regret: Stats; auc: Stats; success_rate: number | null;
  mean_evaluations_to_threshold_among_successes: number | null; mean_iterative_runtime_seconds: number;
  fallback_count: number; fallback_frequency: number; invalid_candidate_rate: number; new_evaluations: number };
type Selection = { seed: number; report: { configuration: { model: string }; source: string; rationale: string[];
  alternatives: { model: string; reason: string }[]; warnings: string[] }; profile: { dataset: { n_samples: number }; search_space: { n_dimensions: number; space_type: string } } };
type Iteration = { seed: number; method: string; iteration: number; configuration: Record<string, unknown>; objective: number;
  best_objective: number; estimated_simple_regret: number; recommendation_runtime: number; evaluation_runtime: number; fallback: boolean };
type Usage = { model_calls: number; input_tokens: number | null; output_tokens: number | null; estimated_cost: number | null; api_latency_seconds: number };
export type BenchmarkResults = { config: { seed: number; random_seeds: number; optimization_direction: string; [key: string]: unknown };
  iterations: Iteration[]; summary: { metrics: Record<string, Metric>; convergence: Curves;
    reference: { estimated_optimum: number; evaluations: number; note: string };
    seeds: { seed: number }[]; surrogate_selections: Selection[]; llm_usage: Record<string, Usage>;
    runtime: Record<string, number | string>; evaluation_counts: Record<string, number>;
    paired_comparison: { oriented_bo_minus_llm: Stats; bo_win_rate: number; tie_rate: number; mean_difference_95_percent_t_interval: number[] | null } } };
const number = (value: number | null | undefined) => value == null ? "Unavailable" : value.toLocaleString(undefined, { maximumFractionDigits: 3 });
const historyKey = "object-eda-benchmark-history";
type History = { id: string; name: string };

export function BenchmarkProgress({ status, runId }: { status: Status; runId: string }) {
  return <section className="benchmark-card" aria-live="polite"><h2>Experiment progress</h2><p><strong>{status.status}</strong> · {status.phase}</p>
    <p>Seed {status.seed ?? "—"} · iteration {status.iteration} · {status.method ? labels[status.method as typeof methods[number]] : "Setup / summary"}</p>
    <progress aria-label="Completed new function evaluations" max={status.total_evaluations} value={status.completed_evaluations} />
    <p>{status.completed_evaluations} completed / {status.total_evaluations - status.completed_evaluations} remaining new evaluations.</p>
    <ul>{methods.map(method => <li key={method}>{labels[method]}: {status.method_completed[method]} evaluations completed</li>)}</ul>
    {status.error && <p role="alert" className="bo-error">{status.error}</p>}
    {["failed", "interrupted"].includes(status.status) && <a href={`${apiBase}/benchmark/${runId}/download`}>Download partial artifacts</a>}
  </section>;
}

export function ResultsDashboard({ results, runId }: { results: BenchmarkResults; runId: string }) {
  const [seedView, setSeedView] = useState("mean");
  const index = seedView === "mean" ? null : Number(seedView);
  const selection = results.summary.surrogate_selections[index ?? 0];
  const metrics = results.summary.metrics;
  const paired = results.summary.paired_comparison;
  const selectedSeed = results.summary.seeds[index ?? 0].seed;
  const cost = (method: string) => {
    const groups = method === "llm_optimizer" ? [results.summary.llm_usage.llm_optimizer] :
      [results.summary.llm_usage.eda, results.summary.llm_usage.selection];
    return groups.every(group => group.estimated_cost !== null) ? groups.reduce((sum, group) => sum + (group.estimated_cost ?? 0), 0) : null;
  };
  const rows: [string, (method: string) => number | null][] = [
    ["Mean final best objective", method => metrics[method].final_best.mean],
    ["Standard deviation across seeds", method => metrics[method].final_best.std],
    ["Median final best", method => metrics[method].final_best.median],
    ["Minimum final best", method => metrics[method].final_best.minimum],
    ["Maximum final best", method => metrics[method].final_best.maximum],
    ["Mean estimated simple regret", method => metrics[method].estimated_simple_regret.mean],
    ["Mean evaluations to threshold (successful seeds)", method => metrics[method].mean_evaluations_to_threshold_among_successes],
    ["Success rate (fraction of seeds)", method => metrics[method].success_rate],
    ["Mean convergence area", method => metrics[method].auc.mean],
    ["Mean iterative runtime per seed (s)", method => metrics[method].mean_iterative_runtime_seconds],
    ["Total estimated API cost (USD)", cost],
    ["New function evaluations (all seeds)", method => metrics[method].new_evaluations],
    ["Fallback frequency", method => metrics[method].fallback_frequency],
    ["Invalid proposal rate", method => metrics[method].invalid_candidate_rate],
  ];
  return <div className="benchmark-results">
    <section className="benchmark-card"><h2>Performance summary</h2>
      <div className="bo-table-scroll"><table className="bo-profile-table"><caption>Paired comparison across {results.summary.seeds.length} seeds</caption>
        <thead><tr><th scope="col">Metric</th>{methods.map(method => <th scope="col" key={method}>{labels[method]}</th>)}</tr></thead>
        <tbody>{rows.map(([label, value]) => <tr key={label}><th scope="row">{label}</th>{methods.map(method => <td key={method}>{number(value(method))}</td>)}</tr>)}</tbody>
      </table></div>
      <p>Positive paired difference favors BO: <strong>{number(paired.oriented_bo_minus_llm.mean)}</strong>. BO win rate: {number(paired.bo_win_rate * 100)}%; ties: {number(paired.tie_rate * 100)}%.</p>
      <p>95% paired Student-t interval: {paired.mean_difference_95_percent_t_interval?.map(number).join(" to ") ?? "Unavailable with one seed"}. Small-seed estimates are uncertain.</p>
      <p>Estimated reference: {number(results.summary.reference.estimated_optimum)} from {results.summary.reference.evaluations} independent samples. {results.summary.reference.note}</p>
      <p className="benchmark-muted">BO API cost includes EDA and selection; baseline API cost covers direct recommendations. Rates must be supplied in setup; missing usage or rates is shown as unavailable.</p>
      {metrics.llm_optimizer.fallback_count > 0 && <p className="benchmark-warning">The LLM baseline used {metrics.llm_optimizer.fallback_count} fallback samples. Interpret this run with its fallback frequency.</p>}
    </section>
    <label className="benchmark-seed">Chart and seed details<select value={seedView} onChange={event => setSeedView(event.target.value)}>
      <option value="mean">Mean across seeds</option>{results.summary.seeds.map((seed, i) => <option key={seed.seed} value={i}>Seed {seed.seed}</option>)}
    </select></label>
    <ConvergenceChart curves={results.summary.convergence} seedIndex={index} reference={results.summary.reference.estimated_optimum} direction={results.config.optimization_direction} />
    <section className="benchmark-card"><h2>Surrogate selection · seed {selection.seed}</h2>
      {index === null && <p>Showing the first seed’s selection. Switch seeds above to inspect each independent setup.</p>}
      <h3>{selection.report.configuration.model}</h3><p>Selection source: {selection.report.source}.</p>
      <details><summary>Surrogate configuration</summary><pre>{JSON.stringify(selection.report.configuration, null, 2)}</pre></details>
      <p>{selection.profile.dataset.n_samples} initial observations · {selection.profile.search_space.n_dimensions} dimensions · {selection.profile.search_space.space_type} space.</p>
      <ul>{selection.report.rationale.map((reason, i) => <li key={i}>{reason}</li>)}</ul>
      <h3>Alternatives</h3><ul>{selection.report.alternatives.map(item => <li key={item.model}>{item.model}: {item.reason}</li>)}</ul>
      <details><summary>Selection warnings</summary><ul>{selection.report.warnings.map((warning, i) => <li key={i}>{warning}</li>)}</ul></details>
    </section>
    <section className="benchmark-card"><h2>Runtime and API usage</h2>
      <details><summary>Setup costs, iterative costs and evaluator call counts</summary><pre>{JSON.stringify({ runtime: results.summary.runtime, api_usage: results.summary.llm_usage, ground_truth_calls: results.summary.evaluation_counts }, null, 2)}</pre></details>
      <details><summary>Review benchmark configuration</summary><pre>{JSON.stringify(results.config, null, 2)}</pre></details>
      <div className="benchmark-downloads">{[["all", "Download all artifacts (ZIP)"], ["summary", "Download summary JSON"], ["llm", "Download LLM CSV"], ["bo", "Download BO CSV"]].map(([artifact, label]) =>
        <a key={artifact} href={`${apiBase}/benchmark/${runId}/download?artifact=${artifact}`}>{label}</a>)}</div>
    </section>
    <section className="benchmark-card"><h2>Iteration history · seed {selectedSeed}</h2>
      <div className="bo-table-scroll"><table className="bo-profile-table"><caption>Separate feedback histories</caption><thead><tr>{["Method", "Evaluation", "Configuration", "Objective", "Best so far", "Estimated regret", "Status"].map(label => <th key={label} scope="col">{label}</th>)}</tr></thead>
      <tbody>{results.iterations.filter(row => row.seed === selectedSeed).map(row => <tr key={`${row.method}-${row.iteration}`}>
        <td>{labels[row.method as typeof methods[number]]}</td><td>{row.iteration}</td><td><code>{JSON.stringify(row.configuration)}</code></td>
        <td>{number(row.objective)}</td><td>{number(row.best_objective)}</td><td>{number(row.estimated_simple_regret)}</td><td>{row.fallback ? "Seeded fallback" : "Valid proposal"}</td>
      </tr>)}</tbody></table></div>
    </section>
  </div>;
}

export default function BenchmarkPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [history, setHistory] = useState<History[]>([]);
  const [runId, setRunId] = useState("");
  const [status, setStatus] = useState<Status | null>(null);
  const [results, setResults] = useState<BenchmarkResults | null>(null);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  useEffect(() => {
    fetch(`${apiBase}/benchmark/models`).then(async response => { if (!response.ok) throw new Error("Could not load models"); return response.json(); })
      .then(setModels).catch(() => setError("Could not load benchmark models. Start the updated backend."));
    const historyTimer = setTimeout(() => {
      try { const stored = JSON.parse(localStorage.getItem(historyKey) ?? "[]");
        if (Array.isArray(stored)) setHistory(stored.filter(item => item && typeof item.id === "string" && typeof item.name === "string").slice(0, 50));
      } catch { /* History is optional. */ }
    }, 0);
    return () => clearTimeout(historyTimer);
  }, []);
  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const response = await fetch(`${apiBase}/benchmark/${runId}/status`, { signal: controller.signal });
        const state = await response.json();
        if (!response.ok) throw new Error(state.detail ?? "Could not load progress.");
        setStatus(state);
        if (state.status === "completed") {
          const response = await fetch(`${apiBase}/benchmark/${runId}/results`, { signal: controller.signal });
          const body = await response.json();
          if (!response.ok) throw new Error(body.detail ?? "Could not load results.");
          setResults(body); setError("");
        } else if (state.status === "queued" || state.status === "running") timer = setTimeout(poll, 2000);
      } catch (caught) {
        if (!controller.signal.aborted) { setError(caught instanceof Error ? caught.message : "Could not load progress."); timer = setTimeout(poll, 5000); }
      }
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [runId]);

  async function start(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault(); setStarting(true); setError("");
    const form = new FormData(event.currentTarget);
    const optionalNumber = (name: string) => String(form.get(name) ?? "").trim() ? Number(form.get(name)) : null;
    const config = { model_id: form.get("model_id"), name: form.get("name"), initial_samples: Number(form.get("initial_samples")),
      optimization_iterations: Number(form.get("optimization_iterations")), random_seeds: Number(form.get("random_seeds")),
      seed: Number(form.get("seed")), optimization_direction: form.get("optimization_direction"), threshold: optionalNumber("threshold"),
      input_cost_per_million: optionalNumber("input_rate"), output_cost_per_million: optionalNumber("output_rate") };
    try {
      const response = await fetch(`${apiBase}/benchmark/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config) });
      const body = await response.json();
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Invalid benchmark configuration.");
      const next = [{ id: body.run_id, name: String(config.name) }, ...history].slice(0, 50);
      setHistory(next); try { localStorage.setItem(historyKey, JSON.stringify(next)); } catch { /* Run still proceeds. */ }
      setStatus(null); setResults(null); setRunId(body.run_id);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not start benchmark."); }
    finally { setStarting(false); }
  }
  const active = starting || (!!runId && !status && !results) || status?.status === "running" || status?.status === "queued";
  return <main className="benchmark-page bo-setup">
    <nav><Link href="/">← Object EDA</Link></nav>
    <header><p className="benchmark-eyebrow">OBJECT EDA · EXPERIMENTS</p><h1>Optimization Benchmark</h1>
      <p>Compare direct adaptive LLM optimization with Bayesian optimization using an agent-selected surrogate.</p></header>
    <form className="benchmark-card" onSubmit={start}><h2>Configure comparison</h2>
      <fieldset disabled={active}><legend>Shared problem and evaluation budget</legend><div className="bo-grid">
        <label>Pretrained ground-truth model<select name="model_id" required>{models.map(model => <option key={model.id} value={model.id}>{model.name}</option>)}</select></label>
        <label>Benchmark name<input name="name" defaultValue="Optimization comparison" maxLength={120} required /></label>
        <label>Initial observations<input name="initial_samples" type="number" min="5" max="100" defaultValue="20" required /></label>
        <label>New evaluations per method per seed<input name="optimization_iterations" type="number" min="1" max="100" defaultValue="50" required /></label>
        <label>Number of paired seeds<input name="random_seeds" type="number" min="1" max="10" defaultValue="10" required /></label>
        <label>Starting random seed<input name="seed" type="number" min="0" max="1000000" defaultValue="0" required /></label>
        <label>Direction<select name="optimization_direction"><option value="maximize">Maximize</option><option value="minimize">Minimize</option></select></label>
        <label>Objective threshold (optional)<input name="threshold" type="number" step="any" /></label>
      </div>
      <details><summary>Optional estimated API pricing</summary><p>Use USD per million tokens. Leave blank when unavailable; these rates apply to all configured LLM calls.</p><div className="bo-grid">
        <label>Input rate<input name="input_rate" type="number" min="0" step="any" /></label><label>Output rate<input name="output_rate" type="number" min="0" step="any" /></label>
      </div></details>
      {models.map(model => <div className="benchmark-model" key={model.id}><p>{model.limitation}</p>
        <details><summary>Review fixed model and supported search domain</summary><p><a href={model.source} target="_blank" rel="noreferrer">Model source</a> · revision <code>{model.revision}</code></p><pre>{JSON.stringify(model.problem, null, 2)}</pre></details>
        {!model.available && <p className="bo-error">Model files or dependencies are unavailable. See backend setup documentation.</p>}</div>)}
      <p>Default: 20 shared initial evaluations, 50 new evaluations per method, 10 seeds. An additional independent reference search estimates regret. Runs use paid model API calls and may take several minutes. One run is allowed at a time.</p>
      <button className="bo-primary" type="submit" disabled={active || !models.some(model => model.available)}>{starting ? "Starting…" : "Run Comparison"}</button>
      </fieldset>
    </form>
    {error && <p className="bo-error" role="alert">{error}</p>}
    {status && <BenchmarkProgress status={status} runId={runId} />}
    {results && <ResultsDashboard key={runId} results={results} runId={runId} />}
    <section className="benchmark-card"><h2>Experiment history</h2><p>Runs opened or started in this browser. Results persist on the backend.</p>
      {!history.length && <p>No experiments yet.</p>}
      <ul>{history.map(item => <li key={item.id}><button type="button" disabled={!!active} onClick={() => { setRunId(item.id); setStatus(null); setResults(null); setError(""); }}>{item.name} · {item.id.slice(0, 8)}</button></li>)}</ul>
    </section>
  </main>;
}
