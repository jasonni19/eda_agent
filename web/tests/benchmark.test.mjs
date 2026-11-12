import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const require = createRequire(import.meta.url);
function compile(source) {
  return ts.transpileModule(source, { compilerOptions: {
    jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020,
  } }).outputText.replace(/from "(react(?:\/jsx-runtime)?)"/g,
    (_, name) => `from ${JSON.stringify(pathToFileURL(require.resolve(name)).href)}`);
}
const chartSource = await readFile(new URL("../app/benchmark/convergence-chart.tsx", import.meta.url), "utf8");
const chartUrl = `data:text/javascript;base64,${Buffer.from(compile(chartSource)).toString("base64")}`;
const { default: Chart } = await import(chartUrl);
// The router is irrelevant to component rendering. Supply an ordinary anchor for navigation.
const linkUrl = `data:text/javascript;base64,${Buffer.from(compile('import { createElement } from "react"; export default function Link(props) { return createElement("a", props); }')).toString("base64")}`;
const pageSource = await readFile(new URL("../app/benchmark/page.tsx", import.meta.url), "utf8");
const compiled = compile(pageSource).replace('"./convergence-chart"', JSON.stringify(chartUrl)).replace('"next/link"', JSON.stringify(linkUrl));
const { default: BenchmarkPage, ResultsDashboard, BenchmarkProgress } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);
const curve = { mean: [10, 12, 15], std: [1, 2, 3], seeds: [[9, 10, 12], [11, 14, 18]] };
const curves = { llm_optimizer: curve, agent_selected_bo: curve };
const stats = { mean: 15, std: 3, median: 15, minimum: 12, maximum: 18 };
const metric = { final_best: stats, estimated_simple_regret: stats, auc: stats, success_rate: null,
  mean_evaluations_to_threshold_among_successes: null, mean_iterative_runtime_seconds: 2,
  fallback_count: 1, fallback_frequency: .5, invalid_candidate_rate: 0, new_evaluations: 4 };
const usage = { model_calls: 1, input_tokens: 10, output_tokens: 10, estimated_cost: null, api_latency_seconds: 1 };
const selection = { seed: 0, report: { configuration: { model: "GaussianProcess" }, source: "llm", rationale: ["Initial profile"],
  alternatives: [{ model: "RandomForest", reason: "Tree uncertainty is heuristic" }], warnings: ["Limited observations"] },
  profile: { dataset: { n_samples: 20 }, search_space: { n_dimensions: 3, space_type: "mixed" } } };
const results = { config: { seed: 0, random_seeds: 2, optimization_direction: "maximize" }, iterations: [{ seed: 0,
  method: "llm_optimizer", iteration: 1, configuration: { x: 1 }, objective: 12, best_objective: 12, estimated_simple_regret: 3,
  recommendation_runtime: 1, evaluation_runtime: 0.1, fallback: false }], summary: {
  metrics: { llm_optimizer: metric, agent_selected_bo: metric }, convergence: curves,
  reference: { estimated_optimum: 20, evaluations: 100, note: "Estimated only" }, seeds: [{ seed: 0 }, { seed: 1 }],
  surrogate_selections: [selection, { ...selection, seed: 1 }], llm_usage: { eda: usage, selection: usage, llm_optimizer: usage },
  runtime: { total_seconds: 4 }, evaluation_counts: { total: 100 },
  paired_comparison: { oriented_bo_minus_llm: stats, bo_win_rate: .5, tie_rate: 0, mean_difference_95_percent_t_interval: [1, 3] } } };

test("benchmark configuration renders defaults, limits, history and paid-run notice", () => {
  const html = renderToStaticMarkup(createElement(BenchmarkPage));
  assert.match(html, /Optimization Benchmark/);
  for (const [name, value] of [["initial_samples", 20], ["optimization_iterations", 50], ["random_seeds", 10]])
    assert.match(html, new RegExp(`name="${name}"[^>]*value="${value}"`));
  assert.match(html, /Run Comparison/);
  assert.match(html, /paid model API calls/);
  assert.match(html, /Experiment history/);
});

test("progress shows seed, both method budgets, remaining evaluations and interruption", () => {
  const state = { status: "interrupted", phase: "Backend restart", seed: 2, iteration: 4, method: "agent_selected_bo",
    completed_evaluations: 7, total_evaluations: 100, method_completed: { llm_optimizer: 3, agent_selected_bo: 4 }, error: "Stopped" };
  const html = renderToStaticMarkup(createElement(BenchmarkProgress, { status: state, runId: "test-id" }));
  assert.match(html, /Seed 2/);
  assert.match(html, /7 completed \/ 93 remaining/);
  assert.match(html, /LLM Optimization: 3/);
  assert.match(html, /Agent-Selected BO: 4/);
  assert.match(html, /<progress[^>]*max="100"[^>]*value="7"/);
  assert.match(html, /Download partial artifacts/);
});

test("aggregate graph renders trajectories and sample-deviation bands", () => {
  const html = renderToStaticMarkup(createElement(Chart, { curves, seedIndex: null, reference: 20, direction: "maximize" }));
  assert.equal((html.match(/<polyline/g) ?? []).length, 2);
  assert.equal((html.match(/<polygon/g) ?? []).length, 2);
  assert.match(html, /shared initial observations at evaluation 0/);
  assert.match(html, /sample standard deviation/);
  assert.match(html, /type="range"/);
  assert.match(html, /Higher is better/);
});

test("individual seed chart omits uncertainty bands and supports minimization", () => {
  const html = renderToStaticMarkup(createElement(Chart, { curves, seedIndex: 1, reference: 20, direction: "minimize" }));
  assert.doesNotMatch(html, /<polygon/);
  assert.match(html, /Lower is better/);
  assert.match(html, /Individual seed/);
});

test("results show paired metrics, surrogate explanations, iteration history and downloads", () => {
  const html = renderToStaticMarkup(createElement(ResultsDashboard, { results, runId: "test-id" }));
  for (const label of ["Mean final best objective", "Standard deviation", "Mean estimated simple regret", "95% paired Student-t interval",
    "GaussianProcess", "Alternatives", "Iteration history", "Unavailable", "fallback samples"]) assert.ok(html.includes(label), label);
  for (const artifact of ["all", "summary", "llm", "bo"]) assert.ok(html.includes(`download?artifact=${artifact}`));
  assert.match(html, /seed details/);
});
