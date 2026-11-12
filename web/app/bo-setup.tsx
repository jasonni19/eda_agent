"use client";

import { useState } from "react";
import SurrogateResults from "./surrogate-results";

type Category = string | number | boolean;
export type DatasetColumn = {
  name: string;
  numeric: boolean;
  suggested_type: "continuous" | "integer" | "categorical";
  bounds: [number, number] | null;
  categories: Category[];
};
type Recommendation = { column: string; rationale?: string };
type Variable = { name: string; type: DatasetColumn["suggested_type"]; min: string; max: string; categories: Category[] };
type Constraint = { variable: string; operator: "<=" | ">=" | "=="; value: string };
const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export type BOProfile = {
  dataset: { n_samples: number; n_complete_samples: number; n_complete_in_domain_samples: number;
    n_training_samples: number; training_sample_dimension_ratio: number | null };
  search_space: { n_dimensions: number; space_type: string; continuous_variables: number; integer_variables: number;
    categorical_variables: number; full_cardinality: string; full_search_space_size: number | null };
  objective: { column: string; direction: string; mean: number | null; standard_deviation: number | null; skewness: number | null };
  coverage: { in_bounds_percentage: number | null; out_of_bounds_samples: number; unevaluable_domain_samples: number;
    constraint_violation_samples: number; constraint_unevaluable_samples: number };
  data_quality: { duplicate_configurations: number; max_absolute_correlation: number | null;
    missing_percentage: Record<string, number | null>; invalid_value_counts: Record<string, number>;
    outliers: Record<string, { count: number | null; percentage: number | null }> };
  warnings: string[];
};

function displayNumber(value: number | null, percentage = false) {
  return value === null ? "Unavailable" : `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 }).format(value)}${percentage ? "%" : ""}`;
}

function downloadJson(content: string, filename: string) {
  const url = URL.createObjectURL(new Blob([content], { type: "application/json" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename; anchor.click(); URL.revokeObjectURL(url);
}

export function BOProfileSummary({ profile }: { profile: BOProfile }) {
  return <div className="bo-profile-result">
    <p role="status">Analysis complete for <strong>{profile.objective.column}</strong> ({profile.objective.direction}).</p>
    <dl className="bo-profile-metrics">
      <div><dt>Historical observations</dt><dd>{profile.dataset.n_samples}</dd></div>
      <div><dt>Usable training observations</dt><dd>{profile.dataset.n_training_samples}</dd></div>
      <div><dt>Decision dimensions</dt><dd>{profile.search_space.n_dimensions}</dd></div>
      <div><dt>Within configured domain</dt><dd>{displayNumber(profile.coverage.in_bounds_percentage, true)}</dd></div>
    </dl>
    <p>Training observations have complete values, lie inside the configured domain, and satisfy every constraint.</p>
    <table className="bo-profile-table"><caption>Problem characteristics</caption><tbody>
      <tr><th scope="row">Search space</th><td>{profile.search_space.space_type}: {profile.search_space.continuous_variables} continuous, {profile.search_space.integer_variables} integer, {profile.search_space.categorical_variables} categorical</td></tr>
      <tr><th scope="row">Full search-space size</th><td>{profile.search_space.full_cardinality === "non_finite" ? "Non-finite (includes continuous variables)" : displayNumber(profile.search_space.full_search_space_size)}</td></tr>
      <tr><th scope="row">Complete observations / in-domain</th><td>{profile.dataset.n_complete_samples} / {profile.dataset.n_complete_in_domain_samples}</td></tr>
      <tr><th scope="row">Training observations per dimension</th><td>{displayNumber(profile.dataset.training_sample_dimension_ratio)}</td></tr>
      <tr><th scope="row">Outside domain / unevaluable domain</th><td>{profile.coverage.out_of_bounds_samples} / {profile.coverage.unevaluable_domain_samples}</td></tr>
      <tr><th scope="row">Constraint violations / unevaluable</th><td>{profile.coverage.constraint_violation_samples} / {profile.coverage.constraint_unevaluable_samples}</td></tr>
      <tr><th scope="row">Duplicate configurations</th><td>{profile.data_quality.duplicate_configurations}</td></tr>
      <tr><th scope="row">Maximum absolute correlation</th><td>{displayNumber(profile.data_quality.max_absolute_correlation)}</td></tr>
      <tr><th scope="row">Objective mean / standard deviation</th><td>{displayNumber(profile.objective.mean)} / {displayNumber(profile.objective.standard_deviation)}</td></tr>
      <tr><th scope="row">Objective skewness</th><td>{displayNumber(profile.objective.skewness)}</td></tr>
    </tbody></table>
    <details><summary>Column quality details</summary><div className="bo-table-scroll"><table className="bo-profile-table">
      <caption>Selected columns</caption><thead><tr><th scope="col">Column</th><th scope="col">Missing</th><th scope="col">Invalid values</th><th scope="col">IQR outliers</th></tr></thead>
      <tbody>{Object.entries(profile.data_quality.missing_percentage).map(([name, missing]) => <tr key={name}>
        <th scope="row">{name}</th><td>{displayNumber(missing, true)}</td><td>{profile.data_quality.invalid_value_counts[name] ?? 0}</td>
        <td>{profile.data_quality.outliers[name] ? displayNumber(profile.data_quality.outliers[name].percentage, true) : "Not applicable"}</td>
      </tr>)}</tbody>
    </table></div></details>
    {profile.warnings.length ? <details className="bo-profile-warnings" open><summary>Warnings and observations ({profile.warnings.length})</summary>
      <ul>{profile.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></details> : <p>No analyzer warnings.</p>}
    <button type="button" onClick={() => downloadJson(JSON.stringify(profile, null, 2), "bo_profile.json")}>Download bo_profile.json</button>
    <details><summary>View full profile JSON</summary><pre>{JSON.stringify(profile, null, 2)}</pre></details>
  </div>;
}

export default function BOSetup({ sessionId, columns, recommendations }: {
  sessionId: string; columns: DatasetColumn[]; recommendations: Recommendation[];
}) {
  const numeric = columns.filter((column) => column.numeric && column.bounds);
  const suggested = recommendations.find((item) => numeric.some((column) => column.name === item.column));
  const [objective, setObjective] = useState(suggested?.column ?? numeric[0]?.name ?? "");
  const [direction, setDirection] = useState<"maximize" | "minimize">("maximize");
  const [variables, setVariables] = useState<Variable[]>([]);
  const [constraints, setConstraints] = useState<Constraint[]>([]);
  const [confirmed, setConfirmed] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState("");
  const [profile, setProfile] = useState<BOProfile | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState("");

  function changed() { setConfirmed(false); setSaved(""); setError(""); setProfile(null); setProfileError(""); }
  function updateVariable(index: number, patch: Partial<Variable>) {
    changed(); setVariables(variables.map((variable, i) => i === index ? { ...variable, ...patch } : variable));
  }
  function constraintCategories(constraint: Constraint) {
    const variable = variables.find((item) => item.name === constraint.variable);
    const column = columns.find((item) => item.name === constraint.variable);
    return variable?.type === "categorical" ? variable.categories : column && !column.numeric ? column.categories : null;
  }
  const problems: string[] = [];
  if (!numeric.some((column) => column.name === objective)) problems.push("Select a numeric objective with finite observations.");
  if (!variables.length) problems.push("Select at least one variable that you can control.");
  for (const variable of variables) {
    if (variable.type === "categorical") {
      if (!variable.categories.length) problems.push(`${variable.name}: select at least one category.`);
    } else {
      const min = Number(variable.min), max = Number(variable.max);
      if (!variable.min.trim() || !variable.max.trim() || !Number.isFinite(min) || !Number.isFinite(max) || min >= max)
        problems.push(`${variable.name}: enter a finite minimum below the maximum.`);
      if (variable.type === "integer" && (!Number.isSafeInteger(min) || !Number.isSafeInteger(max)))
        problems.push(`${variable.name}: use whole-number bounds within the supported numeric range.`);
    }
  }
  constraints.forEach((constraint, index) => {
    const categories = constraintCategories(constraint);
    const value = Number(constraint.value);
    if (!constraint.variable || !columns.some((column) => column.name === constraint.variable)) problems.push(`Constraint ${index + 1}: choose a column.`);
    if (categories) {
      if (constraint.operator !== "==" || !categories.some((category) => JSON.stringify(category) === constraint.value))
        problems.push(`Constraint ${index + 1}: choose an allowed category and equality.`);
    } else if (!constraint.value.trim() || !Number.isFinite(value)) problems.push(`Constraint ${index + 1}: enter a finite numeric value.`);
  });
  for (const variable of variables) {
    const restrictions = constraints.filter((constraint) => constraint.variable === variable.name);
    if (variable.type === "categorical") {
      if (new Set(restrictions.map((constraint) => constraint.value)).size > 1) problems.push(`${variable.name}: equality constraints conflict.`);
    } else {
      let low = Number(variable.min), high = Number(variable.max);
      for (const constraint of restrictions) {
        if (constraint.operator === ">=" || constraint.operator === "==") low = Math.max(low, Number(constraint.value));
        if (constraint.operator === "<=" || constraint.operator === "==") high = Math.min(high, Number(constraint.value));
      }
      if (low > high || (variable.type === "integer" && Math.ceil(low) > Math.floor(high))) problems.push(`${variable.name}: constraints leave no feasible values within its bounds.`);
    }
  }
  const problem = {
    objective: { column: objective, direction },
    decision_variables: variables.map((variable) => variable.type === "categorical"
      ? { name: variable.name, type: variable.type, categories: variable.categories }
      : { name: variable.name, type: variable.type, bounds: [Number(variable.min), Number(variable.max)] }),
    constraints: constraints.map((constraint) => ({ ...constraint, value: constraintCategories(constraint)
      ? (constraintCategories(constraint)?.find((category) => JSON.stringify(category) === constraint.value) ?? "") : Number(constraint.value) })),
  };

  async function loadProfile() {
    setProfileLoading(true); setProfileError(""); setProfile(null);
    try {
      const endpoint = `${apiBase}/analyses/${sessionId}/bo-profile`;
      let response = await fetch(endpoint);
      // Sessions saved before the analyzer was introduced may need generation.
      if (response.status === 404 || response.status === 409) response = await fetch(endpoint, { method: "POST" });
      const body = await response.json();
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Could not load the BO profile. Check that the updated API is running.");
      setProfile(body as BOProfile);
    } catch (caught) { setProfileError(caught instanceof Error ? caught.message : "Could not load the BO profile."); }
    finally { setProfileLoading(false); }
  }

  async function save() {
    if (problems.length || !confirmed || saving) return;
    setSaving(true); setError(""); setProfile(null); setProfileError("");
    try {
      const response = await fetch(`${apiBase}/analyses/${sessionId}/bo-problem`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(problem),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail :
        Array.isArray(body.detail) ? body.detail.map((item: { msg: string }) => item.msg).join("; ") : "Could not save this configuration.");
      setSaved(JSON.stringify(body, null, 2));
      await loadProfile();
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not save this configuration."); }
    finally { setSaving(false); }
  }

  function download() {
    downloadJson(saved, "bo_problem.json");
  }

  return <section id="bo-setup" className="bo-setup" aria-label="Bayesian Optimization setup">
    <h2>Set up Bayesian Optimization</h2>
    <p>Define what to improve and what you can control. Confirm your setup to run the deterministic BO Problem Analyzer and review its results below.</p>
    <fieldset disabled={saving || profileLoading}>
      <legend>1. Objective</legend>
      <p>The numeric measurement you want to improve.</p>
      <div className="bo-grid">
        <label>Objective column<select value={objective} onChange={(event) => {
          changed(); setObjective(event.target.value); setVariables(variables.filter((variable) => variable.name !== event.target.value));
        }}><option value="">Choose a numeric column</option>{numeric.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}</select></label>
        <label>Goal<select value={direction} onChange={(event) => { changed(); setDirection(event.target.value as typeof direction); }}>
          <option value="maximize">Maximize (higher is better)</option><option value="minimize">Minimize (lower is better)</option>
        </select></label>
      </div>
      {recommendations.filter((item) => numeric.some((column) => column.name === item.column)).map((item) =>
        <p className="bo-hint" key={item.column}>EDA suggestion: <strong>{item.column}</strong>. {item.rationale} You can choose another objective.</p>)}
      {!numeric.length && <p className="bo-error">This dataset has no numeric objective with finite observations.</p>}
    </fieldset>
    <fieldset disabled={saving || profileLoading}>
      <legend>2. Decision Variables</legend>
      <p>Select only columns you can deliberately change. Suggested bounds reflect observed data; edit them to your feasible operating limits.</p>
      {columns.map((column) => {
        const index = variables.findIndex((variable) => variable.name === column.name);
        const variable = variables[index];
        return <div className="bo-variable" key={column.name}>
          <label className="bo-check"><input type="checkbox" checked={!!variable} disabled={column.name === objective} onChange={(event) => {
            changed(); setVariables(event.target.checked ? [...variables, {
              name: column.name, type: column.suggested_type, min: column.bounds ? String(column.bounds[0]) : "",
              max: column.bounds ? String(column.bounds[1]) : "", categories: [...column.categories],
            }] : variables.filter((item) => item.name !== column.name));
          }} />{column.name} <small>{column.name === objective ? "Objective" : column.suggested_type}</small></label>
          {variable && <div className="bo-grid">
            <label>Variable type<select value={variable.type} onChange={(event) => updateVariable(index, { type: event.target.value as Variable["type"] })}>
              {column.numeric && <><option value="continuous">Continuous</option><option value="integer">Integer</option></>}<option value="categorical">Categorical</option>
            </select></label>
            {variable.type === "categorical" ? <div className="bo-categories"><span>Allowed categories</span>{column.categories.map((category, categoryIndex) =>
              <label className="bo-check" key={categoryIndex}><input type="checkbox" checked={variable.categories.includes(category)} onChange={(event) => updateVariable(index, {
                categories: event.target.checked ? [...variable.categories, category] : variable.categories.filter((item) => item !== category),
              })} />{String(category) || "(empty text)"}</label>)}</div> : <>
              <label>Minimum<input type="number" step={variable.type === "integer" ? "1" : "any"} value={variable.min} onChange={(event) => updateVariable(index, { min: event.target.value })} /></label>
              <label>Maximum<input type="number" step={variable.type === "integer" ? "1" : "any"} value={variable.max} onChange={(event) => updateVariable(index, { max: event.target.value })} /></label>
            </>}
          </div>}
        </div>;
      })}
    </fieldset>
    <fieldset disabled={saving || profileLoading}>
      <legend>3. Constraints (Optional)</legend>
      <p>Limit a column with a simple comparison. Categorical columns support equality only.</p>
      {constraints.map((constraint, index) => {
        const categories = constraintCategories(constraint);
        function update(patch: Partial<Constraint>) { changed(); setConstraints(constraints.map((item, i) => i === index ? { ...item, ...patch } : item)); }
        return <div className="bo-constraint" key={index}>
          <div className="bo-grid">
            <label>Column<select value={constraint.variable} onChange={(event) => update({ variable: event.target.value, value: "", operator: "==" })}>
              <option value="">Choose a column</option>{columns.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}
            </select></label>
            <label>Comparison<select value={constraint.operator} onChange={(event) => update({ operator: event.target.value as Constraint["operator"] })}>
              {!categories && <><option value="<=">Less than or equal to (≤)</option><option value=">=">Greater than or equal to (≥)</option></>}<option value="==">Equal to (=)</option>
            </select></label>
            <label>Value{categories ? <select value={constraint.value} onChange={(event) => update({ value: event.target.value })}>
              <option value="">Choose a category</option>{categories.map((category, i) => <option key={i} value={JSON.stringify(category)}>{String(category) || "(empty text)"}</option>)}
            </select> : <input type="number" step="any" value={constraint.value} onChange={(event) => update({ value: event.target.value })} />}</label>
          </div>
          {constraint.variable && !variables.some((variable) => variable.name === constraint.variable) &&
            <p className="bo-hint">{constraint.variable} is not a decision variable. This requirement is recorded, but cannot be directly enforced by changing that column alone.</p>}
          <button type="button" onClick={() => { changed(); setConstraints(constraints.filter((_, i) => i !== index)); }}>Remove constraint {index + 1}</button>
        </div>;
      })}
      <button type="button" onClick={() => { changed(); setConstraints([...constraints, { variable: "", operator: "==", value: "" }]); }}>Add constraint</button>
    </fieldset>
    <fieldset disabled={saving || profileLoading}>
      <legend>4. Review and Confirm</legend>
      <p><strong>{direction === "maximize" ? "Maximize" : "Minimize"} {objective || "your objective"}</strong> by changing {variables.length ? variables.map((variable) => variable.name).join(", ") : "the variables you select"}. {constraints.length} constraint(s).</p>
      {problems.length > 0 && <ul className="bo-error" aria-live="polite">{problems.map((problem, index) => <li key={index}>{problem}</li>)}</ul>}
      {!problems.length && <details><summary>Review problem JSON</summary><pre>{JSON.stringify(problem, null, 2)}</pre></details>}
      <label className="bo-check"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />I reviewed the objective, controllable variables, limits, and constraints.</label>
      <button className="bo-primary" type="button" disabled={!!problems.length || !confirmed || saving} onClick={save}>{saving ? "Saving…" : "Confirm and save BO problem"}</button>
      {error && <p role="alert" className="bo-error">{error}</p>}
      {saved && <div role="status"><p>Saved bo_problem.json for this dataset.</p><button type="button" onClick={download}>Download bo_problem.json</button></div>}
    </fieldset>
    <section className="bo-profile" aria-label="BO Problem Analyzer" aria-busy={saving || profileLoading}>
      <h3>5. BO Problem Analyzer</h3>
      {!saved && !saving && <p>Confirm and save your BO setup above to calculate dataset quality, search-space coverage, and usable training observations.</p>}
      {(saving || profileLoading) && <p role="status">Analyzing your configured BO problem…</p>}
      {profileError && <div><p role="alert" className="bo-error">Your BO problem is saved, but the profile could not be loaded: {profileError}</p>
        <button type="button" disabled={profileLoading || saving} onClick={loadProfile}>Retry loading analyzer results</button></div>}
      {profile && <BOProfileSummary profile={profile} />}
    </section>
    {profile && <SurrogateResults sessionId={sessionId} />}
  </section>;
}
