import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

// Compile the actual component without starting the Cloudflare hosting runtime.
const require = createRequire(import.meta.url);
const source = await readFile(new URL("../app/bo-setup.tsx", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: {
  jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020,
} }).outputText.replace(/from "(react(?:\/jsx-runtime)?)"/g,
  (_, name) => `from ${JSON.stringify(pathToFileURL(require.resolve(name)).href)}`);
const { default: BOSetup, BOProfileSummary } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);

const columns = [
  { name: "efficiency", numeric: true, suggested_type: "continuous", bounds: [0.5, 0.8], categories: [0.5, 0.8] },
  { name: "pressure", numeric: true, suggested_type: "integer", bounds: [1, 10], categories: [1, 10] },
  { name: "material", numeric: false, suggested_type: "categorical", bounds: null, categories: ["Steel", "Aluminum"] },
];
function render(recommendations = [], dataset = columns) {
  return renderToStaticMarkup(createElement(BOSetup, { sessionId: "test", columns: dataset, recommendations }));
}

test("numeric EDA suggestion is selected; users must opt in to controllable variables", () => {
  const html = render([{ column: "pressure", rationale: "Example suggestion" }]);
  assert.match(html, /value="pressure" selected=""/);
  assert.match(html, /EDA suggestion/);
  assert.doesNotMatch(html, /type="checkbox"[^>]*checked/);
  assert.match(html, /Select at least one variable that you can control/);
  assert.match(html, /disabled="">Confirm and save BO problem/);
});

test("nonnumeric recommendations cannot become objectives", () => {
  const html = render([{ column: "material" }]);
  assert.match(html, /value="efficiency" selected=""/);
  assert.doesNotMatch(html, /<option value="material"/);
  assert.match(html, /material/); // Still visible as a potential decision variable.
  assert.match(html, /Minimize \(lower is better\)/);
});

test("dataset without numeric observations clearly blocks submission", () => {
  const html = render([], [columns[2]]);
  assert.match(html, /no numeric objective with finite observations/);
  assert.match(html, /disabled="">Confirm and save BO problem/);
});

test("the setup provides all four sections and explains observed bounds", () => {
  const html = render();
  for (const section of ["1. Objective", "2. Decision Variables", "3. Constraints (Optional)", "4. Review and Confirm"])
    assert.ok(html.includes(section));
  assert.match(html, /Suggested bounds reflect observed data/);
  assert.match(html, /Add constraint/);
  assert.match(html, /5\. BO Problem Analyzer/);
  assert.match(html, /Confirm and save your BO setup above/);
});

const profile = JSON.parse(await readFile(new URL("../../examples/bo_profile.json", import.meta.url), "utf8"));

test("analyzer results render training counts, coverage, warnings, and profile download", () => {
  const html = renderToStaticMarkup(createElement(BOProfileSummary, { profile }));
  assert.match(html, /Usable training observations/);
  assert.match(html, /<dd>4<\/dd>/);
  assert.match(html, /75%/);
  assert.match(html, /efficiency/);
  assert.match(html, /Non-finite \(includes continuous variables\)/);
  assert.match(html, /Warnings and observations/);
  assert.match(html, /Constraint on cost is not directly controllable/);
  assert.match(html, /Download bo_profile.json/);
  assert.match(html, /View full profile JSON/);
});

test("unavailable statistics are labeled and zero observations remain visible", () => {
  const insufficient = structuredClone(profile);
  insufficient.objective.mean = null;
  insufficient.objective.standard_deviation = null;
  insufficient.objective.skewness = null;
  insufficient.dataset.n_training_samples = 0;
  insufficient.coverage.in_bounds_percentage = null;
  insufficient.warnings = [];
  const html = renderToStaticMarkup(createElement(BOProfileSummary, { profile: insufficient }));
  assert.match(html, /Unavailable/);
  assert.match(html, /<dd>0<\/dd>/);
  assert.match(html, /No analyzer warnings/);
  assert.doesNotMatch(html, /NaN|undefined/);
});
