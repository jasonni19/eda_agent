"use client";
import { useState } from "react";

export type Curve = { mean: number[]; std: (number | null)[]; seeds: number[][] };
export type Curves = Record<string, Curve>;
export const methods = ["llm_optimizer", "agent_selected_bo"] as const;
export const labels = { llm_optimizer: "LLM Optimization", agent_selected_bo: "Agent-Selected BO" };
const colors = { llm_optimizer: "#a16b00", agent_selected_bo: "#0879ad" };

export default function ConvergenceChart({ curves, seedIndex, reference, direction }: {
  curves: Curves; seedIndex: number | null; reference: number; direction: string;
}) {
  const [showReference, setShowReference] = useState(false);
  const [inspection, setInspection] = useState(0);
  const values = Object.fromEntries(methods.map(method => [method, seedIndex === null ? curves[method].mean : curves[method].seeds[seedIndex]]));
  const count = values.llm_optimizer.length;
  const all = methods.flatMap(method => values[method].flatMap((value, index) => {
    const spread = seedIndex === null ? (curves[method].std[index] ?? 0) : 0;
    return [value - spread, value + spread];
  }));
  if (showReference) all.push(reference);
  const low = Math.min(...all), high = Math.max(...all);
  const padding = Math.max((high - low) * .08, 1);
  const min = low - padding, max = high + padding;
  const x = (index: number) => 85 + index / Math.max(1, count - 1) * 650;
  const y = (value: number) => 300 - (value - min) / (max - min) * 260;
  const point = Math.min(inspection, count - 1);
  const format = (value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  function band(method: typeof methods[number]) {
    const upper = values[method].map((value, index) => `${x(index)},${y(value + (curves[method].std[index] ?? 0))}`);
    const lower = values[method].map((value, index) => `${x(index)},${y(value - (curves[method].std[index] ?? 0))}`).reverse();
    return [...upper, ...lower].join(" ");
  }
  return <section className="benchmark-card" aria-label="Convergence chart">
    <h2>Convergence</h2>
    <p>Best objective including the shared initial observations at evaluation 0. {direction === "maximize" ? "Higher" : "Lower"} is better.</p>
    <label className="benchmark-inline"><input type="checkbox" checked={showReference} onChange={event => setShowReference(event.target.checked)} />Show independent estimated reference</label>
    <svg viewBox="0 0 780 355" role="img" aria-label="Best objective versus additional function evaluations"
      onMouseMove={event => { const box = event.currentTarget.getBoundingClientRect();
        setInspection(Math.max(0, Math.min(count - 1, Math.round(((event.clientX - box.left) / box.width * 780 - 85) / 650 * (count - 1))))); }}>
      <title>LLM optimization and agent-selected BO convergence</title>
      {[0, .25, .5, .75, 1].map(fraction => <g key={fraction}>
        <line x1="85" x2="735" y1={y(min + fraction * (max - min))} y2={y(min + fraction * (max - min))} stroke="#dce4eb" />
        <text x="77" y={y(min + fraction * (max - min)) + 4} textAnchor="end" fontSize="11">{format(min + fraction * (max - min))}</text>
        <text x={85 + 650 * fraction} y="322" textAnchor="middle" fontSize="11">{Math.round((count - 1) * fraction)}</text>
      </g>)}
      {methods.map(method => <g key={method}>
        {seedIndex === null && curves[method].std.some(value => value !== null) && <polygon points={band(method)} fill={colors[method]} fillOpacity=".12" />}
        <polyline points={values[method].map((value, index) => `${x(index)},${y(value)}`).join(" ")} fill="none" stroke={colors[method]} strokeWidth="2.5" />
        <circle cx={x(point)} cy={y(values[method][point])} r="4" fill={colors[method]} />
      </g>)}
      {showReference && <line x1="85" x2="735" y1={y(reference)} y2={y(reference)} stroke="#737373" strokeDasharray="6 5" />}
      <line x1={x(point)} x2={x(point)} y1="40" y2="300" stroke="#777" strokeDasharray="3 4" />
      <text x="410" y="349" textAnchor="middle" fontSize="12">Additional function evaluations</text>
    </svg>
    <label>Inspect evaluation {point}<input type="range" min="0" max={count - 1} value={point} onChange={event => setInspection(Number(event.target.value))} /></label>
    <div className="benchmark-legend" aria-live="polite">{methods.map(method => <span key={method} style={{ color: colors[method] }}><strong>{labels[method]}</strong>: {format(values[method][point])}</span>)}</div>
    <p className="benchmark-muted">{seedIndex === null ? "Bands show ±1 sample standard deviation across seeds, not confidence intervals." : "Individual seed; no variability band."}</p>
  </section>;
}
