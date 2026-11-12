"use client";

import { useState } from "react";

type Report = {
  configuration: { model: string; kernel: string | null; uncertainty: string; [key: string]: unknown };
  confidence: string; source: string; rationale: string[];
  alternatives: { model: string; reason: string }[]; warnings: string[];
};
const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export default function SurrogateResults({ sessionId }: { sessionId: string }) {
  const [report, setReport] = useState<Report | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function generate() {
    setBusy(true); setError(""); setReport(null);
    try {
      const response = await fetch(`${apiBase}/analyses/${sessionId}/bo-recommend`, { method: "POST" });
      const body = await response.json();
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Could not generate recommendation.");
      setReport(body);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not generate recommendation."); }
    finally { setBusy(false); }
  }
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = "surrogate_report.json";
    anchor.click(); URL.revokeObjectURL(url);
  }
  return <section className="bo-profile" aria-label="Surrogate recommendation" aria-busy={busy}>
    <h3>6. Surrogate Recommendation</h3>
    <p>Compare Gaussian Process, Random Forest, and Extra Trees using the measured diagnostics above.</p>
    <button type="button" disabled={busy} onClick={generate}>{busy ? "Selecting surrogate…" : report ? "Regenerate recommendation" : "Recommend a surrogate"}</button>
    {error && <p className="bo-error" role="alert">{error}</p>}
    {report && <div aria-live="polite">
      <h4>{report.configuration.model}{report.configuration.kernel ? ` · ${report.configuration.kernel}` : ""}</h4>
      <p>Confidence: {report.confidence}. Source: {report.source === "llm" ? "AI selection" : "Deterministic fallback"}.</p>
      <ul>{report.rationale.map((reason, i) => <li key={i}>{reason}</li>)}</ul>
      <details><summary>Suggested configuration</summary><pre>{JSON.stringify(report.configuration, null, 2)}</pre></details>
      <h4>Alternatives and tradeoffs</h4>
      <ul>{report.alternatives.map((item) => <li key={item.model}><strong>{item.model}</strong>: {item.reason}</li>)}</ul>
      <h4>Limitations and warnings</h4><ul>{report.warnings.map((warning, i) => <li key={i}>{warning}</li>)}</ul>
      <p>Offline recommendation mode. Optimization execution requires a connected objective evaluator and fitting adapters.</p>
      <button type="button" onClick={download}>Download surrogate report</button>
      <a href="#bo-setup">Return to BO Setup</a>
    </div>}
  </section>;
}
