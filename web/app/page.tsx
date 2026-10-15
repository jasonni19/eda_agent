"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import Image from "next/image";
import BOSetup, { DatasetColumn } from "./bo-setup";

type ChatMessage = {
  id: number;
  role: "assistant" | "user";
  text: string;
  file?: string;
  report?: Record<string, unknown>;
  plots?: PlotArtifact[];
  markdown?: { filename: string; content: string };
  plotsArchive?: { filename: string; data_uri: string };
  analysisTrace?: TraceStep[];
  sessionId?: string;
  datasetColumns?: DatasetColumn[];
};

type PlotArtifact = {
  title?: string;
  rationale?: string;
  filename?: string | null;
  image_data_uri?: string | null;
};

type TraceStep = {
  tool: string;
  label: string;
  operation: string;
  arguments: Record<string, unknown>;
  status: "completed" | "error";
  result_preview: string;
};

const progressSteps = [  // just a visual progress indicator
  "Reading column types and dataset shape",
  "Ranking possible target columns",
  "Checking missing values and duplicates",
  "Analyzing distributions and transformations",
  "Examining redundancy and outliers",
  "Planning key EDA visualizations",
  "Writing and validating your report",
];

function reportFilename(report: Record<string, unknown>) {
  const source = typeof report.source_file === "string" ? report.source_file : "dataset.csv";
  const originalName = source.split(/[\\/]/).pop() ?? "dataset.csv";
  const stem = originalName.replace(/\.[^.]+$/, "");
  const safeStem = stem.replace(/[<>:"/\\|?*]+/g, "_").trim() || "dataset";
  return `${safeStem}_eda_report.json`;
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [working, setWorking] = useState(false);
  const [progress, setProgress] = useState(0);
  const [activeFileName, setActiveFileName] = useState("");
  const [queuePosition, setQueuePosition] = useState({ current: 0, total: 0 });
  const [error, setError] = useState("");
  const [visibleChartMessages, setVisibleChartMessages] = useState<Set<number>>(new Set());
  const [visibleTraceMessages, setVisibleTraceMessages] = useState<Set<number>>(new Set());
  const [expandedTraceSteps, setExpandedTraceSteps] = useState<Set<string>>(new Set());
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 1,
      role: "assistant",
      text: "Hi — I’m Obi, your EDA assistant. Upload a CSV or multiple CSVs and I’ll inspect its quality, distributions, relationships, and unusual values, then package the findings as a validated JSON report. I will also provide downloadable charts and a markdown report.",
    },
  ]);
  const inputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, working, progress]);

  useEffect(() => {
    if (!working) return;
    const timer = window.setInterval(() => {
      setProgress((current) => Math.min(current + 1, progressSteps.length - 1));
    }, 2300);
    return () => window.clearInterval(timer);
  }, [working]);

  function acceptFiles(candidates: File[]) {
    setError("");
    const valid: File[] = [];
    const rejected: string[] = [];

    for (const candidate of candidates) {
      if (!candidate.name.toLowerCase().endsWith(".csv")) {
        rejected.push(`${candidate.name} is not a CSV`);
      } else if (candidate.size > 15 * 1024 * 1024) {
        rejected.push(`${candidate.name} is larger than 15 MB`);
      } else {
        valid.push(candidate);
      }
    }

    if (rejected.length) setError(rejected.join(" · "));
    setFiles((current) => {
      const existing = new Set(current.map((item) => `${item.name}:${item.size}:${item.lastModified}`));
      return [...current, ...valid.filter((item) => !existing.has(`${item.name}:${item.size}:${item.lastModified}`))];
    });
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFiles(Array.from(event.target.files ?? []));
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFiles(Array.from(event.dataTransfer.files ?? []));
  }

  async function analyze() {
    if (!files.length || working) return;
    const queue = [...files];
    setFiles([]);
    setError("");
    setWorking(true);
    setQueuePosition({ current: 1, total: queue.length });

    try {
      for (const [index, selected] of queue.entries()) {
        setActiveFileName(selected.name);
        setQueuePosition({ current: index + 1, total: queue.length });
        setProgress(0);
        setMessages((current) => [
          ...current,
          { id: Date.now() + index * 10, role: "user", text: "Analyze this dataset", file: selected.name },
        ]);

        try {
          const form = new FormData();
          form.append("file", selected);
          const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
          const response = await fetch(`${apiBase}/analyze`, { method: "POST", body: form });
          const body = await response.json();
          if (!response.ok) throw new Error(body.detail ?? "The analysis could not be completed.");
          const report = body.report as Record<string, unknown>;
          const plots = Array.isArray(body.plot_images) ? body.plot_images as PlotArtifact[] : [];
          const markdown = body.markdown_report as { filename: string; content: string };
          const plotsArchive = body.plots_archive as { filename: string; data_uri: string };
          const analysisTrace = Array.isArray(body.analysis_trace) ? body.analysis_trace as TraceStep[] : [];
          setProgress(progressSteps.length - 1);
          setMessages((current) => [
            ...current,
            {
              id: Date.now() + index * 10 + 1,
              role: "assistant",
              text: `Analysis complete for ${selected.name}. I found ${Array.isArray(report.findings) ? report.findings.length : 0} reportable findings and validated the output against the EDA schema.`,
              report,
              plots,
              markdown,
              plotsArchive,
              analysisTrace,
              sessionId: body.session_id,
              datasetColumns: body.dataset_columns,
            },
          ]);
        } catch (caught) {
          setMessages((current) => [
            ...current,
            {
              id: Date.now() + index * 10 + 1,
              role: "assistant",
              text: `${selected.name}: ${caught instanceof Error ? caught.message : "Something went wrong while analyzing this file."}`,
            },
          ]);
        }
      }
    } finally {
      setWorking(false);
      setActiveFileName("");
      setQueuePosition({ current: 0, total: 0 });
    }
  }

  function download(report: Record<string, unknown>) {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = reportFilename(report);
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function downloadPlot(plot: PlotArtifact) {
    if (!plot.image_data_uri) return;
    const anchor = document.createElement("a");
    anchor.href = plot.image_data_uri;
    anchor.download = plot.filename ?? "eda_chart.png";
    anchor.click();
  }

  function downloadMarkdown(markdown: { filename: string; content: string }) {
    const blob = new Blob([markdown.content], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = markdown.filename;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function downloadPlotsArchive(archive: { filename: string; data_uri: string }) {
    const anchor = document.createElement("a");
    anchor.href = archive.data_uri;
    anchor.download = archive.filename;
    anchor.click();
  }

  function toggleCharts(messageId: number) {
    setVisibleChartMessages((current) => {
      const next = new Set(current);
      if (next.has(messageId)) next.delete(messageId);
      else next.add(messageId);
      return next;
    });
  }

  function toggleTrace(messageId: number) {
    setVisibleTraceMessages((current) => {
      const next = new Set(current);
      if (next.has(messageId)) next.delete(messageId);
      else next.add(messageId);
      return next;
    });
  }

  function toggleTraceStep(messageId: number, stepIndex: number) {
    const key = `${messageId}:${stepIndex}`;
    setExpandedTraceSteps((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="Object EDA home">
          <span className="logo-frame">
            <Image src="/object-tech-logo.jpg" alt="Object Tech Inc." priority />
          </span>
          <span className="product-name">Object EDA</span>
        </a>
        <div className="system-status"><span className="status-dot" /> SYSTEM ONLINE</div>
      </header>

      <section className="workspace">
        <section className="chat-panel" aria-label="EDA assistant chat">
          <div className="chat-heading">
            <div><div><strong>EDA Agent</strong><small>● READY FOR INPUT</small></div></div>
           
          </div>

          <div className="messages" aria-live="polite">
            {messages.map((message) => (
              <div className={`message-wrap ${message.role}`} key={message.id}>
                {message.role === "assistant" && <span className="avatar">✦</span>}
                <div className="message">
                  <p>{message.text}</p>
                  {message.file && <div className="file-chip"><span>CSV</span><div><strong>{message.file}</strong><small>Ready for analysis</small></div></div>}
                  {message.report && (
                    <>
                      {typeof message.report.executive_summary === "string" && (
                        <section className="executive-summary">
                          <strong>Executive summary</strong>
                          <p>{message.report.executive_summary}</p>
                        </section>
                      )}
                      {message.analysisTrace && message.analysisTrace.length > 0 && (
                        <section className="analysis-trace">
                          <button
                            className="trace-toggle"
                            type="button"
                            aria-expanded={visibleTraceMessages.has(message.id)}
                            onClick={() => toggleTrace(message.id)}
                          >
                            <span>{visibleTraceMessages.has(message.id) ? "⌃" : "⌄"}</span>
                            {visibleTraceMessages.has(message.id) ? "Hide analysis steps" : "Show analysis steps"}
                            <small>{message.analysisTrace.length} tool calls</small>
                          </button>
                          {visibleTraceMessages.has(message.id) && (
                            <div className="trace-steps">
                              {message.analysisTrace.map((step, stepIndex) => {
                                const stepKey = `${message.id}:${stepIndex}`;
                                const expanded = expandedTraceSteps.has(stepKey);
                                return (
                                  <article className={`trace-step ${step.status}`} key={stepKey}>
                                    <button
                                      className="trace-step-heading"
                                      type="button"
                                      aria-expanded={expanded}
                                      onClick={() => toggleTraceStep(message.id, stepIndex)}
                                    >
                                      <span className="trace-chevron">{expanded ? "⌄" : "›"}</span>
                                      <span className="trace-check">{step.status === "completed" ? "✓" : "!"}</span>
                                      <strong>{step.label}</strong>
                                      <code>{step.tool}</code>
                                    </button>
                                    <p>{step.operation}</p>
                                    {expanded && (
                                      <div className="trace-detail">
                                        <div><small>Operation</small><code>{step.operation}</code></div>
                                        {Object.keys(step.arguments ?? {}).length > 0 && (
                                          <div><small>Parameters</small><pre>{JSON.stringify(step.arguments, null, 2)}</pre></div>
                                        )}
                                        <div><small>Result preview</small><pre>{step.result_preview}</pre></div>
                                      </div>
                                    )}
                                  </article>
                                );
                              })}
                            </div>
                          )}
                        </section>
                      )}
                      {message.plots && message.plots.some((plot) => plot.image_data_uri) && (
                        <section className="chart-gallery" aria-label="Generated EDA charts">
                          <div className="chart-gallery-heading">
                            <div>
                              <strong>Generated charts</strong>
                              <small>{message.plots.filter((plot) => plot.image_data_uri).length} PNG files</small>
                            </div>
                            <button
                              className="chart-toggle"
                              type="button"
                              aria-expanded={visibleChartMessages.has(message.id)}
                              onClick={() => toggleCharts(message.id)}
                            >
                              {visibleChartMessages.has(message.id) ? "Hide charts" : "Show charts"}
                            </button>
                          </div>
                          {visibleChartMessages.has(message.id) && message.plots.filter((plot) => plot.image_data_uri).map((plot, plotIndex) => (
                            <article className="chart-card" key={`${plot.filename ?? plot.title}-${plotIndex}`}>
                              {/* eslint-disable-next-line @next/next/no-img-element */}
                              <img src={plot.image_data_uri!} alt={plot.title ?? "EDA chart"} />
                              <div className="chart-caption">
                                <div><strong>{plot.title ?? "EDA chart"}</strong><small>{plot.rationale}</small></div>
                                <button onClick={() => downloadPlot(plot)}>Download PNG <span>↓</span></button>
                              </div>
                            </article>
                          ))}
                        </section>
                      )}
                      <div className="report-card">
                        <div className="report-icon">{'{ }'}</div>
                        <div><strong>{reportFilename(message.report)}</strong><small>Validated JSON report</small></div>
                        <button onClick={() => download(message.report!)}>Download <span>↓</span></button>
                      </div>
                      {message.markdown && (
                        <div className="report-card">
                          <div className="report-icon">MD</div>
                          <div><strong>{message.markdown.filename}</strong><small>Human-readable EDA summary</small></div>
                          <button onClick={() => downloadMarkdown(message.markdown!)}>Download <span>↓</span></button>
                        </div>
                      )}
                      {message.plotsArchive && (
                        <div className="report-card">
                          <div className="report-icon">ZIP</div>
                          <div><strong>{message.plotsArchive.filename}</strong><small>Folder containing all PNG charts</small></div>
                          <button onClick={() => downloadPlotsArchive(message.plotsArchive!)}>Download <span>↓</span></button>
                        </div>
                      )}
                      {message.sessionId && message.datasetColumns && <BOSetup
                        sessionId={message.sessionId}
                        columns={message.datasetColumns}
                        recommendations={Array.isArray(message.report.target_recommendations) ? message.report.target_recommendations : []}
                      />}
                    </>
                  )}
                </div>
              </div>
            ))}

            {working && (
              <div className="message-wrap assistant thinking-wrap">
                <span className="avatar">✦</span>
                <div className="thinking-card">
                  <div className="thinking-title"><span className="pulse-ring" />Analyzing {activeFileName}</div>
                  <p>{progressSteps[progress]}</p>
                  <div className="progress-track"><span style={{ width: `${18 + progress * 19}%` }} /></div>
                  <small>File {queuePosition.current} of {queuePosition.total} · Running deterministic pandas checks</small>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="composer-area">
            {files.length > 0 && <div className="selected-files">
              {files.map((selected, index) => (
                <div className="selected-file" key={`${selected.name}:${selected.size}:${selected.lastModified}`}>
                  <span>CSV</span><strong>{selected.name}</strong><small>{(selected.size / 1024).toFixed(1)} KB</small>
                  <button aria-label={`Remove ${selected.name}`} onClick={() => setFiles((current) => current.filter((_, itemIndex) => itemIndex !== index))}>×</button>
                </div>
              ))}
            </div>}
            <div
              className={`composer ${dragging ? "dragging" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <input ref={inputRef} type="file" accept=".csv,text/csv" multiple onChange={onFileChange} hidden />
              <button className="attach-button" onClick={() => inputRef.current?.click()} aria-label="Choose CSV files">＋</button>
              <button className="upload-copy" onClick={() => inputRef.current?.click()}>
                <strong>{files.length ? `${files.length} CSV${files.length === 1 ? "" : "s"} ready to analyze` : "Upload CSVs to begin"}</strong>
                <small>{files.length ? "Files will be processed one by one" : "Drag and drop or choose files · max 15 MB each"}</small>
              </button>
              <button className="send-button" disabled={!files.length || working} onClick={analyze} aria-label="Analyze CSV files">↑</button>
            </div>
            {error && <p className="error-text">{error}</p>}
            <p className="composer-note">AI-generated interpretation can make mistakes. Verify important findings.</p>
          </div>
        </section>
      </section>
    </main>
  );
}
