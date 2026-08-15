"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import Image from "next/image";

type ChatMessage = {
  id: number;
  role: "assistant" | "user";
  text: string;
  file?: string;
  report?: Record<string, unknown>;
};

const progressSteps = [
  "Reading column types and dataset shape",
  "Checking missing values and duplicates",
  "Calculating descriptive statistics",
  "Examining correlations and outliers",
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
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [working, setWorking] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 1,
      role: "assistant",
      text: "Hi — I’m Obi, your EDA assistant. Upload a CSV and I’ll inspect its quality, distributions, relationships, and unusual values, then package the findings as a validated JSON report.",
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

  function acceptFile(candidate?: File) {
    setError("");
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith(".csv")) {
      setError("Please choose a .csv file.");
      return;
    }
    if (candidate.size > 15 * 1024 * 1024) {
      setError("Please choose a CSV smaller than 15 MB.");
      return;
    }
    setFile(candidate);
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  async function analyze() {
    if (!file || working) return;
    const selected = file;
    setMessages((current) => [
      ...current,
      { id: Date.now(), role: "user", text: "Analyze this dataset", file: selected.name },
    ]);
    setFile(null);
    setError("");
    setProgress(0);
    setWorking(true);

    try {
      const form = new FormData();
      form.append("file", selected);
      const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
      const response = await fetch(`${apiBase}/analyze`, { method: "POST", body: form });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "The analysis could not be completed.");
      setProgress(progressSteps.length - 1);
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: "assistant",
          text: `Analysis complete. I found ${Array.isArray(body.findings) ? body.findings.length : 0} reportable findings and validated the output against the EDA schema.`,
          report: body,
        },
      ]);
    } catch (caught) {
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: "assistant",
          text: caught instanceof Error ? caught.message : "Something went wrong while analyzing the file.",
        },
      ]);
    } finally {
      setWorking(false);
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
            <div><div><strong>EDA AGENT</strong><small>● READY FOR INPUT</small></div></div>
           
          </div>

          <div className="messages" aria-live="polite">
            {messages.map((message) => (
              <div className={`message-wrap ${message.role}`} key={message.id}>
                {message.role === "assistant" && <span className="avatar">✦</span>}
                <div className="message">
                  <p>{message.text}</p>
                  {message.file && <div className="file-chip"><span>CSV</span><div><strong>{message.file}</strong><small>Ready for analysis</small></div></div>}
                  {message.report && (
                    <div className="report-card">
                      <div className="report-icon">{'{ }'}</div>
                      <div><strong>{reportFilename(message.report)}</strong><small>Validated JSON report</small></div>
                      <button onClick={() => download(message.report!)}>Download <span>↓</span></button>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {working && (
              <div className="message-wrap assistant thinking-wrap">
                <span className="avatar">✦</span>
                <div className="thinking-card">
                  <div className="thinking-title"><span className="pulse-ring" />Analyzing your dataset</div>
                  <p>{progressSteps[progress]}</p>
                  <div className="progress-track"><span style={{ width: `${18 + progress * 19}%` }} /></div>
                  <small>Running deterministic pandas checks</small>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="composer-area">
            {file && <div className="selected-file"><span>CSV</span><strong>{file.name}</strong><small>{(file.size / 1024).toFixed(1)} KB</small><button aria-label="Remove file" onClick={() => setFile(null)}>×</button></div>}
            <div
              className={`composer ${dragging ? "dragging" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <input ref={inputRef} type="file" accept=".csv,text/csv" onChange={onFileChange} hidden />
              <button className="attach-button" onClick={() => inputRef.current?.click()} aria-label="Choose CSV file">＋</button>
              <button className="upload-copy" onClick={() => inputRef.current?.click()}>
                <strong>{file ? "CSV ready to analyze" : "Upload a CSV to begin"}</strong>
                <small>{file ? "Send when you’re ready" : "Drag and drop or choose a file · max 15 MB"}</small>
              </button>
              <button className="send-button" disabled={!file || working} onClick={analyze} aria-label="Analyze CSV">↑</button>
            </div>
            {error && <p className="error-text">{error}</p>}
            <p className="composer-note">AI-generated interpretation can make mistakes. Verify important findings.</p>
          </div>
        </section>
      </section>
    </main>
  );
}
