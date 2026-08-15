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

const progressSteps = [  // just a visual progress indicator
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
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [working, setWorking] = useState(false);
  const [progress, setProgress] = useState(0);
  const [activeFileName, setActiveFileName] = useState("");
  const [queuePosition, setQueuePosition] = useState({ current: 0, total: 0 });
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
          setProgress(progressSteps.length - 1);
          setMessages((current) => [
            ...current,
            {
              id: Date.now() + index * 10 + 1,
              role: "assistant",
              text: `Analysis complete for ${selected.name}. I found ${Array.isArray(body.findings) ? body.findings.length : 0} reportable findings and validated the output against the EDA schema.`,
              report: body,
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
