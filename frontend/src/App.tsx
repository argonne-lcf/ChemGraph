import { useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine,
  ArrowUp,
  Atom,
  Check,
  ChevronRight,
  FlaskConical,
  LoaderCircle,
  Menu,
  MessageSquare,
  Paperclip,
  Plus,
  SlidersHorizontal,
  Sparkles,
  X,
} from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import {
  api,
  ApiError,
  post,
  terminal,
  type Artifact,
  type Capabilities,
  type Run,
  type Session,
  type SessionDetail,
} from "./api";
import MoleculeViewer from "./MoleculeViewer";

const workflowLabel = (value: string) =>
  value === "multi_agent" ? "Multi-agent" : "Single agent";
const statusLabel: Record<string, string> = {
  queued: "Queued",
  running: "Working",
  waiting_for_input: "Needs your response",
  completed: "Complete",
  failed: "Failed",
  interrupted: "Interrupted",
};
type Submission = {
  sessionId: string;
  body: { query: string; attachments: string[]; request_id: string };
};
type Progress = { id: string; kind: string; tool?: string };

function RichText({ children }: { children: string }) {
  return (
    <div className="markdown">
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {children}
      </Markdown>
    </div>
  );
}

export default function App() {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selected, setSelected] = useState<string | null>(() =>
    new URLSearchParams(window.location.search).get("session"),
  );
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [model, setModel] = useState("");
  const [workflow, setWorkflow] = useState("single_agent");
  const [draft, setDraft] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const [progress, setProgress] = useState<Progress[]>([]);
  const [connected, setConnected] = useState(true);
  const [sidebar, setSidebar] = useState(false);
  const [resultsOpen, setResultsOpen] = useState(() => window.innerWidth > 950);
  const [artifactId, setArtifactId] = useState<string | null>(null);
  const [answered, setAnswered] = useState<string | null>(null);
  const [pending, setPending] = useState<Submission | null>(null);
  const uploadInput = useRef<HTMLInputElement>(null);
  const endOfChat = useRef<HTMLDivElement>(null);
  const activeRun = detail?.runs.find((run) => !terminal(run.status));
  const waiting = activeRun?.status === "waiting_for_input";
  const artifacts = detail?.runs.flatMap((run) => run.artifacts) || [];
  const structures = artifacts.filter((artifact) => artifact.preview_url);
  const currentArtifact =
    structures.find((artifact) => artifact.id === artifactId) ||
    structures[structures.length - 1];

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      api<Capabilities>("/capabilities", { signal: controller.signal }),
      api<Session[]>("/sessions", { signal: controller.signal }),
    ])
      .then(([caps, list]) => {
        setCapabilities(caps);
        setModel(caps.models[0] || "");
        setSessions(list);
      })
      .catch((error) => {
        if (!controller.signal.aborted) setError(error.message);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const url = new URL(window.location.href);
    if (selected) url.searchParams.set("session", selected);
    else url.searchParams.delete("session");
    window.history.replaceState(null, "", url);
    if (!selected) {
      setDetail(null);
      return;
    }
    const controller = new AbortController();
    api<SessionDetail>(`/sessions/${selected}`, { signal: controller.signal })
      .then(setDetail)
      .catch((error) => {
        if (!controller.signal.aborted) setError(error.message);
      });
    return () => controller.abort();
  }, [selected, revision]);

  useEffect(() => {
    if (!activeRun) return;
    setProgress([]);
    setConnected(true);
    const stream = new EventSource(`/api/v1/runs/${activeRun.id}/events`);
    stream.onopen = () => setConnected(true);
    stream.onerror = () => {
      setConnected(false);
      setRevision((value) => value + 1);
    };
    stream.addEventListener("progress", (event) => {
      const message = event as MessageEvent;
      const data = JSON.parse(message.data);
      setProgress((previous) =>
        previous.some((item) => item.id === message.lastEventId)
          ? previous
          : [...previous.slice(-99), { id: message.lastEventId, ...data }],
      );
    });
    stream.addEventListener("status", () => setRevision((value) => value + 1));
    stream.addEventListener("human_response", () =>
      setRevision((value) => value + 1),
    );
    stream.addEventListener("end", () => {
      stream.close();
      setRevision((value) => value + 1);
      void api<Session[]>("/sessions")
        .then(setSessions)
        .catch((error) => setError(error.message));
    });
    return () => stream.close();
  }, [activeRun?.id]);

  useEffect(() => {
    endOfChat.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [detail?.runs.length, activeRun?.status]);

  function chooseSession(id: string | null) {
    if (busy || pending) return;
    setSelected(id);
    setDetail(null);
    setDraft("");
    setFiles([]);
    setError("");
    setArtifactId(null);
    setSidebar(false);
    setAnswered(null);
  }

  async function send(event: React.FormEvent) {
    event.preventDefault();
    if (busy || (!draft.trim() && !pending) || (!model && !selected)) return;
    setBusy(true);
    setError("");
    try {
      if (waiting && activeRun?.question_id) {
        await api(
          `/runs/${activeRun.id}/response`,
          post({ question_id: activeRun.question_id, answer: draft }),
        );
        setAnswered(activeRun.question_id);
        setDraft("");
        setRevision((value) => value + 1);
        return;
      }
      let submission = pending;
      if (!submission) {
        let sessionId = selected;
        if (!sessionId) {
          const session = await api<Session>(
            "/sessions",
            post({ model, workflow }),
          );
          sessionId = session.id;
          setSelected(sessionId);
          setDetail({ ...session, runs: [] });
        }
        const attachments: string[] = [];
        for (const file of files) {
          if (file.size > (capabilities?.upload_limit || 0))
            throw new Error(`${file.name} exceeds the upload limit.`);
          const artifact = await api<Artifact>(
            `/sessions/${sessionId}/uploads?filename=${encodeURIComponent(file.name)}`,
            {
              method: "POST",
              headers: { "Content-Type": "application/octet-stream" },
              body: file,
            },
          );
          attachments.push(artifact.id);
        }
        submission = {
          sessionId,
          body: { query: draft, attachments, request_id: crypto.randomUUID() },
        };
        setPending(submission);
      }
      await api<Run>(
        `/sessions/${submission.sessionId}/runs`,
        post(submission.body),
      );
      setPending(null);
      setDraft("");
      setFiles([]);
      setRevision((value) => value + 1);
      setSessions(await api<Session[]>("/sessions"));
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Unable to send the request.",
      );
      if (
        error instanceof ApiError &&
        error.status >= 400 &&
        error.status < 500
      ) {
        setPending(null);
        setRevision((value) => value + 1);
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={`workspace ${resultsOpen ? "" : "results-hidden"}`}>
      <aside className={`sidebar ${sidebar ? "open" : ""}`}>
        <a
          className="brand"
          href="/"
          onClick={(event) => {
            event.preventDefault();
            chooseSession(null);
          }}
        >
          <span className="brand-icon">
            <Atom size={24} />
          </span>
          <span>
            ChemGraph<small>CHEMISTRY WORKSPACE</small>
          </span>
        </a>
        <button
          className="new-chat"
          disabled={busy || !!pending}
          onClick={() => chooseSession(null)}
        >
          <Plus size={18} /> New conversation
        </button>
        <div className="section-label">
          YOUR CONVERSATIONS <span>{sessions.length}</span>
        </div>
        <nav className="session-list" aria-label="Conversations">
          {sessions.map((session) => (
            <button
              key={session.id}
              className={`session ${selected === session.id ? "selected" : ""}`}
              disabled={busy || !!pending}
              onClick={() => chooseSession(session.id)}
            >
              <MessageSquare size={16} />
              <span>{session.title}</span>
            </button>
          ))}
          {!sessions.length && (
            <p className="quiet sidebar-empty">
              Your conversations will appear here.
            </p>
          )}
        </nav>
        <div className="sidebar-bottom">
          <div className="availability">
            <span className="dot" />
            {capabilities?.demo
              ? "Local demo workspace"
              : "Institution workspace"}
          </div>
          <div className="profile">
            <span className="avatar">
              {capabilities?.user.slice(0, 2).toUpperCase() || "CG"}
            </span>
            <span>
              {capabilities?.user || "Connecting…"}
              <small>
                {capabilities?.calculators.join(", ").toUpperCase() ||
                  "Computational chemistry"}
              </small>
            </span>
          </div>
        </div>
      </aside>
      {sidebar && (
        <button
          className="sidebar-scrim"
          aria-label="Close conversations"
          onClick={() => setSidebar(false)}
        />
      )}
      <main className="main-panel">
        <header className="workspace-header">
          <button
            className="icon-button mobile-menu"
            aria-label="Open conversations"
            onClick={() => setSidebar(true)}
          >
            <Menu size={20} />
          </button>
          <div className="breadcrumb">
            Workspace <ChevronRight size={14} />
            <strong>{detail?.title || "New conversation"}</strong>
          </div>
          <button
            className={`results-toggle ${resultsOpen ? "active" : ""}`}
            aria-expanded={resultsOpen}
            onClick={() => setResultsOpen(!resultsOpen)}
          >
            <FlaskConical size={16} />
            Results
          </button>
        </header>
        <div className="chat-settings">
          <div>
            <SlidersHorizontal size={15} />
            <span>Run settings</span>
          </div>
          <label>
            Model
            <select
              aria-label="Model"
              value={detail?.model || model}
              disabled={!!selected || busy}
              onChange={(event) => setModel(event.target.value)}
            >
              {(capabilities?.models || []).map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
              {detail && !capabilities?.models.includes(detail.model) && (
                <option value={detail.model}>{detail.model}</option>
              )}
            </select>
          </label>
          <label>
            Workflow
            <select
              aria-label="Workflow"
              value={detail?.workflow || workflow}
              disabled={!!selected || busy}
              onChange={(event) => setWorkflow(event.target.value)}
            >
              {(capabilities?.workflows || ["single_agent", "multi_agent"]).map(
                (value) => (
                  <option key={value} value={value}>
                    {workflowLabel(value)}
                  </option>
                ),
              )}
            </select>
          </label>
        </div>
        <div className="chat-scroll">
          {!detail?.runs.length && (
            <section className="welcome">
              <div className="welcome-mark">
                <FlaskConical size={32} strokeWidth={1.4} />
              </div>
              <div className="eyebrow">FROM QUESTION TO CALCULATION</div>
              <h1>
                Explore your next
                <br />
                <span>molecular question.</span>
              </h1>
              <p>
                Build structures, run simulations, and explore the results.
                <br />
                Start with a question or attach a structure.
              </p>
              <div className="suggestions">
                {[
                  [
                    "Optimize a structure",
                    capabilities?.demo
                      ? "Optimize a copper dimer with EMT"
                      : "Build a water molecule from SMILES O and optimize its geometry.",
                  ],
                  [
                    "Compare molecules",
                    "Compare the optimized structures of water and methanol.",
                  ],
                  [
                    "Explore an attachment",
                    "Describe the attached structure and suggest a suitable calculation.",
                  ],
                ].map(([title, query]) => (
                  <button key={title} onClick={() => setDraft(query)}>
                    <Sparkles size={16} />
                    <span>{title}</span>
                    <ChevronRight size={15} />
                  </button>
                ))}
              </div>
            </section>
          )}
          {detail?.runs.map((run) => (
            <article className="exchange" key={run.id}>
              <div className="user-message">
                <div className="message-label">YOU</div>
                <p>{run.query}</p>
                {run.attachments.length > 0 && (
                  <div className="attachment-chips">
                    {run.attachments.map((file) => (
                      <a key={file.id} href={file.url}>
                        <Paperclip size={13} />
                        {file.name}
                      </a>
                    ))}
                  </div>
                )}
              </div>
              <div className="assistant-message">
                <span className="assistant-avatar">
                  <Atom size={20} />
                </span>
                <div className="assistant-body">
                  <div className="assistant-heading">
                    <strong>ChemGraph</strong>
                    <span className={`status-badge ${run.status}`}>
                      {!terminal(run.status) ? (
                        <LoaderCircle className="spin" size={12} />
                      ) : run.status === "completed" ? (
                        <Check size={12} />
                      ) : null}
                      {statusLabel[run.status]}
                    </span>
                  </div>
                  {run.responses.map((response, index) => (
                    <div className="follow-up-history" key={index}>
                      <RichText>{response.question}</RichText>
                      <p>
                        <strong>Your response:</strong> {response.answer}
                      </p>
                    </div>
                  ))}
                  {run.final_text && <RichText>{run.final_text}</RichText>}
                  {run.id === activeRun?.id && (
                    <div className="run-progress" aria-live="polite">
                      {progress
                        .filter((item) => item.tool)
                        .slice(-6)
                        .map((item) => (
                          <div key={item.id}>
                            {item.kind.endsWith("finished") ? (
                              <Check size={13} />
                            ) : (
                              <span className="progress-dot" />
                            )}
                            <span>
                              {item.tool?.replaceAll("_", " ")}
                              {item.kind.endsWith("failed") ? " — failed" : ""}
                            </span>
                          </div>
                        ))}
                      {!progress.length && (
                        <p>
                          {run.status === "queued"
                            ? "Your calculation is queued."
                            : "Preparing the calculation…"}
                        </p>
                      )}
                      {!connected && (
                        <p>
                          Connection interrupted. Reconnecting to your
                          calculation…
                        </p>
                      )}
                    </div>
                  )}
                  {run.question && (
                    <div className="question">
                      <strong>A quick clarification</strong>
                      <RichText>{run.question}</RichText>
                      <small>
                        {answered === run.question_id
                          ? "Response sent. Waiting for the agent…"
                          : "Reply in the message box below to continue."}
                      </small>
                    </div>
                  )}
                  {run.error && (
                    <div className="run-error">
                      <p>{run.error}</p>
                      <small>Run {run.id}</small>
                      <button
                        onClick={() => {
                          setDraft(run.query);
                          setError("");
                        }}
                      >
                        Use this question again
                      </button>
                    </div>
                  )}
                  {!!run.artifacts.length && (
                    <button
                      className="inline-results"
                      onClick={() => {
                        setResultsOpen(true);
                        setArtifactId(
                          run.artifacts.find((a) => a.preview_url)?.id || null,
                        );
                      }}
                    >
                      <FlaskConical size={14} />
                      {run.artifacts.length} result files
                      <ChevronRight size={14} />
                    </button>
                  )}
                </div>
              </div>
            </article>
          ))}
          <div ref={endOfChat} />
        </div>
        <div className="composer-area">
          {error && (
            <div role="alert" className="error-banner">
              <span>{error}</span>
              <button
                className="icon-button"
                aria-label="Dismiss error"
                onClick={() => setError("")}
              >
                <X size={15} />
              </button>
            </div>
          )}
          {capabilities && !capabilities.models.length && (
            <div className="error-banner">
              No model providers are configured. Ask your administrator to
              enable a provider.
            </div>
          )}
          {selected && (
            <div className="settings-note">
              Model and workflow stay with this conversation. Start a new
              conversation to change them.
            </div>
          )}
          <form
            className={`composer ${waiting ? "needs-response" : ""}`}
            onSubmit={send}
          >
            {!!files.length && (
              <div className="attachment-chips">
                {files.map((file, index) => (
                  <span key={`${file.name}-${index}`}>
                    <Paperclip size={13} />
                    {file.name}
                    <button
                      type="button"
                      disabled={busy || !!pending}
                      aria-label={`Remove ${file.name}`}
                      onClick={() =>
                        setFiles(files.filter((_, i) => i !== index))
                      }
                    >
                      <X size={12} />
                    </button>
                  </span>
                ))}
              </div>
            )}
            <textarea
              aria-label={waiting ? "Response to agent" : "Message"}
              placeholder={
                waiting
                  ? "Reply to continue the calculation…"
                  : "Ask a chemistry question…"
              }
              value={draft}
              disabled={
                busy ||
                !!pending ||
                (!!activeRun && !waiting) ||
                answered === activeRun?.question_id
              }
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey &&
                  !event.nativeEvent.isComposing
                ) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              rows={2}
            />
            <div className="composer-toolbar">
              <div>
                <input
                  ref={uploadInput}
                  type="file"
                  multiple
                  accept=".xyz,.pdb,.cif,.traj,.json,.csv,.txt"
                  aria-label="Attach structure or data files"
                  hidden
                  onChange={(event) => {
                    const next = [
                      ...files,
                      ...Array.from(event.target.files || []),
                    ];
                    if (next.length > 10)
                      setError("Attach up to 10 files per message.");
                    else setFiles(next);
                    event.target.value = "";
                  }}
                />
                <button
                  type="button"
                  className="attach-button"
                  disabled={busy || !!activeRun || !!pending}
                  onClick={() => uploadInput.current?.click()}
                >
                  <Paperclip size={17} />
                  <span>Attach files</span>
                </button>
              </div>
              <button
                className="send-button"
                type="submit"
                aria-label={
                  pending
                    ? "Retry send"
                    : waiting
                      ? "Send response"
                      : "Send message"
                }
                disabled={
                  busy ||
                  (!draft.trim() && !pending) ||
                  !capabilities?.models.length ||
                  (!!activeRun && !waiting && !pending) ||
                  answered === activeRun?.question_id
                }
              >
                {busy ? (
                  <LoaderCircle className="spin" size={18} />
                ) : pending ? (
                  "Retry send"
                ) : (
                  <ArrowUp size={20} />
                )}
              </button>
            </div>
          </form>
          <div className="composer-footnote">
            {capabilities?.demo
              ? "Demo mode · Real EMT calculation · No model credentials required"
              : "Review simulation settings and results before using them in your research."}
          </div>
        </div>
      </main>
      {resultsOpen && (
        <aside className="results-panel">
          <div className="results-heading">
            <div>
              <FlaskConical size={18} />
              <h2>Results</h2>
            </div>
            <button
              className="icon-button"
              aria-label="Close results"
              onClick={() => setResultsOpen(false)}
            >
              <X size={17} />
            </button>
          </div>
          {currentArtifact ? (
            <>
              <div className="result-label">MOLECULAR VIEWER</div>
              <select
                className="structure-select"
                aria-label="Structure or trajectory"
                value={currentArtifact.id}
                onChange={(event) => setArtifactId(event.target.value)}
              >
                {structures.map((artifact) => (
                  <option key={artifact.id} value={artifact.id}>
                    {artifact.name} · Run {artifact.run_id?.slice(0, 6)}
                  </option>
                ))}
              </select>
              <MoleculeViewer artifact={currentArtifact} />
            </>
          ) : (
            <div className="empty-results">
              <div className="empty-molecule">
                <Atom size={52} strokeWidth={1} />
              </div>
              <h3>A closer look, right here.</h3>
              <p>
                Molecular structures and trajectories will appear as your
                calculations finish.
              </p>
            </div>
          )}
          <div className="artifacts-heading">
            <span className="result-label">ARTIFACTS</span>
            <span>{artifacts.length}</span>
          </div>
          <div className="artifact-list">
            {artifacts.map((artifact) => (
              <a className="artifact" href={artifact.url} key={artifact.id}>
                <span className="file-icon">
                  {artifact.name.split(".").pop()?.toUpperCase()}
                </span>
                <span>
                  <strong>{artifact.name}</strong>
                  <small>
                    {Math.max(1, Math.round(artifact.size / 1024))} KB · Run{" "}
                    {artifact.run_id?.slice(0, 6)}
                  </small>
                </span>
                <ArrowDownToLine size={15} />
              </a>
            ))}
            {!artifacts.length && (
              <p className="quiet">
                Downloadable files are saved with each calculation.
              </p>
            )}
          </div>
          <div className="results-note">
            <Check size={14} />
            <span>Results stay attached to their conversation.</span>
          </div>
        </aside>
      )}
    </div>
  );
}
