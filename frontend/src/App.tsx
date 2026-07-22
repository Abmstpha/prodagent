import { useEffect, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface AskResponse {
  answer: string;
  blocked: boolean;
  critic: { verdict: string; reason: string };
  confidence: string;
  agreement: number;
  k: number;
  cost_usd: number;
  duration_s: number;
  disclosure: string;
}

interface Turn {
  question: string;
  response?: AskResponse;
  error?: string;
}

const STAGES = [
  { icon: "🛡", label: "L1 guardrails — screening the question" },
  { icon: "🔎", label: "Gathering evidence — corpus (Pinecone + BM25) & web" },
  { icon: "🧠", label: "Reasoning — 3 independent chains, majority vote" },
  { icon: "⚖️", label: "Critic agent — verifying every claim" },
];

const SAMPLES = [
  "Which renewable source is cheapest per kWh?",
  "Compare capacity factors of solar, wind and hydro",
  "What share of global electricity does hydropower generate?",
  "What is the largest power station in the world?",
];

function parseSections(answer: string) {
  const grab = (name: string, next: string[]) => {
    const re = new RegExp(
      `\\**${name}[\\s:*]*\\n?([\\s\\S]*?)(?=\\n\\s*\\**(?:${next.join("|")})[\\s:*]|$)`,
      "i",
    );
    return answer.match(re)?.[1]?.trim() ?? "";
  };
  const evidence = grab("EVIDENCE", ["ANALYSIS", "CONCLUSION", "CONFIDENCE"]);
  const analysis = grab("ANALYSIS", ["CONCLUSION", "CONFIDENCE"]);
  const conclusion = grab("CONCLUSION", ["CONFIDENCE"]);
  const confidence = grab("CONFIDENCE", []);
  const parsed = evidence || analysis || conclusion;
  return parsed ? { evidence, analysis, conclusion, confidence } : null;
}

function Stages() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const t = setInterval(
      () => setStep((s) => Math.min(s + 1, STAGES.length - 1)),
      14000,
    );
    return () => clearInterval(t);
  }, []);
  return (
    <div className="stages">
      {STAGES.map((s, i) => (
        <div key={i} className={`stage ${i < step ? "done" : i === step ? "active" : ""}`}>
          <span className="stage-icon">{s.icon}</span>
          <span>{s.label}</span>
          {i === step && <span className="pulse" />}
          {i < step && <span className="check">✓</span>}
        </div>
      ))}
    </div>
  );
}

function AnswerCard({ r }: { r: AskResponse }) {
  const sections = parseSections(r.answer);
  return (
    <div className={`bubble agent ${r.blocked ? "blocked" : ""}`}>
      {r.blocked || !sections ? (
        <pre>{r.answer}</pre>
      ) : (
        <div className="sections">
          {sections.evidence && (
            <section>
              <h4>📚 Evidence</h4>
              <pre>{sections.evidence}</pre>
            </section>
          )}
          {sections.analysis && (
            <section>
              <h4>🧩 Analysis</h4>
              <pre>{sections.analysis}</pre>
            </section>
          )}
          {sections.conclusion && (
            <section className="conclusion">
              <h4>✅ Conclusion</h4>
              <pre>{sections.conclusion}</pre>
            </section>
          )}
          {sections.confidence && (
            <section>
              <h4>🎯 Confidence</h4>
              <pre>{sections.confidence}</pre>
            </section>
          )}
        </div>
      )}
      {!r.blocked && (
        <div className="meta">
          <span className={`verdict ${r.critic.verdict.toLowerCase()}`}>
            ⚖️ critic: {r.critic.verdict}
          </span>
          <span>🗳 {r.agreement}/{r.k} voices agree</span>
          <span>💲{r.cost_usd.toFixed(4)}</span>
          <span>⏱ {r.duration_s.toFixed(1)}s</span>
        </div>
      )}
      {!r.blocked && r.critic.reason && (
        <div className="critic-reason">“{r.critic.reason}”</div>
      )}
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const chatEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, loading]);

  async function ask(q0?: string) {
    const q = (q0 ?? question).trim();
    if (!q || loading) return;
    setQuestion("");
    setLoading(true);
    setTurns((t) => [...t, { question: q }]);
    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(detail.detail ?? `HTTP ${res.status}`);
      }
      const data: AskResponse = await res.json();
      setTurns((t) =>
        t.map((turn, i) => (i === t.length - 1 ? { ...turn, response: data } : turn)),
      );
    } catch (e) {
      setTurns((t) =>
        t.map((turn, i) =>
          i === t.length - 1 ? { ...turn, error: String(e) } : turn,
        ),
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <header>
        <div className="brand">
          <span className="logo">⚡</span>
          <div>
            <h1>WattWise</h1>
            <p className="subtitle">
              An AI research agent for renewable energy — every answer grounded in
              sources, voted over 3 reasoning chains and reviewed by a critic agent.
            </p>
          </div>
        </div>
        <div className="pills">
          <span className="pill">Hybrid RAG · Pinecone + BM25</span>
          <span className="pill">Self-Consistency k=3</span>
          <span className="pill">Generator–Critic</span>
          <span className="pill">L1/L4 guardrails</span>
          <span className="pill">$2 budget cap</span>
        </div>
      </header>

      <section className="how">
        <h3>How a question flows</h3>
        <div className="steps">
          <div className="step">
            <span className="step-icon">🛡</span>
            <b>1 · Guardrails</b>
            <p>Unicode normalisation + injection patterns screen the input; risky tool actions are gated.</p>
          </div>
          <div className="step">
            <span className="step-icon">🔎</span>
            <b>2 · Evidence</b>
            <p>The agent searches a curated corpus (solar · wind · hydro) via Pinecone + BM25, and the web when needed.</p>
          </div>
          <div className="step">
            <span className="step-icon">🧠</span>
            <b>3 · Reasoning</b>
            <p>Three independent EVIDENCE → ANALYSIS → CONCLUSION chains; the majority answer wins.</p>
          </div>
          <div className="step">
            <span className="step-icon">⚖️</span>
            <b>4 · Critic</b>
            <p>A second agent verifies every claim against the sources before you see the answer.</p>
          </div>
        </div>
      </section>

      <section className="chat">
        {turns.length === 0 && (
          <div className="samples">
            <p>Try one:</p>
            {SAMPLES.map((s) => (
              <button key={s} className="chip" onClick={() => ask(s)}>
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((turn, i) => (
          <div key={i} className="turn">
            <div className="bubble user">{turn.question}</div>
            {turn.error && <div className="bubble error">{turn.error}</div>}
            {turn.response && <AnswerCard r={turn.response} />}
            {!turn.response && !turn.error && i === turns.length - 1 && <Stages />}
          </div>
        ))}
        <div ref={chatEnd} />
      </section>

      <footer>
        <div className="ask-row">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
            placeholder="Ask about solar, wind or hydro…"
            disabled={loading}
          />
          <button onClick={() => ask()} disabled={loading || !question.trim()}>
            {loading ? "…" : "Ask"}
          </button>
        </div>
        <p className="disclosure">
          ⚠️ You are interacting with an AI system (EU AI Act limited-risk transparency
          notice) · <a href="https://github.com/Abmstpha/agenticapp">GitHub</a> ·{" "}
          <a href={`${API_URL}/health`}>API health</a> ·{" "}
          <a href={`${API_URL}/metrics`}>metrics</a>
        </p>
      </footer>
    </main>
  );
}
