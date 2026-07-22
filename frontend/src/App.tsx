import { useState } from "react";

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

export default function App() {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);

  async function ask() {
    const q = question.trim();
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
        <h1>⚡ prodagent</h1>
        <p className="subtitle">Renewable-energy research agent — solar · wind · hydro</p>
        <p className="disclosure">
          You are interacting with an AI research agent (EU AI Act limited-risk
          transparency notice).
        </p>
      </header>

      <section className="chat">
        {turns.map((turn, i) => (
          <div key={i} className="turn">
            <div className="bubble user">{turn.question}</div>
            {turn.error && <div className="bubble error">{turn.error}</div>}
            {turn.response && (
              <div className={`bubble agent ${turn.response.blocked ? "blocked" : ""}`}>
                <pre>{turn.response.answer}</pre>
                {!turn.response.blocked && (
                  <div className="meta">
                    <span className={`verdict ${turn.response.critic.verdict.toLowerCase()}`}>
                      critic: {turn.response.critic.verdict}
                    </span>
                    <span>confidence: {turn.response.confidence}</span>
                    <span>
                      agreement: {turn.response.agreement}/{turn.response.k}
                    </span>
                    <span>${turn.response.cost_usd.toFixed(4)}</span>
                    <span>{turn.response.duration_s.toFixed(1)}s</span>
                  </div>
                )}
              </div>
            )}
            {!turn.response && !turn.error && i === turns.length - 1 && (
              <div className="bubble agent thinking">searching · reasoning · critic review…</div>
            )}
          </div>
        ))}
      </section>

      <footer>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="e.g. Which renewable source is cheapest per kWh?"
          disabled={loading}
        />
        <button onClick={ask} disabled={loading || !question.trim()}>
          {loading ? "…" : "Ask"}
        </button>
      </footer>
    </main>
  );
}
