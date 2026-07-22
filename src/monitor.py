"""Production monitoring (Lab 4): cost, latency, error rate per tool, alerts."""
from collections import defaultdict


class AgentMonitor:
    def __init__(self):
        self.n_runs = 0
        self.total_cost = 0.0
        self.alerts = []
        self.tools = defaultdict(lambda: {"calls": 0, "errors": 0, "ms_total": 0})

    def record_run(self, question, response, duration_s, cost_usd):
        self.n_runs += 1
        self.total_cost += cost_usd
        if duration_s > 60:
            self._alert(f"SLOW RUN: {duration_s:.1f}s")
        if cost_usd > 0.50:
            self._alert(f"EXPENSIVE RUN: ${cost_usd:.4f}")
        if not response or len(response) < 20:
            self._alert(f"EMPTY RESPONSE: {str(question)[:40]}")

    def record_tool(self, name, success, duration_ms):
        self.tools[name]["calls"] += 1
        self.tools[name]["ms_total"] += duration_ms
        if not success:
            self.tools[name]["errors"] += 1
            rate = self.tools[name]["errors"] / self.tools[name]["calls"]
            if rate > 0.20:
                self._alert(f"ERRORS {name}: {rate:.0%}")

    def _alert(self, msg):
        self.alerts.append(msg)
        print(f"⚠  {msg}")

    def snapshot(self) -> dict:
        return {
            "runs": self.n_runs,
            "total_cost_usd": round(self.total_cost, 4),
            "alerts": self.alerts[-20:],
            "tools": {
                name: {
                    "calls": s["calls"],
                    "error_rate": round(s["errors"] / s["calls"], 3) if s["calls"] else 0.0,
                    "avg_ms": round(s["ms_total"] / s["calls"]) if s["calls"] else 0,
                }
                for name, s in self.tools.items()
            },
        }


MONITOR = AgentMonitor()
