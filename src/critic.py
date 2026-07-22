"""Generator-Critic pattern: the critic produces a visible verdict before any answer is returned."""
import json
import re

from .tracing import traceable

CRITIC_SYSTEM = """You are a critic agent. You receive a QUESTION, the CONTEXT the
generator saw, and its ANSWER. Verify:
1. Every EVIDENCE line is supported by the CONTEXT (no invented facts).
2. The CONCLUSION follows from the EVIDENCE.
3. The CONFIDENCE level matches the quantity and independence of the evidence.

Reply with JSON only: {"verdict": "APPROVED" or "REVISE", "reason": "<one sentence>"}"""


@traceable(name="critic_review", run_type="chain")
def review(client, question: str, context: str, answer: str, budget=None) -> dict:
    """One critic pass. Returns {"verdict": ..., "reason": ...}."""
    reply = client.complete(
        [{"role": "system", "content": CRITIC_SYSTEM},
         {"role": "user", "content":
             f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"}],
        temperature=0.0, max_tokens=256,
    )
    if budget is not None and reply.usage:
        budget.record(getattr(client, "model", "mistral-large-latest"),
                      reply.usage.get("input_tokens", 0),
                      reply.usage.get("output_tokens", 0))
    raw = reply.content or ""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            verdict = str(data.get("verdict", "REVISE")).upper()
            if verdict not in ("APPROVED", "REVISE"):
                verdict = "REVISE"
            return {"verdict": verdict, "reason": str(data.get("reason", ""))[:300]}
        except json.JSONDecodeError:
            pass
    return {"verdict": "REVISE", "reason": "Critic output unparseable — do not trust the answer."}
