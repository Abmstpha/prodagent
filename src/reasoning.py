"""Reasoning stack (Lab 3): few-shot CoT + Self-Consistency k=3 on the final synthesis."""
import re
from collections import Counter

from .tracing import traceable

SYSTEM_SYNTHESIS = """You are a research synthesis agent for renewable-energy questions.

Performance measure:
- ACCURACY: every claim must trace to a source in the provided context.
- COMPLETENESS: answer all aspects of the question.
- CONFIDENCE: express uncertainty when evidence is insufficient.

Failure modes to avoid:
- Empty answer or "I don't know" when sources exist in the context.
- Citing a fact that is not in the context.
- HIGH confidence supported by a single source.

Always use this format:

EVIDENCE:
  - [fact 1 with source]
  - [fact 2 with source]

ANALYSIS:
  Step 1: [first reasoning step]
  Step 2: [second reasoning step]
  Step 3: [reconcile any contradictions]

CONCLUSION: [your answer]
CONFIDENCE: HIGH / MEDIUM / LOW — [one-sentence justification]

Example:
Question: Which renewable source is cheapest?
EVIDENCE:
  - wind.md: onshore wind LCOE $0.033/kWh in 2023, lowest of all sources
  - solar.md: utility-scale solar LCOE $0.044/kWh in 2023
ANALYSIS:
  Step 1: Both sources give 2023 weighted-average LCOE figures
  Step 2: $0.033 (onshore wind) < $0.044 (solar) < $0.057 (hydro)
  Step 3: No contradicting source in the context
CONCLUSION: Onshore wind is the cheapest, at about $0.033 per kWh
CONFIDENCE: HIGH — two independent sources give consistent, dated figures

If the context does not contain the information needed, say so explicitly:
CONCLUSION: Insufficient data in the corpus to answer.
CONFIDENCE: LOW — no supporting source in the context.
"""

_WORD = re.compile(r"[a-z0-9]{4,}")


def extract_conclusion(text: str) -> str:
    m = re.search(r"CONCLUSION\s*:?\**\s*(.+?)(?:\n\**CONFIDENCE|$)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text[-200:]


def extract_confidence(text: str) -> str:
    m = re.search(r"CONFIDENCE[\s:*]*?(HIGH|MEDIUM|LOW)", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def _similar(a: str, b: str, threshold: float = 0.35) -> bool:
    """Token-overlap similarity — Lab 3 showed that raw wording comparison undercounts
    agreement, so conclusions are clustered by Jaccard overlap instead."""
    ta, tb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def majority_vote(conclusions: list) -> tuple:
    """Cluster near-identical conclusions, return (winner_index, cluster_size)."""
    clusters = []   # list of lists of indices
    for i, c in enumerate(conclusions):
        for cluster in clusters:
            if _similar(conclusions[cluster[0]], c):
                cluster.append(i)
                break
        else:
            clusters.append([i])
    best = max(clusters, key=len)
    return best[0], len(best)


@traceable(name="self_consistency_synthesis", run_type="chain")
def self_consistent_answer(client, question: str, context: str, k: int = 3,
                           budget=None, scripts=None) -> dict:
    """k independent reasoning chains at T=0.8, majority vote on the conclusion.

    Reserved for the final synthesis — k× token cost.
    """
    answers = []
    for i in range(k):
        c = client if scripts is None else scripts[i]
        voice = traceable(name=f"reasoning_voice_{i + 1}", run_type="llm")(c.complete)
        reply = voice(
            [{"role": "system", "content": SYSTEM_SYNTHESIS},
             {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}],
            temperature=0.8, max_tokens=1024,
        )
        text = reply.content or ""
        if getattr(reply, "finish_reason", None) == "length":   # truncated: retry once, larger
            reply = c.complete(
                [{"role": "system", "content": SYSTEM_SYNTHESIS},
                 {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}],
                temperature=0.8, max_tokens=2048,
            )
            text = reply.content or text
        if budget is not None and reply.usage:
            budget.record(getattr(c, "model", "mistral-large-latest"),
                          reply.usage.get("input_tokens", 0),
                          reply.usage.get("output_tokens", 0))
        answers.append({"full": text, "conclusion": extract_conclusion(text),
                        "confidence": extract_confidence(text)})

    winner, agreement = majority_vote([a["conclusion"] for a in answers])
    return {
        "answer": answers[winner]["full"],
        "conclusion": answers[winner]["conclusion"],
        "confidence": answers[winner]["confidence"],
        "agreement": agreement,
        "k": k,
        "all": answers,
    }
