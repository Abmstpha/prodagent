"""Agent tools. Every tool returns a string — never raises (production checklist)."""
import os

import requests

from .retriever import production_retrieve

_MEMORY: dict[str, str] = {}


def search_knowledge(query: str) -> str:
    """Search the local renewable-energy corpus (hybrid BM25+dense+RRF, reranked)."""
    try:
        results = production_retrieve(query, k_final=3)
        if not results:
            return "No results found in the corpus."
        return "\n---\n".join(results)
    except Exception as e:
        return f"search_knowledge error: {e}"


def recall_memory(topic: str) -> str:
    """Recall findings stored earlier in this session."""
    try:
        hits = {k: v for k, v in _MEMORY.items() if topic.lower() in k.lower()
                or topic.lower() in v.lower()}
        if not hits:
            return f"No stored finding matches '{topic}'."
        return "\n".join(f"- {k}: {v}" for k, v in hits.items())
    except Exception as e:
        return f"recall_memory error: {e}"


def store_finding(finding: str, source: str) -> str:
    """Store a finding with its source in session memory."""
    try:
        key = f"finding_{len(_MEMORY) + 1} ({source})"
        _MEMORY[key] = finding
        return f"Stored: {key}"
    except Exception as e:
        return f"store_finding error: {e}"


def web_search(query: str) -> str:
    """Web search via the Tavily API. Read-only; results are sanitised before use."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "web_search unavailable: TAVILY_API_KEY not set."
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": 3,
                  "search_depth": "basic"},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return "No web results."
        return "\n---\n".join(f"{res.get('title', '')} ({res.get('url', '')})\n"
                              f"{res.get('content', '')[:500]}" for res in results)
    except requests.Timeout:
        return "web_search error: request timed out after 15s."
    except Exception as e:
        return f"web_search error: {e}"
