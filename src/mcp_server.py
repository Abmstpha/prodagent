"""mcp_server.py — 3-tool MCP server for WattWise, the renewable-energy research agent.

Run standalone:  python -m src.mcp_server
Inspect:         npx @modelcontextprotocol/inspector python -m src.mcp_server
"""
from mcp.server.fastmcp import FastMCP

try:                                    # package mode (backend, tests)
    from .retriever import production_retrieve
except ImportError:                     # standalone mode (inspector, stdio client)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.retriever import production_retrieve

mcp = FastMCP("wattwise-tools")

_MEMORY: dict[str, str] = {}


@mcp.tool()
def search_corpus(query: str) -> str:
    """Search the renewable-energy corpus (solar, wind, hydro documents).

    Uses the production retrieval pipeline: hybrid BM25 + dense + RRF fusion over
    child chunks, mapped to parent chunks, reranked by a cross-encoder score.
    Returns the 3 most relevant passages separated by '---'.
    Use for any factual question about solar, wind or hydro power.
    Do NOT use for topics outside renewable energy — use nothing instead.

    Args:
        query: natural-language question or keywords, e.g. "offshore wind capacity factor".
    """
    try:
        results = production_retrieve(query, k_final=3)
        if not results:
            return "No results found in the corpus."
        return "\n---\n".join(results)
    except Exception as e:
        return f"search_corpus error: {e}"


@mcp.tool()
def store_finding(finding: str, source: str) -> str:
    """Store a research finding with its source for later recall in this session.

    Use after search_corpus returns a fact worth keeping, so recall_memory can
    answer repeat questions without another search.

    Args:
        finding: one factual sentence, e.g. "Onshore wind LCOE was $0.033/kWh in 2023".
        source: where it came from, e.g. "wind.md".
    """
    try:
        key = f"finding_{len(_MEMORY) + 1} ({source})"
        _MEMORY[key] = finding
        return f"Stored: {key}"
    except Exception as e:
        return f"store_finding error: {e}"


@mcp.tool()
def recall_memory(topic: str) -> str:
    """Recall findings stored earlier in this session with store_finding.

    Use BEFORE searching again: if the finding is already stored, this is free.
    Do NOT use for maths or for topics never stored in this session.

    Args:
        topic: keyword to match against stored findings, e.g. "LCOE".
    """
    try:
        hits = {k: v for k, v in _MEMORY.items()
                if topic.lower() in k.lower() or topic.lower() in v.lower()}
        if not hits:
            return f"No stored finding matches '{topic}'."
        return "\n".join(f"- {k}: {v}" for k, v in hits.items())
    except Exception as e:
        return f"recall_memory error: {e}"


if __name__ == "__main__":
    mcp.run()
