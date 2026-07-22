"""MCP server inspection: 3 tools listed, each callable over stdio — the 3/3 inspector tests."""
import asyncio
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent


async def _session_run(fn):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "src.mcp_server"], cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


def test_lists_exactly_three_tools():
    async def fn(session):
        tools = await session.list_tools()
        return sorted(t.name for t in tools.tools)
    names = asyncio.run(_session_run(fn))
    assert names == ["recall_memory", "search_corpus", "store_finding"]


def test_all_tools_have_docstrings():
    async def fn(session):
        tools = await session.list_tools()
        return [(t.name, t.description or "") for t in tools.tools]
    for name, desc in asyncio.run(_session_run(fn)):
        assert len(desc) > 50, f"{name} docstring too short for tool selection"


@pytest.mark.parametrize("tool,args,expect", [
    ("search_corpus", {"query": "largest power station"}, "Three Gorges"),
    ("store_finding", {"finding": "Onshore wind LCOE $0.033/kWh (2023)",
                       "source": "wind.md"}, "Stored"),
    ("recall_memory", {"topic": "anything-not-stored-in-this-fresh-process"}, "No stored finding"),
])
def test_tool_calls(tool, args, expect):
    async def fn(session):
        result = await session.call_tool(tool, args)
        return result.content[0].text
    out = asyncio.run(_session_run(fn))
    assert expect in out
