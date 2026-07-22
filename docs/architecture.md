# Architecture

```
                       ┌──────────────────────── prodagent ────────────────────────┐
 user question ──▶ L1 input filter ──▶ agent loop (max 6 steps, Mistral large)      │
                   (NFKC + patterns)      │  tools via L4 gate + per-tool quotas    │
                                          │    search_knowledge ── hybrid BM25+TFIDF+RRF
                                          │                        └ parent-child → rerank
                                          │    recall_memory / store_finding (MONITOR)
                                          │    web_search (Tavily, sanitised)      │
                                          ▼                                        │
                        gathered context (every tool result sanitised)             │
                                          ▼                                        │
                        Self-Consistency k=3 synthesis (few-shot CoT:              │
                        EVIDENCE / ANALYSIS / CONCLUSION / CONFIDENCE)             │
                                          ▼                                        │
                        critic agent ── APPROVED ▶ response                        │
                              └─ REVISE ▶ one regeneration ▶ re-review             │
                                                                                   │
                   TokenBudget: hard $2.00 cap recorded after every LLM call       │
                   AgentMonitor: cost, latency, per-tool error rate, alerts        │
                   AGENT_VERSION: sha256 hash of both system prompts               │
└──────────────────────────────────────────────────────────────────────────────────┘

 backend/  FastAPI: POST /ask (rate-limited 10/min/IP) · GET /health · GET /metrics
           /mcp — MCP streamable-HTTP endpoint (API-key protected), same 3 tools
 frontend/ React TS chat UI: answer + critic verdict + confidence + agreement + cost
 src/mcp_server.py — stdio MCP server (search_corpus, store_finding, recall_memory)
```

## PEAS

| | |
|---|---|
| **Performance** | RAGAS faithfulness + answer_relevancy ≥ baseline on 5 eval questions; 5/5 injection tests blocked; run cost ≤ $2.00 (hard cap); excluded failure mode: empty answer when sources exist |
| **Environment** | Local corpus of 3 renewable-energy documents (solar, wind, hydro); Mistral API; Tavily web search; no external write APIs |
| **Actuators** | search_knowledge, recall_memory, store_finding, web_search; MCP server (stdio + HTTP) |
| **Sensors** | User question (L1-filtered); tool results (sanitised before entering context) |

## Non-obvious design decision

Self-Consistency votes on **meaning, not wording**: Lab 3 showed three equivalent
conclusions scoring 1/3 agreement when compared by keyword signature. `majority_vote`
clusters conclusions by token-overlap (Jaccard ≥ 0.35) before voting, so paraphrases
of the same answer count as agreement while a genuine dissent ("insufficient data")
still forms its own cluster.
