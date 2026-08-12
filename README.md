# reader_langgraph

A simulated native-speaker reader, orchestrated as a LangGraph pipeline.

## What this is

A new project — not a port. It reuses concepts proven in a prior system
(`GEP_Genome-Evolution-Protocol`): a reader-persona prompting approach, an evolving
pattern memory, an audited review trail, and provider-agnostic model routing. Those
concepts are reimplemented here on LangGraph/LangChain with their own code and their
own tests — nothing is imported live from the prior system.

## Structure

```
reader_langgraph/
├── pyproject.toml
├── src/
│   └── reader_langgraph/
│       ├── domain/     ← the actual work: prompts, memory, provider routing, persistence
│       ├── nodes/       ← one file per graph step — reads state, calls domain/, returns an update
│       ├── state.py     ← the shared graph state schema (not yet created)
│       ├── graph.py     ← StateGraph wiring (not yet created)
│       └── cli.py       ← entrypoint (not yet created)
└── tests/
```

## Architecture

Layering rules, quality gates, and architect stop-points for this project are defined
in `.claude/skills/langgraph-coding-agent/SKILL.md` — load it before writing or editing
any node, state field, or graph wiring.

## Status

Scaffolding only. No domain logic, nodes, or graph wiring implemented yet.
