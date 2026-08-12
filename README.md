# content_batch_graph

A LangGraph pipeline automating batch content review: translate/generate content,
run independent critic rounds, verify every finding against the real file before
trusting it, fix, validate with real programmatic checks, get human confirmation —
repeated per phase, with durable recurring patterns proposed (never written) back
for human approval at the end.

## What this is

A new project — not a port. It automates a manual protocol proven across real use:
translate a source file into several target languages, run two **independent**
critic passes over each (not retries — deliberately fresh reads, since a fix in
round 1 can introduce a new problem round 2 needs to catch), verify every finding
against the actual file before acting on it, fix, re-validate with real tools
(JSON validity, corpus-wide structural validators) — not just LLM judgment — and
confirm with a human at defined checkpoints before proceeding.

It also reuses concepts from a second prior system
(`GEP_Genome-Evolution-Protocol`): a persona-driven prompting approach and an
evolving pattern memory (a recurring finding becomes a durable rule, but only ever
on explicit human confirmation — never written silently).

Both are reimplemented here on LangGraph/LangChain with this project's own code and
its own tests — nothing is imported live from either prior system.

## Structure

```
content_batch_graph/
├── pyproject.toml
├── uv.lock
├── src/
│   └── content_batch_graph/
│       ├── domain/      ← the actual work: prompts/roles, memory, provider routing, persistence, validators
│       ├── nodes/       ← one file per graph step — reads state, calls domain/, returns an update
│       ├── state.py     ← the shared graph state schema (not yet created)
│       ├── graph.py     ← StateGraph wiring (not yet created)
│       └── cli.py       ← entrypoint (not yet created)
└── tests/
```

## Tooling

Managed with [`uv`](https://astral.sh/uv) — matches the official LangGraph project
template's tooling. `uv sync` installs from the committed `uv.lock`.

## Architecture

Layering rules, quality gates, and architect stop-points for this project are defined
in `.claude/skills/langgraph-coding-agent/SKILL.md` — load it before writing or editing
any node, state field, domain module, or graph wiring.

## Status

Scaffolding + dependencies only. No domain logic, nodes, or graph wiring implemented yet.
