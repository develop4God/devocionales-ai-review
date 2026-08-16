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

## Batch CLI (`content-batch`)

Offline devotional generation for a whole year via a provider's batch API. This is
an operator tool that lives beside the graph, not inside it — a 365-day batch is a
multi-day, human-paced workflow, not a graph run.

**One command, end to end** (build → submit → poll → download → collect):

```bash
content-batch pipeline --lang tl --version ASND --year 2026 \
    --provider fireworks_batch_devotional_gen \
    [--poll-interval 30] [--timeout 86400] [--out <path>] [--dry-run] [--limit N]
```

It prints a `[n/5]` status line per phase and a final summary (records collected,
error count, output path). Polling blocks, reporting batch status at each interval.
Any step failing (upload/submit error, batch `failed`/`expired`/`cancelled`, timeout,
download error, unusable results) stops the run immediately with a
`step n/5 (<name>) failed: …` message on stderr and exit code 1 — it never silently
continues to the next step.

`--dry-run` runs step 1 only, then prints what the remaining four steps would do.
It makes no network call and needs no API key.

**Or run each step by hand** (useful for resuming a partially-completed batch):

```bash
content-batch build-year --lang es --version RVR1960 --year 2026 --dry-run
content-batch submit     --input <jsonl>
content-batch poll       --batch-id <id>
content-batch download   --output-file-id <id> --dest <path>
content-batch collect    --results <jsonl> --lang es --version RVR1960 --year 2026
```

Providers come from `config/providers.yml`; the batch API key is read from the
provider's `env_var` (`FIREWORKS_API_KEY` for the default provider).

## Architecture

Layering rules, quality gates, and architect stop-points for this project are defined
in `.claude/skills/langgraph-coding-agent/SKILL.md` — load it before writing or editing
any node, state field, domain module, or graph wiring.

## Status

Scaffolding + dependencies only. No domain logic, nodes, or graph wiring implemented yet.
