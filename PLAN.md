# content_batch_graph — Implementation Plan

A LangGraph pipeline automating a proven manual protocol: translate/generate content,
run independent critic rounds, verify every finding against the real file before
trusting it, fix, validate with real programmatic checks, get human confirmation —
repeated per phase, with durable recurring patterns proposed (never written) back for
human approval at the end.

See `README.md` for what this is and why. See
`.claude/skills/langgraph-coding-agent/SKILL.md` for the architecture rules, quality
gates, and architect stop-points every change here follows.

Built as independent, real, runnable slices — each proven with real tests before the
next starts. Nothing below is built speculatively ahead of the slice that needs it.

---

## Slice 1 — Graph mechanics: flag → verify → human_confirm

**Status: done.**

- Real `StateGraph`: `flag_pass` → `verify_pass` → `human_confirm`
- Real `SqliteSaver` checkpointer — proven to survive across genuinely separate OS
  processes (`interrupt()` / `Command(resume=...)`), not just in-process state
- `domain/verify.py` — the core protocol discipline: a finding is never trusted until
  its quoted text is confirmed present, verbatim, in the real source file
- `domain/flag.py` — deterministic stub (finds literal `"teh"`), proves graph mechanics
  without needing a model call yet
- 12 tests: domain logic + full graph mechanics incl. cross-process resume

## Slice 1.5 — Provider resolution layer

**Status: done.**

- `config/providers.yml` — provider definitions, no hardcoded provider list in Python
  (same philosophy as GEP's `providers.yml`)
- `domain/providers.py` — resolves a provider id to a real LangChain chat model
  (`ChatAnthropic` today; `ChatOpenAI` branch ready for any OpenAI-compatible endpoint)
- Decision recorded: the Claude Code CLI adapter (`langchain-claude-code-cli`) was
  evaluated and rejected for now — its `claude-code-sdk` dependency has a real,
  currently-unfixed bug against the installed CLI version, and it doesn't support
  `with_structured_output()`, which this architecture depends on. Real Anthropic API
  key used instead. Revisit the CLI adapter path later if/when it's fixed upstream —
  the model call is isolated to one function (`domain/providers.get_model`), so
  switching back is a contained change, not a redesign.
- 4 tests, including the missing-API-key failure path

## Slice 1.6 — Local provider support (`client_type: local`/`api`)

**Status: done.**

Added after discovering both cloud providers tried for Slice 2 had real, currently
blocking problems (see below) — architect asked for a local/remote flag, matching
GEP's `client_type` distinction, so testing isn't blocked on cloud provider issues.

- `providers.yml` entries now declare `client_type: local` or `api`
- `ollama_local` (qwen2.5:0.5b, the model already pulled locally) added and set as
  `default_provider` — needs no API key, `domain/providers.py`'s key-presence check is
  skipped entirely for `client_type: local`
- Verified end-to-end: a real `.with_structured_output()` call against local Ollama
  returns a correctly-typed Pydantic object. This is the first genuinely working
  structured-output call in the project.
- 2 new tests (17 total), including that the default provider needs no env setup

## Slice 2 — Real model call in the flag pass

**Status: in progress — unblocked via local Ollama, cloud providers have known issues.**

Provider testing history (for the next session, so these aren't re-discovered):
- **OpenAI**: key valid, wiring correct, but account has `insufficient_quota` (no
  billing credits) — not a code problem, needs credits added at
  platform.openai.com/settings/organization/billing if OpenAI is wanted later.
- **Cerebras**: currently **broken** — `langchain-cerebras==0.8.1` imports
  `_convert_chunk_to_generation_chunk` from `langchain_openai.chat_models.base`, which
  doesn't exist in current `langchain-openai` (tried 0.3.34 through 1.4.3, same
  failure on all). This is a real upstream incompatibility in `langchain-cerebras`
  itself, not a version-pin problem we can fix by adjusting our own `pyproject.toml`.
  Revisit when `langchain-cerebras` publishes a release compatible with current
  `langchain-openai`.
- **Local Ollama**: works, see Slice 1.6. This is the provider actually used to build
  and test Slice 2 for now.
- Also fixed along the way: `langchain-openai` was pinned `>=0.3,<0.4` (very stale;
  current is 1.4.x) from an initial guess that was never revisited — corrected to
  `>=1.4,<2.0`. Same category of mistake as the earlier `langgraph-checkpoint-sqlite`
  pin bug in Slice 1 — a version ceiling set without checking what else in the
  dependency graph actually needs, worth double-checking pins more carefully going
  forward rather than trusting an initial guess.

- [ ] Design the role/persona shape as data (persona, instructions, flag categories) —
      not hardcoded per content type, matching the "role-agnostic" direction discussed
- [ ] Populate exactly one real role first (native-speaker linguistic check), proven
      against real content before any second role is designed
- [ ] Replace `domain/flag.py`'s stub body with a real call: `get_model()` +
      `.with_structured_output()` against a `Finding`-shaped Pydantic schema
- [ ] Test against a real file from `devocionales-json` (not synthetic fixtures) —
      per architect's stated preference for proving against real content
- [ ] Verify `verify_pass` still correctly rejects any hallucinated/unverifiable
      findings the real model produces (it should — `verify.py` doesn't change)

## Slice 3 — Fix + validate loop

**Status: not started.**

- [ ] `fix_pass` node — proposes a correction for each human-approved finding
- [ ] `validate_pass` node — real programmatic checks (JSON validity at minimum; more
      once a real content type/schema is chosen)
- [ ] Loop-back edge: validate fails → back to fix; validate passes → continue
      (the piece GEP never had — a structural guarantee a fix didn't introduce a
      regression, not just an LLM's self-assessment)

## Slice 4 — Two independent critic rounds

**Status: not started.**

- [ ] Wrap flag→verify→fix→validate to run twice per phase
- [ ] Round 2 must be blind to round 1's specific findings (only aware the file
      changed) — the independence rule from the source protocol. This is the trickiest
      state-design piece: easy to accidentally leak round 1 context into round 2's
      prompt or state.
- [ ] Handle the "oscillating finding" case (round 2 flags something that would undo a
      round 1 fix) — protocol says stop and let a human decide, not auto-resolve

## Slice 5 — Multi-target fan-out

**Status: not started.**

- [ ] `Send()`-based fan-out — one pipeline run per target language/file, matching
      "spawn one translator per language" from the source protocol
- [ ] Only meaningful once slices 2–4 work correctly for a single target

## Slice 6 — Durable pattern memory + final report

**Status: not started.**

- [ ] A recurring finding across runs can be proposed as a durable rule (the
      genome-equivalent from GEP) — proposed only, never written without explicit
      human confirmation
- [ ] Final cross-check / summary step, mirroring the source protocol's Phase 3

---

## Open architecture decisions (not yet made)

- Content-type adapters: how a specific file format (GEP-style devotional JSON,
  plain-text study, Discovery/Encounters JSON) is turned into `(reference, text)`
  units the flag pass iterates over. Agreed to be "structure-aware per content type"
  in principle; no adapter has been designed yet.
- Whether per-criterion atomic judgment (one LLM call per flag category, avoiding the
  documented "halo effect" bias) is worth its ~N× cost — deferred until slice 2/3 are
  proven and real audit data exists to judge whether it's actually needed.

## Working agreements (apply across all sessions)

- Never paste real API keys/secrets into chat — write them directly into `.env`
  (gitignored) and just confirm they're there.
- Architecture-level decisions (dependencies, checkpointer/persistence backend,
  provider integration shape, state schema) get researched against current
  community/official practice and brought as a reasoned recommendation with evidence —
  not a bare multiple-choice menu.
- Every real code change ships with tests in the same change; the existing suite runs
  before and after to separate new regressions from pre-existing ones.
- Nothing gets committed without being shown to the architect first (staged, reported,
  confirmed) — see `.claude/skills/langgraph-coding-agent/SKILL.md` for the full gate
  list and architect stop-points.
