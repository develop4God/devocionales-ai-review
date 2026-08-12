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

**Status: done — proven end-to-end against real content via local Ollama.**

Provider testing history (for the next session, so these aren't re-discovered):
- **OpenAI**: key valid, wiring correct, but account has `insufficient_quota` (no
  billing credits) — not a code problem, needs credits added at
  platform.openai.com/settings/organization/billing if OpenAI is wanted later.
- **Cerebras**: was **broken** on `langchain-cerebras==0.8.1` (imported
  `_convert_chunk_to_generation_chunk` from `langchain_openai.chat_models.base`,
  removed upstream). **Fixed**: `langchain-cerebras==0.8.2` (released 2025-11-24)
  resolves the import; `pyproject.toml`'s existing `>=0.8,<0.9` pin already permitted
  it, `uv.lock` just needed `uv lock --upgrade-package langchain-cerebras` to actually
  pick it up (the installed `.venv` had silently drifted ahead of the lockfile during
  manual testing — resynced with `uv sync --extra dev`).
- **Local Ollama**: works, see Slice 1.6. Used to build and prove Slice 2's mechanics
  initially; no longer the configured default (see model comparison below).

**Cerebras model comparison (2026-08-12)**, run against real `devocionales-json`
content with the `native_reader` role, comparing `gpt-oss-120b`, `zai-glm-4.7`, and
`gemma-4-31b` — all three share identical rate/token limits (5 req/min, 2400/day,
30K tok/min, 1M tok/day), so that wasn't a differentiator:
- **`gpt-oss-120b`** (chosen, now `default_provider`): only **Production**-status
  model of the three; consistent 1–5s latency; correctly quoted the deliberate typo
  verbatim; was the most willing to actually surface real findings on the devotional
  excerpt (the other two returned zero findings on the same borderline-awkward text —
  plausibly a legitimate stricter-sensitivity judgment call on their part, not a
  compliance failure, since all three caught the unambiguous typo case identically).
- **`zai-glm-4.7`**: correct behavior in testing, but scheduled for Cerebras
  deprecation **2026-08-17** — ruled out as a lasting default on that basis alone.
- **`gemma-4-31b`**: still **Preview** status; one run spiked to 61.6s on a trivial
  clean-text case (vs. ~1–2s normal) — too unstable to trust as default right now.
- All three hit transient `429 queue_exceeded` errors under shared free-tier/preview
  traffic load during testing — environmental, not a code bug; retries succeeded.
- `providers.yml` was cleaned up after the comparison: `zai-glm-4.7`/`gemma-4-31b`
  entries were removed (they were only added to run this comparison, not meant to
  stay as permanent config); `cerebras_default` (`qwen-3-235b`, untouched by this
  comparison) and `ollama_local` remain as other available providers.
- `domain/flag.py::run_flag_pass` gained a `provider_id: str | None = None` param
  (additive, defaults preserve prior behavior) specifically to make this kind of
  side-by-side comparison possible without editing config between runs.
- Also fixed along the way: `langchain-openai` was pinned `>=0.3,<0.4` (very stale;
  current is 1.4.x) from an initial guess that was never revisited — corrected to
  `>=1.4,<2.0`. Same category of mistake as the earlier `langgraph-checkpoint-sqlite`
  pin bug in Slice 1 — a version ceiling set without checking what else in the
  dependency graph actually needs, worth double-checking pins more carefully going
  forward rather than trusting an initial guess.

- [x] Design the role/persona shape as data — `config/roles.yml` +
      `domain/roles.get_role()`, mirroring `providers.yml`'s no-hardcoding pattern.
      Persona is a template string with `{language}` substituted at call time; each
      role declares its own `categories` list.
- [x] Populate exactly one real role: `native_reader` (typo / grammar /
      awkward_phrasing, comments in English, quotes verbatim) — matches the manual
      protocol description given for this role. Pattern-repetition summarization
      (originally part of the same description) was scoped out to Slice 6 (durable
      pattern memory), where PLAN.md already reserves it.
- [x] `domain/flag.py`'s stub replaced with a real call: `get_model()` +
      `ChatPromptTemplate` (system persona / human source text) +
      `.with_structured_output()` against a Pydantic schema built per-role via
      `create_model()`, with `category` constrained to a `Literal` of the role's
      declared categories (schema-level enforcement — a free-text description alone
      wasn't reliably followed by the small local model). New required state field:
      `BatchState.language`.
- [x] Tested against a real file from `devocionales-json`
      (`Devocional_year_2025_en_NIV.json`, a real `reflexion` field) — 5 findings, all
      verbatim-verified, categories all valid after the Literal constraint was added.
- [x] `verify_pass`/`verify_findings` confirmed still correctly separates
      verified/rejected — the local model (qwen2.5:0.5b) was observed "correcting" a
      typo's `quoted_text` instead of quoting it verbatim in one manual test; that
      finding was correctly rejected, not passed through. `verify.py` itself did not
      change. `tests/test_graph.py` now monkeypatches `run_flag_pass` so graph-
      mechanics tests stay fast/deterministic; real-model behavior is covered
      separately in `tests/test_flag.py`.

**Note for later roles:** small local models drift from free-text instructions
(category naming) but comply with schema-level constraints (`Literal`). Prefer
encoding a role's hard constraints in the generated schema over relying on prompt
wording alone, especially while testing against `ollama_local`.

## Slice 3 — Fix + validate loop

**Status: done — proven end-to-end against real content.**

- [x] `fix_pass` node (`domain/fix.py`, `nodes/fix_pass.py`) — a single real model
      call rewrites the whole `file_text`, applying the minimal correction for each
      `human_decision == "approved"` finding, and returns a concise `fix_summary` for
      human review. Fully automatic — no per-finding approval step, per architect's
      direction (no per-language fix criteria to encode manually; one AI step, one
      summary to review after).
- [x] `validate_pass` node (`domain/validate.py`, `nodes/validate_pass.py`) — real
      programmatic check, no LLM: `json.loads()` on `fixed_text`. Matches PLAN.md's
      stated minimum; schema/field-level checks deferred until a content-type adapter
      is designed (still an open architecture decision, see below).
- [x] Loop-back edge: `validate_pass` fails → back to `fix_pass`, capped at
      `MAX_FIX_ATTEMPTS = 3` (`graph.py`) so a persistently-broken fix can't loop
      forever; validate passes → `END`.
- [x] `human_confirm` gained a real conditional edge: `approved` → `fix_pass`,
      anything else → `END`. Previously `human_confirm` always went straight to
      `END` regardless of decision — this is the first slice where the decision
      actually branches the graph. `BatchState` gained `fixed_text`, `fix_summary`,
      `fix_attempts`, `validation_passed`, `validation_error`.
- [x] Proven end-to-end against real `devocionales-json` content (the same
      `Devocional_year_2025_en_NIV.json` entry used for Slice 2): flag → verify (4
      verified, 0 rejected) → human_confirm(approved) → fix (real corrections: a
      comma splice, an unclear Isaiah 55:7 citation phrase, an awkward prayer-closing
      rephrase) → validate (`validation_passed: True`, JSON structure intact,
      `fix_attempts: 1`).
- [x] Graph tests (`test_graph.py`) stub `run_fix_pass` the same way `run_flag_pass`
      is stubbed, so graph-mechanics tests stay fast/deterministic; two pre-existing
      tests that resumed with `"approved"` were silently making real (rate-limited)
      Cerebras calls once fix_pass was wired in and needed the same stub added.
      Real-model behavior has its own tests in `test_fix.py`/`test_validate.py`.

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
