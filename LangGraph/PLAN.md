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

## Slice 3.5 — Independent critic pass (single round) + surgical fix + drift check

**Status: done — proven end-to-end, including two real production bugs found
and fixed via real multi-language corpus testing (2026-08-12).**

This is NOT Slice 4's "two independent rounds" — it's the single-critic-pass
foundation that Slice 4 will need. Built (and evolved) across several sessions;
documented together here since PLAN.md fell behind actual git history for a
while and this reconciles it.

- [x] `critic_pass` node (`domain/critic.py`, `nodes/critic_pass.py`) — one real,
      independent model call **per finding** (not one batched call for all
      findings), specifically to avoid the "halo effect" bias one call over a
      list of findings would risk. Judges `is_valid` + proposes exact
      `replacement_text` + `critic_reasoning`.
- [x] `drift_check_pass` node (`domain/drift.py`) — re-checks the fixed text for
      new/unresolved issues after a fix is applied; wired into the graph with a
      real conditional edge (`drift_check_pass` → `human_confirm` again on
      drift, → `validate_pass` on clean).
- [x] KWF (Komisyon sa Wikang Filipino) dictionary grounding
      (`domain/dictionary.py`) — live query against kwfdiksiyonaryo.ph. Only
      ever used as a **hard dismiss**: a real KWF headword can never be a typo,
      short-circuits before any model call. A miss is never treated as
      evidence of a typo (Filipino dictionaries list roots, not every
      inflected form) — falls through to the critic's own judgment.
- [x] Unicode hyphen-lookalike normalization — the model was observed
      substituting U+2011 (non-breaking hyphen) for a plain ASCII "-" in
      proposed replacements; normalized post-call, before the replacement is
      trusted by anything downstream.
- [x] **Per-finding human approval**: `human_decision` changed from
      `Literal["approved","rejected"]` to `{"apply": [indices into
      critic_findings]}` — a human can approve one finding out of several
      without accepting the rest. Real architect stop-point (state schema +
      human-review semantics), confirmed before implementing.
- [x] `build_pending_review`/`format_pending_review_text`
      (`domain/review_report.py`) — a pre-decision report built straight from
      `critic_findings`, splitting mechanical (typo, dictionary-groundable)
      vs. stylistic (judgment-call, advisory-only) tiers with stable indices
      for the `apply` list. **Meant to be read by the AI doing the review, not
      pasted at the human** — the architect explicitly wants plain-language
      summaries and a recommendation, not raw JSON.
- [x] `build_review_report`/`format_report_text` — the older, complementary
      **post-fix** report (word-level diff, grounded-vs-judgment-only tagging,
      context windows) for confirming what a completed fix actually changed.

**Two real production bugs found via real multi-language corpus testing
(hi/ar/es/en/fil, 2026-08-12) and fixed the same session:**

- **Brazilian Portuguese próclise/ênclise false positive**: the critic flagged
  correct Brazilian Portuguese pronoun placement ("me alegrar") as an error and
  proposed the European-Portuguese form ("alegrar-me") instead — the persona
  was never told which regional standard applies. Fixed with a **post-verdict**
  mechanical dismiss check in `critic.py` (`_dismiss_known_false_positive`),
  deliberately never shown to the model before judgment — priming a model with
  "here's the rule" before it judges risks hallucinated compliance rather than
  real reasoning (architect's own diagnosis, confirmed as the right shape to
  apply to any future dismiss rule).
- **No-op replacement bug** (language-agnostic): the critic returned
  `is_valid=True` with `replacement_text` byte-for-byte identical to
  `quoted_text` (observed on Arabic) — a phantom "fix" that would silently do
  nothing while reporting a false "APPLIED". Same post-verdict dismiss
  mechanism, checked first, applies to every language.

**Controlled-error test (2026-08-12)** — confirms the mechanism itself works:
3 deliberately injected errors (typo, grammar, awkward phrasing) into a real
English entry were all caught, verified, and correctly fixed end-to-end
(flag→verify→critic→fix→drift_check→validate); a 4th flag_pass over-reach was
correctly rejected by critic_pass before reaching the human gate.

**Open problem, not yet solved**: two real 5-entry corpus batches (Portuguese;
then hi/ar/es/en/fil) produced near-zero *trustworthy* findings — most
"findings" on already-reviewed real corpus text turned out to be false
positives on manual verification (the próclise/no-op bugs above, plus an
unaddressed Hindi false positive on "दाखलता" (vine, a correct theological
term) and a Filipino false positive on "pagkaasa" (reliance, a different real
word than "pag-asa"/hope)). Working theory: the real corpus is already fairly
clean, so the critic has little real signal and occasionally manufactures a
plausible-sounding error under implicit pressure to report something.
Dictionary grounding only exists for Filipino (KWF) today — Hindi/Arabic/
generic-Portuguese-beyond-dialect have none. RAG-style grounding (retrieval of
real facts — dictionary hits, past-confirmed-correct terms — fed to the critic
before judgment, NOT rule-priming) was discussed as the likely direction but
not yet built. See Slice 6.

## Slice 4 — Two independent critic rounds

**Status: not started** (distinct from Slice 3.5 above — this is the actual
two-round independence protocol, not yet built).

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
- [x] Clarified (2026-08-14): the "existing fireworks logic" referred to
      `config/providers.yml`'s `fireworks_batch_devotional_gen` entry (batch-only,
      `client_type: batch`) and the `content-batch` CLI (`domain/devotional_gen.py`,
      `domain/batch_collect.py`, `domain/batch_providers.py`) — all built for
      **generation** (date → `{reflexion, oracion}`), not review. See Slice 5.1 below
      for the review-side equivalent this slice actually needs.

### Slice 5.1 — Batch-mode native reader review (round 1), fully automated

**Status: not started — design only, agreed 2026-08-14.**

**Goal (architect's own words):** LangGraph runs this end-to-end with no manual
CLI intervention — no human uploading or downloading a file. The CLI (`content-batch`)
stays a testing/inspection tool only; it is not the production path.

**What already exists, unmodified by this slice:**
- `config/roles.yml`'s `native_reader_batch` role — typo/grammar only (no
  `awkward_phrasing`), plain-text response format
  (`current text: / proposed text: / explanation:`), already written for a batch
  context ("Native Reader (batch-seeded)") but never wired to a batch builder.
- `config/providers.yml`'s `fireworks_batch_devotional_gen` provider entry and
  `batch_common.BatchClient` (upload/submit/poll/download) — transport layer,
  reusable as-is.
- `domain/scan.py::find_devotional_files` / `scan_file_for_pattern` — proves the
  real devocionales-json corpus shape this slice reads from:
  `data.{language}.{date}[i]`, fields `reflexion`/`oracion`, `entry.get("id")`,
  `field_path = "data.{language}.{date}.{i}.{field}"` — matches `BatchState`'s
  existing `entry_id`/`field_path` fields.
- The live `flag_pass` node / `domain/flag.py::run_flag_pass` — **not replaced**.
  Batch mode is an additional path, not a rewrite, the same way
  `client_type: api` vs. `client_type: batch` already coexist in `providers.yml`
  without either replacing the other.

**Research finding (2026-08-14, web-verified — see citations below) that shapes
the design:**
- `interrupt()`/`Command(resume=...)` is the wrong primitive here. It's built for
  human-in-the-loop UX (approve/edit/reject a value); LangGraph re-executes the
  whole node from the top on resume, which would risk re-submitting the batch job.
  No documented LangGraph pattern or example uses it to wait on an external
  webhook/poll result.
- A blocking `while status != done: sleep(); poll()` loop inside one node is not
  officially forbidden, but is not endorsed either, and defeats the point of
  checkpointing (which persists at node *boundaries*, not mid-sleep) — flagged as
  a real problem in at least one real-world LangGraph issue
  (bytedance/deer-flow#1339).
- The closest first-party precedent is LangChain's own async-deep-agents pattern
  (github.com/langchain-ai/async-deep-agents): submit long-running work, return
  immediately, and use a **separate, later invocation** (new run on the same
  thread, triggered externally) to check status — not interrupt/resume.
- No LangChain/LangGraph-native wrapper exists for provider Batch APIs —
  `.batch()` on chat models just calls `.invoke()` N times
  (langchain-ai/langchain#28508, open/unresolved) — confirms the hand-rolled
  `batch_common`/`BatchClient` approach already in this repo is the right level,
  not a missing library to adopt instead.
- `Send()` (map-reduce fan-out) is a poor fit for the *submit* step — the batch
  job is one external call covering N entries, not N node executions — but
  remains the right tool for Slice 5's actual per-language/per-file fan-out once
  results are collected.

**Recommended shape (split-run, not a single blocking run):**
1. `submit_batch_review` node — walks the corpus for the target language/version/
   year (reusing `find_devotional_files`-style discovery), builds one batch record
   per `(entry_id, field)` using `native_reader_batch`'s persona (new domain
   module, e.g. `domain/review_gen.py`, parallel to `devotional_gen.py` — not a
   modification of it), submits via `BatchClient`, writes `batch_id`/`file_id` to
   state, and the run ends normally (no hang).
2. An external trigger (cron, or a thin wrapper — not a human) re-invokes the same
   thread later, hitting `check_batch_review`: polls **once**; not done → ends
   again for the next trigger; done → downloads results, parses the
   `current text: / proposed text: / explanation:` blocks into `list[Finding]`
   keyed by entry/field (new domain module, e.g. `domain/review_collect.py`,
   parallel to `batch_collect.py` — not a modification of it).
3. Existing `verify_pass → prune_pass → critic_pass → human_confirm → ...` runs
   completely unchanged, fed from batch-collected `raw_findings` instead of a live
   `run_flag_pass()` call.

**Still open, genuinely architectural — do not resolve silently when building:**
- Where the mode choice (live vs. batch) lives: a `BatchState` field,
  a `build_graph()`/`compile_graph()` argument, or a separate entrypoint script
  that picks the starting node. Stop-Point 1/2 — architect decides at build time.
- Whether the graph stays one `StateGraph` with a submit/check split, or becomes
  two smaller graphs (submit-graph, review-graph) sharing the same checkpointer
  thread — both satisfy "no manual CLI step," differ in how much of today's single
  `graph.py` changes shape.
- The external trigger mechanism itself (cron entry, systemd timer, a LangGraph
  Platform webhook once deployed there) is out of scope for `content_batch_graph`
  code and belongs to deployment/ops, not this package.

**Explicitly not doing:** replacing `flag_pass`/`domain/flag.py`, replacing
`native_reader` with `native_reader_batch`, or making the CLI the production
entrypoint. Batch mode is additive, mirroring how `native_reader` /
`native_reader_batch` and `client_type: api` / `client_type: batch` already
coexist in this codebase's own config files.

**Update (2026-08-14, real implementation + a real live submit) — corrects
several assumptions above:**

- `domain/review_gen.py` built (units-from-corpus, `build_review_batch`,
  `custom_id_for`) — done, tested, matches the RVR1960 legacy-file naming
  quirk (`Devocional_year_{year}.json`, no `_{lang}_{version}` suffix,
  `find_devotional_files`'s glob does NOT match it — read directly by path
  for that file).
- **`batch_common.BatchClient` was NOT reusable as-is** (contradicts the
  "transport layer, reusable as-is" line above). It was built assuming
  Fireworks' batch API mirrors OpenAI's (`/files`, `/batches`) — a real
  submit attempt returned `404: Path not found: /v1/files`. Full rewrite:
  Fireworks' real batch API is account-scoped resources
  (`accounts/{account_id}/datasets`, `.../batchInferenceJobs`), confirmed
  against `docs.fireworks.ai/guides/batch-inference`. New
  `batch_common/fireworks_template.py` owns every Fireworks-specific
  URL/payload shape; `client.py` is now a thin generic HTTP transport that
  calls into it — mirrors this package's own domain-logic/node split.
- Three more real (not doc-predicted) errors, found only by actually
  submitting:
  1. `base_url` was `.../inference/v1` (the live chat-completions host) —
     the batch/dataset API is plain `.../v1`. Two different base URLs for
     two different Fireworks surfaces.
  2. `create_dataset` needs `exampleCount` in the request body — Fireworks'
     own API reference marks it read-only and doesn't list it as a request
     field; the live server's `400` disagreed with its own docs.
  3. **Job state values are `JOB_STATE_RUNNING`/`JOB_STATE_COMPLETED`/
     `JOB_STATE_FAILED`/`JOB_STATE_EXPIRED`, not the bare words
     (`RUNNING`/`COMPLETED`/...) shown in the docs' own "Job states"
     reference table.** This one would have caused `poll()` to spin until
     timeout on every real outcome, success or failure — caught only
     because a real job was being polled live when the mismatch surfaced.
- Per Fireworks' own documented "Job-level system prompt" shape (and its own
  "best practices" list — cache optimization, shared system prompt): the
  system message is stripped from every dataset line and passed once as the
  batch job's `systemPrompt` instead. `chat_request_record()` (in
  `batch_common/jsonl.py`, shared by `devotional_gen.py` too) no longer
  builds `{custom_id, method, url, body: {model, messages}}` (the old,
  wrong OpenAI-batch envelope) — just `{custom_id, body: {messages}}`,
  model included only at job-submission time.
- `scripts/collect_review_batch.py` — first cut of the "collect" step:
  polls a job id, downloads the output dataset's files once terminal.
  Deliberately just the pull, no parsing into `Finding` objects yet (that's
  `domain/review_collect.py`, still not built).
- **Real live test in progress**: a 366-record half-year batch
  (`es`/RVR1960, `2025-08-01`→`2026-01-30`, `gpt-oss-20b` — swapped in for
  cheap pipeline-mechanics validation before spending quota on
  `deepseek-v4-pro-0813`) was submitted successfully — dataset created,
  uploaded, job created, all confirmed via the job's own status response
  (`state: READY`, `exampleCount: "366"`, correct `systemPrompt` echoed
  back). As of this writing the job has been `JOB_STATE_RUNNING` for 35+
  minutes with `totalInputRequests: 0` / `totalProcessedRequests: 0` —
  past the docs' own documented "contact support if a deployment takes
  >30min to create" threshold, `status.code: OK`, `waitingOnCapacity:
  false`. Architect decision: keep waiting, not yet treated as stuck.
  `job_id=native-reader-es-rvr1960-halfyear-job`,
  `output_dataset_id=native-reader-es-rvr1960-halfyear-out`. Check this job
  first in any future session before starting a new one.
- **Still not built**: `domain/review_collect.py` (parse the plain-text
  `current text: / proposed text: / explanation:` format into
  `list[Finding]`), and everything in "Still open, genuinely architectural"
  above (mode-selection location, one graph vs. two, submit/check nodes in
  `graph.py` itself) — none of that has been touched. Today's work is
  entirely pre-graph: proving the batch transport and record shape work
  against the real API.

## Slice 6 — Durable pattern memory + final report

**Status: partially started, outside the graph.**

- [x] `domain/pattern_memory.py` + `data/pattern_memory.json` — a JSON store of
      critic-confirmed **typo** fixes only (exact-string-safe; grammar/
      awkward_phrasing are context-dependent, never banked). Proposed and
      saved via `save_pattern()`, never auto-applied elsewhere.
- [x] `domain/scan.py` — given a banked pattern, scans the rest of the corpus
      for the same literal string and proposes (never applies) the same fix at
      each match. As of 2026-08-12, scans both `reflexion` AND `oracion`
      fields (was `reflexion`-only before).
- [ ] Not yet built: promoting a *recurring* finding across independent runs
      into a durable rule automatically (today's store only records
      already-confirmed fixes, not pattern detection across runs).
- [ ] Final cross-check / summary step, mirroring the source protocol's Phase 3.
- [ ] RAG-style grounding layer (see Slice 3.5's "open problem") — a natural
      extension of this slice: instead of just banking confirmed *fixes*,
      also bank confirmed *dismissals* (words/constructions wrongly flagged,
      with the evidence for why) and retrieve relevant ones before a critic
      judges a similar span. Not yet designed.

---

## Infrastructure (outside the slice sequence)

- **Repo layout (2026-08-12)**: this project moved from repo root into
  `LangGraph/` (this file's own location), a sibling to a new `GEP/` folder —
  `GEP_Genome-Evolution-Protocol` migrated from a separate repo
  (`~/python/DevocionalesAPI`) via `git subtree`, full 187-commit history
  preserved. Not yet wired to anything in this project; sits as a sibling for
  now, purely a filesystem consolidation.
- **`devocionales-json` dependency (2026-08-12)**: wired as a real `uv` path
  dependency (`[tool.uv.sources]`, editable, points at `../../devocionales-json`
  on this machine) — not a git/PyPI dependency, since the two repos live as
  siblings on the same machine and this stays in sync with that repo's
  `reorg-shared-validation` branch as it evolves. Consumes
  `shared_validation.checks.review_fields` — a per-content-type field-selector
  module built by a prior session on the `devocionales-json` side specifically
  for this project's `field_path` format (`cards.2.content`, dot/digit, no
  translation step needed to splice a fix back in). Not yet actually called by
  any node — available, not integrated. `requires-python` bumped `>=3.11` →
  `>=3.12` to satisfy this dependency.
- **LangSmith tracing (2026-08-12)**: env-vars-only (`LANGSMITH_TRACING`,
  `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` in `.env`), no code change, no new
  dependency (`langsmith` ships transitively via `langchain-core`). Confirmed
  working end-to-end. This is now the way to inspect what a run actually did —
  replaces the ad hoc print-script workflow used for all manual testing so
  far. LangSmith also offers Annotation Queues (could replace
  `build_pending_review`'s manual triage with a real reviewer UI) and Datasets
  (could freeze real test cases — e.g. the controlled-error English test — as
  a permanent regression suite) — researched, not yet wired in.
- **Checkpoint path convention**: real runs should use `data/checkpoints/`
  (gitignored, alongside `data/audit|genomes|logs/`), never `/tmp` — an
  ephemeral filesystem defeats the point of checkpointing across process
  restarts. `compile_graph()`'s docstring states this explicitly.

## Provider routing / worker-count migration (2026-08-28) — planned, not started

**Trigger:** upcoming scale-up to live validation across **10 languages × 2 years each**
(run_live_validation.py / this graph), well beyond the single 2027 ES corpus run this was
built against. Two problems surfaced running that ES corpus with 4 hand-launched worker
processes (2 Groq accounts + 2 local Ollama, each its own `--config`/`--shard`):

1. **No real load-balancing across concurrent workers.** `with_fallbacks()` is sequential
   failover for *one* call (try A, then B, then C) — it does not assign different
   providers to different concurrent workers or track per-key rate-limit state. At
   4-worker scale this was tolerable (providers were assigned by hand, one per
   `--config` file); at 10-language scale, hand-generating a `providers_*.yml` per
   worker per run is not viable.
2. **A worker can die with zero error trace.** Confirmed live on 2026-08-27/28: two Groq
   workers stopped mid-run with no STOPPED/error line in their logs — looked identical
   to a real crash. (Their *second* death, on retry, turned out to be a clean, correctly
   logged "STOPPED on daily quota" — so the mechanism works when it fires; the first,
   silent death is still unexplained and must not be possible in the redesigned driver.)

**Decision (architect-confirmed 2026-08-28):** adopt **LiteLLM** (Python SDK mode, no
separate proxy process) as the provider-routing layer, replacing hand-rolled
`providers_groq1.yml`-style per-worker config files.

**Research trail (2026-08-28) — what was checked and what it showed:**

- [LangGraph Send API / map-reduce parallel execution](https://machinelearningplus.com/gen-ai/langgraph-map-reduce-parallel-execution/) —
  `Send()` fans out N node instances in one graph run, merged via reducers. Ruled out
  for this driver specifically: each corpus item here is its own `invoke()` →
  `interrupt()` → `Command(resume=...)` two-phase sequence on its own `thread_id`, not
  a single-shot input `Send()`/`.batch()` can wrap.
- [LangGraph parallel-interrupt issues (GitHub: #6624, #6533, #6626)](https://github.com/langchain-ai/langgraph/issues/6624) —
  confirmed real, open bugs when multiple `interrupt()` calls fire *within one graph
  invocation* concurrently (identical interrupt IDs, misrouted resumes, lost
  interrupts). Not directly hit by this driver's pattern (each item is a fully separate
  `graph.invoke()`, not concurrent interrupts inside one run) but reinforces: don't
  reach for `Send()`-with-interrupts here.
- [SqliteSaver thread-safety discussion (LangChain GitHub #23630)](https://github.com/langchain-ai/langchain/discussions/23630) —
  confirmed a compiled graph with `SqliteSaver` is safe for concurrent invokes *within
  one process* as long as each gets its own `thread_id` (already true here); the
  file-level-locking danger is specifically *multiple processes* writing the same
  `.sqlite` file, which is why the current design gives each worker process its own
  checkpoint file. Conclusion: real concurrency belongs in one process
  (`ThreadPoolExecutor` over `run_one`), not N separate OS processes each with their
  own SqliteSaver file — supports collapsing the multi-process launcher.
- [`with_fallbacks()` / fallback chains — LangChain OpenTutorial](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/13-langchain-expression-language/11-fallbacks) and
  [Dynamic Failover and Load Balancing LLMs with LangChain](https://medium.com/@andrewnguonly/dynamic-failover-and-load-balancing-llms-with-langchain-e930a094be61) —
  confirmed `with_fallbacks()` is sequential failover for *one* call (try A, then B on
  A's failure) — it has no concept of "N concurrent callers, split across N keys" or
  per-key rate-limit tracking. This directly killed the first design (one shared
  `with_fallbacks()` chain for every worker) — every concurrent worker would race the
  same primary provider first.
- [`RunnableConfigurableAlternatives` reference](https://reference.langchain.com/python/langchain-core/runnables/configurable/RunnableConfigurableAlternatives) —
  lets you swap a runnable's provider at call time via config, but still no built-in
  scheduling/rotation logic across concurrent callers — same gap as `with_fallbacks()`.
- [LiteLLM proxy/reliability docs](https://docs.litellm.ai/docs/proxy/reliability) —
  confirmed LiteLLM ships as both an SDK (no separate service — the relevant mode
  here) and an optional standalone proxy. Its `model_list` + `router_settings` gives
  real multi-key routing (round-robin / least-busy / usage-based) with per-key
  rate-limit awareness, which is the actual feature being asked for
  ("primary, secondary, third, etc, in providers and env").
- Cross-referenced against other gateway-style tools that came up in the same search
  (LLMux, Portkey) — LiteLLM was the most-cited/most-mature option across independent
  sources, not picked from a single source.

**Why LiteLLM over rolling our own scheduler:** a minimal in-house alternative was
considered (`worker_index % len(providers)`, ~15 lines, no new dependency) and was the
initial lean for the *current* single-corpus scale. It was reversed once the real
upcoming scope (10 languages × 2 years, sustained multi-week operation, not a one-shot
732-item run) was stated — that volume justifies letting a maintained library own
key-rotation and rate-limit tracking rather than this codebase inventing and
maintaining its own version of a problem the ecosystem already has a named, common
solution for. Trade-off is real and explicit: one new external dependency
(Stop-Point 4), and some redundancy with this project's own Groq-specific error
handling in `structured_call.py` (LiteLLM won't know about `json_validate_failed`
etc. — see Scope boundary below).

**Scope boundary — do NOT let this migration silently absorb:**
- `domain/structured_call.py`'s Groq-specific retry logic (`json_validate_failed`,
  `output_parse_failed`, `OutputParserException` on Ollama) — this is real, hard-won
  behavior from actual failures on this project's own schema. LiteLLM doesn't know
  about it. Decide explicitly whether it still wraps every call post-migration, don't
  assume LiteLLM replaces it.
- `client_type: batch` (Fireworks batch API in `providers.yml`) — a completely separate
  code path (`domain/batch_providers.py`), out of scope for this migration.

### Plan for next session

1. **Bring a concrete migration plan before writing code** (Stop-Point 4: new external
   dependency; also touches `domain/providers.py`'s public shape, so effectively
   Stop-Point 1/2-adjacent). Plan must show:
   - New `config/litellm_config.yml` (or equivalent) shape: `model_list` entries for
     every current provider (Groq ×2 keys, Cerebras, Ollama local, Anthropic, OpenAI),
     with explicit primary/secondary/tertiary ordering.
   - Exactly what changes in `domain/providers.py`'s `get_model()` /
     `resolve_default_provider_id()` — signature must stay compatible with every
     existing caller (`flag.py`, `critic.py`, `drift.py`) per the SOLID layering rule;
     nodes/domain callers should need zero changes.
   - Whether `structured_call.py`'s retry logic still wraps the LiteLLM-backed model
     (near-certainly yes) and whether its Groq-specific error-code checks need
     generalizing for LiteLLM's own exception shape.
   - How `--workers N` maps to LiteLLM's routing (does N still mean N concurrent Python
     threads calling one shared LiteLLM-routed model, or does LiteLLM's own concurrency
     handling change that shape?).
2. **Fix loud-failure as an explicit, testable requirement**, not a side effect of the
   migration: a worker that dies for any reason (crash, killed process, unhandled
   exception) must leave an unambiguous record — a non-empty error message in its own
   log at minimum; consider a heartbeat/last-seen file per worker so a stalled-but-alive
   process is distinguishable from a dead one, and a still-pending ledger item is
   distinguishable from "silently never attempted."
3. Only after the plan is reviewed and approved: implement, with tests per Gate 5, and
   the existing test suite run before/after per Gate 4 (`tests/test_providers.py` in
   particular currently asserts exact model-class construction — expect real changes
   there, not just additions).

### Acceptance criteria

- [ ] A written migration plan (config shape, `providers.py` diff shape, retry-logic
      disposition) is reviewed and approved by the architect *before* any implementation
      code is written.
- [ ] `get_model()` / `resolve_default_provider_id()` keep their existing signatures and
      behavior for every current caller — `flag.py`, `critic.py`, `drift.py` require zero
      changes.
- [ ] Running with `--workers N` (N = 1, 2, 3, or more) requires **no per-run generated
      config files** — one command, one shared LiteLLM/provider config, works
      unmodified for any worker count.
- [ ] A killed/crashed worker produces a non-empty, human-readable error record (log
      line and/or ledger-adjacent status file) — verified with a real induced failure
      (e.g. `kill -9` a worker mid-run), not just a code review of the error path.
- [ ] A daily-quota-exhaustion stop (the real, correctly-working case observed
      2026-08-28) continues to produce its current clear "STOPPED on daily quota"
      message — must not regress under the new routing layer.
- [ ] Re-running the same command after a partial run (some workers died, some
      succeeded) resumes correctly from the shared ledger with no duplicate processing
      and no lost pending items — same guarantee `apply_shard`'s docstring describes
      today, must still hold.
- [ ] `structured_call.py`'s Groq/Ollama structured-output retry behavior is proven
      still active post-migration (via test, not inspection) — not silently dropped
      because LiteLLM "probably handles retries too."
- [ ] Full existing test suite passes after migration, with every diff from the current
      baseline (`tests/test_providers.py` especially) explained, not just "made to
      pass."
- [ ] Real end-to-end smoke test: at least one real live item processed successfully
      through the new routing layer against a real provider (not a mocked/fake key)
      before declaring the migration done.

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
