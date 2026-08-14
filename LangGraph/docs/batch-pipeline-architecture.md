# Two pipelines, one domain layer

Architecture note for the `claude/devotional-batch-langgraph-mttiaa` branch: a new
offline batch-generation CLI (`content-batch`) that sits deliberately outside the
compiled review `StateGraph`, while reusing its config and domain code.

## 1. The existing review graph (unchanged)

Seven nodes, one `StateGraph`, SQLite-checkpointed so `interrupt()` survives across
process runs. This is the live, human-in-the-loop flag→fix pipeline
(`src/content_batch_graph/graph.py`). Nothing on this branch touches it.

```
flag_pass -> verify_pass -> critic_pass -> human_confirm
                                             |-- approved --> fix_pass -> drift_check_pass
                                             |                              |-- drift --> human_confirm (loop)
                                             |                              `-- no drift --> validate_pass
                                             |                                                 |-- passed --------> END
                                             |                                                 |-- failed, <3 tries -> fix_pass (loop)
                                             |                                                 `-- cap reached ----> END
                                             `-- rejected --> END
```

## 2. The new batch pipeline (this branch)

Five linear steps run by `content-batch pipeline` — plain sequential Python, no
graph, no checkpointer, no interrupt. Each step is also its own standalone
subcommand for resuming after a failure.

```
1/5 build --JSONL--> 2/5 submit --batch_id--> 3/5 poll (every 30s) --file_id--> 4/5 download --results.jsonl--> 5/5 collect
   |
   `-- --dry-run stops here (prints what 2-5 WOULD do, no network call)

output of 5/5: data/genomes/Devocional_{year}_{lang}_{version}_gen_o.json
```

`cli.py::cmd_pipeline` is pure sequencing — every step calls an existing domain
function or `BatchClient` method; no logic is reimplemented in the CLI layer.

## 3. Where the two share ground

The batch pipeline is not a second graph — it never touches `StateGraph`,
`BatchState`, or the checkpointer. It is a standalone script that reaches into the
same `domain/` package for config resolution.

```
review graph nodes                          content-batch CLI
        |                                            |
        v                                            v
   domain/providers.py  ·  domain/roles.py   (shared config loader:
                                               config/providers.yml, config/roles.yml)
                                                      |
                                                      v
                              batch_providers · devotional_gen · batch_collect · batch_io
                                        (new domain modules, this branch)
```

Both call sites resolve config through the same `providers.py`/`roles.py` loader —
one YAML source, no drift. `batch_common` (a separate sibling package) supplies the
transport client both this project and GEP install.

## 4. Layering check

| Layer | Review graph | Batch pipeline |
|---|---|---|
| Domain logic | `domain/*.py` — flag/verify/critic/fix rules | `devotional_gen.py` (build requests), `batch_collect.py` (parse results) — net-new, correctly placed outside `cli.py` |
| State schema | `BatchState`, one shared TypedDict | none — no checkpointed state; each step passes plain values (paths, ids) to the next |
| Node / step function | node: state in, partial state out | `cmd_*` function: `argparse.Namespace` in, prints + return code out |
| Wiring | `graph.py` — one `StateGraph` | `cli.py::cmd_pipeline` — one straight-line function, no branching graph |

This confirms the file's own doc comment: "deliberately outside the compiled
StateGraph... adds no orchestration of its own." The pipeline command is
sequencing, not a parallel graph — it doesn't reimplement anything `graph.py` or
the domain layer already owns.

## 5. Dry run: what a run without Fireworks looks like

`content-batch pipeline --dry-run` executes step 1 for real — writes the actual
JSONL to `data/batch_input/` and prints token estimates — then **prints** what
steps 2–5 would do without calling `BatchClient` at all. No API key is required for
a dry run.

```
$ content-batch pipeline --lang es --version RVR1960 --year 2026 --dry-run

[1/5] Building batch for es/RVR1960/2026...
      365 records written to data/batch_input/es_RVR1960_2026_<provider_id>_<model>_<ts>.jsonl

Dry run — the remaining steps WOULD run as follows:
  [2/5] Upload ...jsonl and submit a batch job
  [3/5] Poll every 30s (timeout 86400s)
  [4/5] Download results to data/batch_output/..._results.jsonl
  [5/5] Collect into data/genomes/Devocional_2026_es_RVR1960_gen_o.json

Nothing submitted. Re-run without --dry-run to execute.
```

## 6. Findings — senior review

- **[fix before running]** A generated 365-line JSONL output
  (`LangGraph/…gen_deepseek-v3p2_20260814_051019.jsonl`) is tracked in git — looks
  like a dry-run artifact accidentally staged. Remove it from the commit and
  confirm the path is gitignored (batch_input/output already look gitignored
  elsewhere).
- **[confirm before running]** `fireworks_batch_devotional_gen` targets
  `deepseek-v3p2` at `max_tokens=2048` / `temperature=0.7` by default —
  sanity-check those defaults for a devotional (warm, first-person prose) before
  the real submit.
- **[SOLID: sound]** Layering matches the LangGraph skill's rules — no violations
  found. CLI commands are thin (read args → call domain/client → print); no graph
  logic is duplicated; `batch_common` stays provider-agnostic and takes only a
  `BatchProviderConfig`, never a yml path — dependency inversion is respected end
  to end.
- **[minor]** `_estimate_tokens` is a rough heuristic, correctly labeled as such —
  4-chars-per-token, no billing guarantee, comment says so explicitly. Fine for its
  stated purpose (catch an order-of-magnitude mistake), not a bug.

## 7. Proposed next phase (not yet built) — batch output as graph seed

Discussed direction, pending confirmation before implementation:

- The batch pipeline's output file (`Devocional_{year}_{lang}_{version}_gen_o.json`,
  already produced by `write_year_collection`) becomes the **seed input** for the
  existing review graph — no new artifact format needed.
- A new driver (same "operator tool" pattern as `content-batch pipeline`, e.g.
  `content-batch review-year`) would loop the collection file's entries and invoke
  the compiled graph once per entry — `file_path` = the collection file,
  `field_path` = `data[i].reflexion` (and/or `.oracion` — open question, see below),
  `entry_id` = the date. Each entry gets its own `thread_id`/checkpoint.
- The graph itself is **unchanged**: `flag_pass` already runs with the
  `native_reader` role (typo/grammar/awkward-phrasing only) — confirmed this
  already matches "validate only drift or typos" for freshly-generated content, so
  no new role or node is needed for the first pass.
- **Open question:** each day has two reviewable fields, `reflexion` and
  `oracion`. Does review need to cover both (2 graph runs/day, 730/year) or just
  `reflexion`?
