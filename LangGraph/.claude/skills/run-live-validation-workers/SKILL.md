---
name: run-live-validation-workers
description: Verify and generate the commands for a multi-worker run_live_validation.py run (typo/grammar/awkward_phrasing review of a devotional corpus file). Loads before launching any live validation run with more than one worker. Confirms the corpus file exists and has the expected data.<language_key>.<date>[i].<field> shape, the role exists in config/roles.yml, and enough distinct provider ids (each backed by its own real API key) exist in config/providers.yml for the requested worker count — then prints the exact per-worker commands, all sharing one ledger, no shard flag, no shell dispatcher. Use when the user asks to run live validation with N workers, or names a corpus file, role, model, and worker count for a LangGraph review run.
---

# Run Live Validation — Multi-Worker Command Builder

Every live validation run (ES/PT/EN/FR 2027, etc.) has needed the same manual
steps: find the corpus file, pick a role, pick a model, enumerate enough
distinct provider ids/keys for N workers, then hand-write N commands. This
skill does those checks and builds those commands — it does not invoke a
shell dispatcher and does not use `--shard`. Every worker gets the full
pending list from one shared `--ledger`; each worker's own pending
computation (already in `run_live_validation.py`) is what prevents duplicate
work across workers, confirmed clean (0 duplicates) on the real PT ARC 2027
run across 13 different provider ids sharing one ledger.

## Inputs to collect from the user (ask if not given)

1. **Corpus file** — absolute path to the generated devotional JSON (e.g.
   `.../raw_fr_LSG1910_gemma4-3a-26b_<timestamp>.json` or a
   `Devocional_year_<year>_<lang>.json`).
2. **Language / language-key** — e.g. `French` / `fr`, `Spanish` / `es`.
3. **Role** — a role id from `config/roles.yml` (e.g. `native_reader_batch`
   for typo/grammar only, `native_reader` for typo/grammar/awkward_phrasing,
   or a language-specific variant like `native_reader_batch_pt`).
4. **Model family** — which provider id family to use (e.g.
   `groq_gpt_oss_120b`, `groq_gpt_oss_20b`, `cerebras_default`).
5. **Number of workers (N)**.

## Verification steps — do all of these before printing any command

### 1. Corpus file exists and has the expected shape

```bash
python3 -c "
import json, sys
path = '<corpus_file>'
lang_key = '<language_key>'
with open(path) as f:
    d = json.load(f)
lang_data = d.get('data', {}).get(lang_key, {})
if not lang_data:
    print(f'ERROR: no data.{lang_key} found in {path}')
    sys.exit(1)
total = sum(len(v) for v in lang_data.values())
sample_date = sorted(lang_data)[0]
sample_entry = lang_data[sample_date][0]
print(f'OK: {total} entries under data.{lang_key}')
print(f'sample entry fields: {list(sample_entry.keys())}')
"
```

Confirm the fields you intend to validate (usually `reflexion,oracion`) are
actually present as keys on a sample entry — do not assume, read the real
output.

### 2. Role exists in config/roles.yml

```bash
python3 -c "
import yaml
with open('config/roles.yml') as f:
    d = yaml.safe_load(f)
ids = [r['id'] for r in d['roles']]
role = '<role_id>'
print('OK: role found' if role in ids else f'ERROR: {role} not in {ids}')
"
```

### 3. Enough distinct provider ids exist for N workers, each with a real key

List every provider id belonging to the requested model family (e.g. every
`groq_gpt_oss_120b*` id), confirm there are at least N of them, and confirm
their `env_var`s map to distinct, non-empty values in `.env` — **never assume
a fallback key is populated or distinct just because the provider entry
exists**.

```bash
python3 -c "
import yaml
with open('config/providers.yml') as f:
    d = yaml.safe_load(f)
family = '<model_family_prefix>'  # e.g. 'groq_gpt_oss_120b'
matches = [p for p in d['providers'] if p['id'] == family or p['id'].startswith(family + '_fallback')]
for p in matches:
    print(p['id'], '->', p['env_var'])
"
```

Then, for each `env_var` printed, confirm it is set and check the first ~15
chars of each value are distinct (never print full keys):

```bash
for var in <env_var_1> <env_var_2> ...; do
  val=$(grep "^${var}=" .env | cut -d= -f2)
  echo "$var -> ${val:0:15}... (len ${#val})"
done
```

If fewer than N distinct, populated keys exist, **stop and tell the user** —
do not silently reduce the worker count or reuse a key across two workers.

## Output: the N commands

One `run_live_validation.py` invocation per worker. All share:
- the same `--corpus-file`
- the same `--language` / `--language-key` / `--fields`
- the same `--ledger` path (new path per run — do not reuse another run's
  ledger; pick a name like `data/checkpoints/<year>_<lang>_<version>_review_ledger.jsonl`)
- the same `--role`

Each worker gets its own `--provider` (one of the verified distinct ids) and
its own `--checkpoint` (SqliteSaver is not safe for concurrent writers — never
share a checkpoint file across workers).

**No `--shard` flag** — every worker computes its pending list fresh against
the shared ledger, so workers naturally divide the remaining work without a
static partition.

```bash
cd <langgraph_repo_root>

# Worker 1
.venv/bin/python scripts/run_live_validation.py \
  --corpus-file <corpus_file> \
  --language <Language> --language-key <lang_key> --fields <fields> \
  --checkpoint data/checkpoints/<run_name>_worker1.sqlite \
  --ledger data/checkpoints/<run_name>_ledger.jsonl \
  --provider <provider_id_1> --role <role_id>

# Worker 2
... (same, --provider <provider_id_2>, --checkpoint ..._worker2.sqlite)

# ... through Worker N
```

## On a worker exiting non-zero

`run_live_validation.py` already stops loudly on quota exhaustion — it prints
`STOPPED on daily quota after N/M items this run: <error>` to stderr and
exits 1. This is not silent and is not this skill's job to change.

Fallback is manual, not automatic: relaunch the same command with a different
`--provider` (next fallback id, or a different model family entirely, e.g.
`groq_gpt_oss_20b*` once every `groq_gpt_oss_120b*` key is exhausted) and a
fresh `--checkpoint` name, keeping the same `--ledger`. It will pick up
exactly the items still pending — no duplicate work, per the same pending-list
logic that makes multi-worker runs safe in the first place.

## What this skill does NOT do

- Does not add `--shard` — confirmed unnecessary for this pattern; sharding is
  a fixed, one-time partition computed at worker startup, not live
  work-stealing, and isn't needed when every worker already reads the same
  shared ledger fresh.
- Does not automatically retry a different model on quota exhaustion inside
  the Python script — that is a real behavior change to
  `scripts/run_live_validation.py` and is out of scope unless the user
  explicitly asks for it as its own task.
