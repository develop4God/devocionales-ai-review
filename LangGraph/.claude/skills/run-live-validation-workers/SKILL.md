---
name: run-live-validation-workers
description: Verify and generate the commands for a multi-worker run_live_validation.py run (typo/grammar/awkward_phrasing review of a devotional corpus file), launched simultaneously via nohup+& with --shard i/N per worker. Loads before launching any live validation run with more than one worker. Confirms the corpus file exists and has the expected data.<language_key>.<date>[i].<field> shape, the role exists in config/roles.yml, and enough distinct provider ids (each backed by its own real API key) exist in config/providers.yml for the requested worker count — then prints the exact per-worker commands. Use when the user asks to run live validation with N workers, or names a corpus file, role, model, and worker count for a LangGraph review run.
---

# Run Live Validation — Multi-Worker Command Builder

Every live validation run (ES/PT/EN 2027, etc.) has needed the same manual
steps: find the corpus file, pick a role, pick a model, enumerate enough
distinct provider ids/keys for N workers, then hand-write N commands. This
skill does those checks and builds those commands, matching the pattern
confirmed from the real PT ARC 2027 run's own launch commands (recovered from
that session's transcript, 2026-09-18): all N workers launched at the same
instant via `nohup ... &` in a shell loop, **each with its own `--shard i/N`**,
all sharing one `--ledger`.

**`--shard` is required for simultaneous launch, not optional.** Without it,
every worker computes its `pending` list once at startup by reading the
shared ledger — but `pending` is a fixed snapshot, never re-checked during the
run. If several workers start within the same few seconds against a
near-empty ledger, they each get nearly the same full pending list and
duplicate work across all of them (confirmed directly: an FR 2027 run
launched without `--shard`, 7 workers simultaneously, produced up to 7x
duplicate processing on the same entries before the mistake was caught).
`--shard i/N` partitions the full item list once, deterministically, so
simultaneously-launched workers each own a disjoint slice from the start —
no race is possible. This is what "no manual dividing" actually refers to:
you don't compute the slice boundaries yourself, `--shard` does it, but the
flag itself must be passed.

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

### 4. If the shared ledger already exists (resuming a run)

If `--ledger` points at a file with existing rows, `--shard` still applies
against the FULL item list (not the pending list) — a worker's shard
membership never shifts as items get done, per `run_live_validation.py`'s own
module docstring. This is safe to resume with the same N and the same shard
assignment. If N changes between runs (e.g. adding a worker after some
finished), every worker must be relaunched with the new N so `--shard i/N`
values stay consistent — do not mix workers computed against different N.

## Output: the N commands

Launch all N workers **simultaneously**, in the background, via a single
shell loop — this is the real pattern used for the PT ARC 2027 run. All
workers share:
- the same `--corpus-file`
- the same `--language` / `--language-key` / `--fields`
- the same `--ledger` path (new path per run — do not reuse another run's
  ledger; pick a name like `data/checkpoints/<year>_<lang>_<version>_review_ledger.jsonl`)
- the same `--role`

Each worker gets its own `--provider` (one of the verified distinct ids), its
own `--checkpoint` (SqliteSaver is not safe for concurrent writers — never
share a checkpoint file across workers), and `--shard <n>/<N>` (1-indexed).

```bash
cd <langgraph_repo_root>
source .venv/bin/activate  # or use .venv/bin/python directly

CORPUS="<corpus_file>"
LEDGER="data/checkpoints/<run_name>_ledger.jsonl"
LOGDIR="<scratchpad_dir>"
mkdir -p data/checkpoints "$LOGDIR"

PROVIDERS=(<provider_id_1> <provider_id_2> ... <provider_id_N>)

for i in "${!PROVIDERS[@]}"; do
  n=$((i+1))
  prov=${PROVIDERS[$i]}
  nohup .venv/bin/python scripts/run_live_validation.py \
    --corpus-file "$CORPUS" \
    --language <Language> \
    --language-key <lang_key> \
    --fields <fields> \
    --checkpoint "data/checkpoints/<run_name>_worker${n}.sqlite" \
    --ledger "$LEDGER" \
    --provider "$prov" \
    --role <role_id> \
    --shard ${n}/${#PROVIDERS[@]} \
    > "$LOGDIR/<run_name>_worker${n}.log" 2>&1 &
  echo "started worker $n pid $! provider $prov"
done
```

## On a worker exiting non-zero

`run_live_validation.py` already stops loudly on quota exhaustion — it prints
`STOPPED on daily quota after N/M items this run: <error>` to stderr and
exits 1. This is not silent and is not this skill's job to change. Check each
worker's log file in `$LOGDIR` (not just the ledger row count) to see this.

Fallback is manual, not automatic: relaunch that one worker with a different
`--provider` (next fallback id, or a different model family entirely, e.g.
`groq_gpt_oss_20b*` once every `groq_gpt_oss_120b*` key is exhausted),
**the same `--shard i/N` it had** (so it resumes exactly its own slice, no
overlap with the other N-1 workers), a fresh `--checkpoint` name, and the
same `--ledger`.

## Verifying a completed or in-progress run

Always check for duplicates, not just row count — row count alone does not
prove correctness:

```bash
python3 -c "
import json
keys = set()
rows = 0
with open('<ledger_path>') as f:
    for line in f:
        d = json.loads(line)
        keys.add((d['entry_id'], d['field']))
        rows += 1
print('rows:', rows, 'distinct (entry_id, field):', len(keys))
print('duplicates:' , rows - len(keys))
"
```

`rows` should equal `len(keys)` at all times. If `rows > len(keys)`, workers
overlapped — this means `--shard` was omitted, or workers were launched with
inconsistent `N` values, or a worker was relaunched with a different shard
than the one it originally owned.

## What this skill does NOT do

- Does not use `run_live_validation_multi_provider.sh` (removed — a different,
  dynamic-dispatch design that doesn't match how these runs are actually
  launched).
- Does not omit `--shard` — every worker in a simultaneous launch must have
  one; this was tried once, broke, and is documented above as the reason why.
- Does not automatically retry a different model on quota exhaustion inside
  the Python script — that is a real behavior change to
  `scripts/run_live_validation.py` and is out of scope unless the user
  explicitly asks for it as its own task.
