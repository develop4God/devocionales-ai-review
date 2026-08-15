"""
content-batch — CLI for offline devotional batch generation.

This is a standalone operator tool, deliberately outside the compiled StateGraph:
building/submitting/collecting a 365-day batch is a multi-day, human-paced
workflow, not a graph run. It reuses the same domain layer the graph does and
adds no orchestration of its own.

    content-batch build-year --lang es --version RVR1960 --year 2026 --dry-run
    content-batch submit     --input <jsonl>   --provider <id>
    content-batch poll       --batch-id <id>   --provider <id>
    content-batch download   --output-file-id <id> --dest <path> --provider <id>
    content-batch collect    --results <jsonl> --lang es --version RVR1960 --year 2026

`pipeline` chains all five steps in one blocking run (mirroring GEP's
batch_pipeline.py orchestrator):

    content-batch pipeline --lang tl --version ASND --year 2026 [--dry-run]

The native_reader_batch review path (typo/grammar review of an existing
corpus, not generation) is a separate three-step flow — review-build, then
review-submit, then review-collect — not the five-step
build/submit/poll/download/collect shape above, since a review batch reads an
existing corpus rather than a date range (review-build), and BatchClient.submit
needs an already-created+uploaded dataset (create_dataset + upload), which
review-submit does directly rather than reusing cmd_submit's
upload(path)-returns-an-id shape. review-collect parses a downloaded results
file into Finding-shaped output; poll/download themselves are unchanged (a
review job is still just a Fireworks batch job under the hood):

    content-batch review-build   --corpus-dir <path> --lang es [--role native_reader_batch]
    content-batch review-submit  --input <jsonl> --lang es --provider <id> [--job-id <id>]
    content-batch poll           --batch-id <id> --provider <id>
    content-batch download       --output-file-id <id> --dest <path> --provider <id>
    content-batch review-collect --results <jsonl> --lang es [--role native_reader_batch]

Every finding review-collect writes is raw/unverified (see
domain/review_collect.py's own docstring) — a real test batch found at least
one wrong finding (a correct word flagged as a typo), so nothing here should be
applied to real content without a verification pass first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from batch_common import BatchAPIError, BatchClient, read_jsonl, write_jsonl

from content_batch_graph.domain import batch_io
from content_batch_graph.domain.batch_collect import (
    parse_results,
    write_year_collection,
)
from content_batch_graph.domain.batch_providers import get_batch_provider
from content_batch_graph.domain.devotional_gen import build_year_batch
from content_batch_graph.domain.review_collect import (
    parse_review_results,
    write_review_collection,
)
from content_batch_graph.domain.review_gen import (
    build_review_batch,
    build_review_system,
    review_units_from_corpus,
)
from content_batch_graph.domain.roles import get_role

DEFAULT_PROVIDER = "fireworks_batch_devotional_gen"
DEFAULT_ROLE = "devotional_author"
DEFAULT_REVIEW_ROLE = "native_reader_batch"

# Rough heuristic only — enough to catch an order-of-magnitude mistake before a
# 365-request submission, not a billing figure. ~4 chars per token is the usual
# English/Spanish approximation.
_CHARS_PER_TOKEN = 4


def _estimate_tokens(records: list[dict]) -> tuple[int, int]:
    """(estimated prompt tokens, estimated max completion tokens) for a batch."""
    prompt_chars = sum(
        len(m.get("content", ""))
        for r in records
        for m in r["body"].get("messages", [])
    )
    completion = sum(int(r["body"].get("max_tokens", 0)) for r in records)
    return prompt_chars // _CHARS_PER_TOKEN, completion


class PipelineStepError(RuntimeError):
    """A pipeline step failed; carries which step, so the operator can resume there."""


def _build_year_file(args: argparse.Namespace, provider) -> tuple[Path, int]:
    """
    Build a year's records and write them to the conventional input path.

    Shared by `build-year` and `pipeline` step 1 so the two can never drift on
    record content, filename convention, or estimates.
    """
    role = get_role(args.role)
    records = build_year_batch(
        year=args.year,
        lang=args.lang,
        version=args.version,
        provider=provider,
        role=role,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if args.limit:
        records = records[: args.limit]

    out = batch_io.batch_input_path(
        args.lang,
        args.version,
        args.year,
        provider.provider_id,
        batch_io.model_slug(provider.model),
        batch_io.utc_timestamp(),
    )
    write_jsonl(out, records)

    prompt_tokens, completion_tokens = _estimate_tokens(records)
    print(f"Wrote {len(records)} records -> {out}")
    print(f"  provider: {provider.provider_id} ({provider.model})")
    print(f"  est. prompt tokens:     ~{prompt_tokens:,}")
    print(f"  est. max completion:    ~{completion_tokens:,}")
    print(
        "  (shared system prompt is identical across records — the provider's "
        "prefix cache makes actual prompt cost far lower than the raw estimate)"
    )
    return out, len(records)


def cmd_build_year(args: argparse.Namespace) -> int:
    provider = get_batch_provider(args.provider)
    _build_year_file(args, provider)
    if args.dry_run:
        print("Dry run: nothing submitted. Review the file, then run `submit`.")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    provider = get_batch_provider(args.provider)
    client = BatchClient(provider)
    path = batch_io.resolve_batch_input(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Batch input file not found: {path}")

    file_id = client.upload(path)
    print(f"Uploaded {path} -> file_id={file_id}")
    batch_id = client.submit(file_id)
    print(f"Submitted batch -> batch_id={batch_id}")
    print(f"Next: content-batch poll --batch-id {batch_id} --provider {args.provider}")
    return 0


def cmd_review_build(args: argparse.Namespace) -> int:
    """
    Build a native_reader_batch review JSONL from an existing devocionales-json
    corpus — every reflexion/oracion field for --lang, across every Bible
    version file that matches (see domain/scan.find_devotional_files;
    review_units_from_corpus can return units from more than one version for
    one language, e.g. es/RVR1960 + es/NVI both match lang "es" — each record's
    own custom_id carries its actual source version, see review_gen.custom_id_for).

    Writes the file and stops — this is the build-only half; run review-submit
    on the resulting file to actually create the batch job. Split from
    review-submit (rather than one combined command) for the same operator
    reason build-year/submit are split: reviewing a multi-hundred-record file
    before spending a real batch job on it is the point.
    """
    provider = get_batch_provider(args.provider)
    role = get_role(args.role)

    units = review_units_from_corpus(args.corpus_dir, args.lang)
    if args.limit:
        units = units[: args.limit]
    if not units:
        raise ValueError(
            f"No reviewable reflexion/oracion fields found for lang={args.lang!r} "
            f"in {args.corpus_dir!r}."
        )

    records = build_review_batch(
        units,
        args.lang,
        provider,
        role,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    out = batch_io.review_batch_input_path(
        args.lang,
        provider.provider_id,
        batch_io.model_slug(provider.model),
        batch_io.utc_timestamp(),
    )
    write_jsonl(out, records)

    versions = sorted({u.version for u in units})
    prompt_tokens, completion_tokens = _estimate_tokens(records)
    print(f"Wrote {len(records)} records -> {out}")
    print(f"  provider: {provider.provider_id} ({provider.model})")
    print(f"  versions covered: {', '.join(versions)}")
    print(f"  est. prompt tokens:     ~{prompt_tokens:,}")
    print(f"  est. max completion:    ~{completion_tokens:,}")
    print(
        "  (shared system prompt is identical across records — the provider's "
        "prefix cache makes actual prompt cost far lower than the raw estimate)"
    )
    if args.dry_run:
        print("Dry run: nothing submitted. Review the file, then run `review-submit`.")
    return 0


def cmd_review_submit(args: argparse.Namespace) -> int:
    """
    Create the dataset, upload a review-build JSONL, and submit the batch job.

    Uses BatchClient's real create_dataset/upload/submit sequence directly
    (not cmd_submit's client.upload(path) -> client.submit(file_id) shape,
    which calls a stale single-arg BatchClient.submit signature that predates
    this client's account-scoped Fireworks shape) — confirmed end-to-end
    against the live API 2026-08-15 (job native-reader-test-jsonschema2-job:
    10/10 rows completed, 0 truncated, valid JSON matching the Finding schema).

    provider.extra_body (e.g. reasoning_effort: low) is applied automatically
    by BatchClient.submit when --extra-body-override isn't given — see
    BatchClient.submit's own docstring for why this fixed a real truncation
    bug on gpt-oss-20b (reasoning alone exhausted max_tokens on 3/5 rows
    without it).
    """
    provider = get_batch_provider(args.provider)
    client = BatchClient(provider)
    role = get_role(args.role)
    path = batch_io.resolve_batch_input(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Batch input file not found: {path}")

    records = list(read_jsonl(path))
    if not records:
        raise ValueError(f"{path} has no records to submit.")

    job_id = args.job_id or f"review-{batch_io.utc_timestamp()}"
    dataset_id = f"{job_id}-in"
    output_dataset_id = f"{job_id}-out"

    print(f"Creating dataset {dataset_id} ({len(records)} examples)...")
    client.create_dataset(dataset_id, example_count=len(records))
    print(f"Uploading {path}...")
    client.upload(dataset_id, path)
    print("Submitting job...")
    client.submit(
        dataset_id,
        output_dataset_id,
        job_id,
        system_prompt=build_review_system(args.lang, role),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(f"Submitted batch -> job_id={job_id}")
    print(f"  input dataset:  {dataset_id}")
    print(f"  output dataset: {output_dataset_id}")
    print(
        f"Next: content-batch poll --batch-id {job_id} --provider {args.provider}\n"
        f"      content-batch download --output-file-id {output_dataset_id} "
        f"--dest data/batch_output/{job_id} --provider {args.provider}"
    )
    return 0


def cmd_poll(args: argparse.Namespace) -> int:
    client = BatchClient(get_batch_provider(args.provider))
    output_file_id = client.poll(
        args.batch_id, interval=args.interval, timeout=args.timeout
    )
    print(f"Batch completed -> output_file_id={output_file_id}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    client = BatchClient(get_batch_provider(args.provider))
    dest = Path(args.dest)
    client.download(args.output_file_id, dest)
    print(f"Downloaded {args.output_file_id} -> {dest}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    results = batch_io.resolve_batch_output(args.results)
    if not results.exists():
        raise FileNotFoundError(f"Results file not found: {results}")

    records, errors = parse_results(results)
    out = write_year_collection(
        args.lang,
        args.version,
        args.year,
        records,
        errors=errors,
        out_path=Path(args.out) if args.out else None,
    )
    print(f"Collected {len(records)} days ({len(errors)} errors) -> {out}")
    for err in errors[:10]:
        print(f"  ! {err['custom_id']}: {err['reason']}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more (full list is in the output file)")
    return 0


def cmd_review_collect(args: argparse.Namespace) -> int:
    results = batch_io.resolve_batch_output(args.results)
    if not results.exists():
        raise FileNotFoundError(f"Results file not found: {results}")

    role = get_role(args.role)
    review_results, errors = parse_review_results(results, role)
    out = write_review_collection(
        args.lang,
        review_results,
        errors=errors,
        out_path=Path(args.out) if args.out else None,
    )
    total_findings = sum(len(r.findings) for r in review_results)
    print(
        f"Collected {len(review_results)} results, {total_findings} findings "
        f"({len(errors)} errors) -> {out}"
    )
    print(
        "  every finding is raw/unverified — run a verification pass before "
        "applying any of these to real content"
    )
    for err in errors[:10]:
        print(f"  ! {err['custom_id']}: {err['reason']}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more (full list is in the output file)")
    return 0


_PIPELINE_FAILURES = (
    BatchAPIError,
    TimeoutError,
    FileNotFoundError,
    ValueError,
    OSError,
)


def _step(n: int, label: str) -> None:
    print(f"\n[{n}/5] {label}", flush=True)


def cmd_pipeline(args: argparse.Namespace) -> int:
    """
    Run build → submit → poll → download → collect end to end.

    Deliberately pure sequencing of the four existing domain modules plus
    BatchClient: no polling loop, path convention, or parsing rule is
    reimplemented here, so this stays in the CLI layer rather than becoming a
    domain module of its own.
    """
    provider = get_batch_provider(args.provider)
    print(f"content-batch pipeline — {args.lang}/{args.version}/{args.year}")
    print(f"  provider: {provider.provider_id} ({provider.model})")
    if args.dry_run:
        print("  DRY RUN — build only, no network calls")

    # ── [1/5] build ───────────────────────────────────────────────────────
    _step(1, f"Building batch for {args.lang}/{args.version}/{args.year}...")
    try:
        input_path, count = _build_year_file(args, provider)
    except _PIPELINE_FAILURES as exc:
        raise PipelineStepError(f"step 1/5 (build) failed: {exc}") from exc
    print(f"      {count} records written to {input_path}")

    results_path = batch_io.batch_output_path(input_path)
    out_path = Path(args.out) if args.out else None

    if args.dry_run:
        print("\nDry run — the remaining steps WOULD run as follows:")
        print(f"  [2/5] Upload {input_path.name} and submit a batch job")
        print(
            f"        (endpoint {provider.endpoint}, "
            f"completion window {provider.completion_window})"
        )
        print(f"  [3/5] Poll every {args.poll_interval}s (timeout {args.timeout}s)")
        print(f"  [4/5] Download results to {results_path}")
        preview_collection = out_path or batch_io.collection_path(
            args.lang, args.version, args.year, batch_io.utc_timestamp()
        )
        print(f"  [5/5] Collect into {preview_collection}")
        print("\nNothing submitted. Re-run without --dry-run to execute.")
        return 0

    client = BatchClient(provider)

    # ── [2/5] upload + submit ─────────────────────────────────────────────
    _step(2, f"Uploading {input_path.name} and submitting...")
    try:
        file_id = client.upload(input_path)
        print(f"      file_id={file_id}")
        batch_id = client.submit(file_id)
    except _PIPELINE_FAILURES as exc:
        raise PipelineStepError(f"step 2/5 (submit) failed: {exc}") from exc
    print(f"      batch_id={batch_id}")

    # ── [3/5] poll (BatchClient.poll owns the loop and its progress output) ─
    _step(3, f"Polling batch {batch_id} every {args.poll_interval}s...")
    try:
        output_file_id = client.poll(
            batch_id, interval=args.poll_interval, timeout=args.timeout
        )
    except _PIPELINE_FAILURES as exc:
        raise PipelineStepError(
            f"step 3/5 (poll) failed for batch_id={batch_id}: {exc}"
        ) from exc
    print(f"      output_file_id={output_file_id}")

    # ── [4/5] download ────────────────────────────────────────────────────
    _step(4, f"Downloading results to {results_path.name}...")
    try:
        client.download(output_file_id, results_path)
    except _PIPELINE_FAILURES as exc:
        raise PipelineStepError(f"step 4/5 (download) failed: {exc}") from exc
    print(f"      saved {results_path}")

    # ── [5/5] collect ─────────────────────────────────────────────────────
    _step(5, "Collecting results...")
    try:
        records, errors = parse_results(results_path)
        # A partially-bad batch is normal and still collects (see batch_collect),
        # but zero usable days means the results file was unusable — that is a
        # pipeline failure, not a collection with 365 errors in it.
        if not records:
            raise ValueError(
                f"no usable records in {results_path} "
                f"({len(errors)} unusable lines) — results file is malformed or empty"
            )
        collected = write_year_collection(
            args.lang,
            args.version,
            args.year,
            records,
            errors=errors,
            out_path=out_path,
        )
    except _PIPELINE_FAILURES as exc:
        raise PipelineStepError(f"step 5/5 (collect) failed: {exc}") from exc

    print("\nPipeline complete.")
    print(f"  records collected: {len(records)}")
    print(f"  errors:            {len(errors)}")
    print(f"  output:            {collected}")
    for err in errors[:10]:
        print(f"  ! {err['custom_id']}: {err['reason']}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more (full list is in the output file)")
    return 0


def _add_provider_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help=f"Batch provider id from config/providers.yml (default: {DEFAULT_PROVIDER})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="content-batch",
        description="Offline devotional batch generation (build / submit / poll / download / collect).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-year", help="Build a year's batch input JSONL.")
    p_build.add_argument("--lang", required=True)
    p_build.add_argument("--version", required=True)
    p_build.add_argument("--year", type=int, required=True)
    _add_provider_arg(p_build)
    p_build.add_argument("--role", default=DEFAULT_ROLE)
    p_build.add_argument("--max-tokens", type=int, default=2048)
    p_build.add_argument("--temperature", type=float, default=0.7)
    p_build.add_argument("--limit", type=int, help="Only build the first N days.")
    p_build.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the JSONL and report counts/estimates without any network call.",
    )
    p_build.set_defaults(func=cmd_build_year)

    p_submit = sub.add_parser("submit", help="Upload a JSONL and create a batch job.")
    p_submit.add_argument("--input", required=True)
    _add_provider_arg(p_submit)
    p_submit.set_defaults(func=cmd_submit)

    p_rbuild = sub.add_parser(
        "review-build",
        help="Build a native_reader_batch review JSONL from a devocionales-json corpus.",
    )
    p_rbuild.add_argument("--corpus-dir", required=True)
    p_rbuild.add_argument("--lang", required=True)
    _add_provider_arg(p_rbuild)
    p_rbuild.add_argument("--role", default=DEFAULT_REVIEW_ROLE)
    p_rbuild.add_argument("--max-tokens", type=int, default=4098)
    p_rbuild.add_argument("--temperature", type=float, default=0.2)
    p_rbuild.add_argument(
        "--limit", type=int, help="Only build the first N reviewable fields."
    )
    p_rbuild.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the JSONL and report counts/estimates without any network call.",
    )
    p_rbuild.set_defaults(func=cmd_review_build)

    p_rsubmit = sub.add_parser(
        "review-submit",
        help="Create the dataset, upload a review-build JSONL, and submit the batch job.",
    )
    p_rsubmit.add_argument("--input", required=True)
    p_rsubmit.add_argument("--lang", required=True)
    _add_provider_arg(p_rsubmit)
    p_rsubmit.add_argument("--role", default=DEFAULT_REVIEW_ROLE)
    p_rsubmit.add_argument("--max-tokens", type=int, default=4098)
    p_rsubmit.add_argument("--temperature", type=float, default=0.2)
    p_rsubmit.add_argument(
        "--job-id", help="Batch job id (default: review-<UTC timestamp>)."
    )
    p_rsubmit.set_defaults(func=cmd_review_submit)

    p_poll = sub.add_parser(
        "poll", help="Poll a batch job until it reaches a terminal status."
    )
    p_poll.add_argument("--batch-id", required=True)
    _add_provider_arg(p_poll)
    p_poll.add_argument("--interval", type=int, default=30)
    p_poll.add_argument("--timeout", type=int, default=86_400)
    p_poll.set_defaults(func=cmd_poll)

    p_dl = sub.add_parser("download", help="Download a completed batch's output file.")
    p_dl.add_argument("--output-file-id", required=True)
    p_dl.add_argument("--dest", required=True)
    _add_provider_arg(p_dl)
    p_dl.set_defaults(func=cmd_download)

    p_collect = sub.add_parser(
        "collect", help="Parse a results JSONL into a year collection."
    )
    p_collect.add_argument("--results", required=True)
    p_collect.add_argument("--lang", required=True)
    p_collect.add_argument("--version", required=True)
    p_collect.add_argument("--year", type=int, required=True)
    p_collect.add_argument("--out")
    p_collect.set_defaults(func=cmd_collect)

    p_rcollect = sub.add_parser(
        "review-collect",
        help="Parse a review results JSONL into Finding-shaped output.",
    )
    p_rcollect.add_argument("--results", required=True)
    p_rcollect.add_argument("--lang", required=True)
    p_rcollect.add_argument("--role", default=DEFAULT_REVIEW_ROLE)
    p_rcollect.add_argument("--out")
    p_rcollect.set_defaults(func=cmd_review_collect)

    p_pipe = sub.add_parser(
        "pipeline",
        help="Run build → submit → poll → download → collect end to end.",
    )
    p_pipe.add_argument("--lang", required=True)
    p_pipe.add_argument("--version", required=True)
    p_pipe.add_argument("--year", type=int, required=True)
    _add_provider_arg(p_pipe)
    p_pipe.add_argument("--role", default=DEFAULT_ROLE)
    p_pipe.add_argument("--max-tokens", type=int, default=2048)
    p_pipe.add_argument("--temperature", type=float, default=0.7)
    p_pipe.add_argument("--limit", type=int, help="Only build the first N days.")
    p_pipe.add_argument("--poll-interval", type=int, default=30, dest="poll_interval")
    p_pipe.add_argument("--timeout", type=int, default=86_400)
    p_pipe.add_argument("--out", help="Override the collected year output path.")
    p_pipe.add_argument(
        "--dry-run",
        action="store_true",
        help="Build only, then print what the remaining steps would do. No network.",
    )
    p_pipe.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batch_io.ensure_dirs()
    try:
        return args.func(args)
    except (BatchAPIError, FileNotFoundError, ValueError, PipelineStepError) as exc:
        # A missing API key, an unsupported provider, or a missing file is an
        # operator error — a one-line message beats a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
