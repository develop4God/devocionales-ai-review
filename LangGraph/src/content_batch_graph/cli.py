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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from batch_common import BatchAPIError, BatchClient, write_jsonl

from content_batch_graph.domain import batch_io
from content_batch_graph.domain.batch_collect import (
    parse_results,
    write_year_collection,
)
from content_batch_graph.domain.batch_providers import get_batch_provider
from content_batch_graph.domain.devotional_gen import build_year_batch
from content_batch_graph.domain.roles import get_role

DEFAULT_PROVIDER = "fireworks_batch_devotional_gen"
DEFAULT_ROLE = "devotional_author"

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
