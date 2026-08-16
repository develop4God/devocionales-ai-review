"""
Poll a submitted Fireworks review batch job until it completes, then download
its results. First cut of the "collect" step for batch-mode native_reader
review (see PLAN.md Slice 5.1) — just the pull, no parsing into Finding
objects yet.

Usage:
    uv run python scripts/collect_review_batch.py \
        --job-id native-reader-es-rvr1960-halfyear-job \
        --output-dataset-id native-reader-es-rvr1960-halfyear-out \
        --dest data/batch_output/halfyear_review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from batch_common import BatchAPIError, BatchClient

from content_batch_graph.domain.batch_providers import get_batch_provider

DEFAULT_PROVIDER = "fireworks_batch_devotional_gen"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dataset-id", required=True)
    parser.add_argument(
        "--dest", required=True, help="Directory to download results into."
    )
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=86_400)
    args = parser.parse_args()

    client = BatchClient(get_batch_provider(args.provider))

    print(f"Polling job {args.job_id} every {args.interval}s...")
    try:
        state = client.poll(args.job_id, interval=args.interval, timeout=args.timeout)
    except (BatchAPIError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Job reached terminal state: {state}")

    dest = Path(args.dest)
    print(f"Downloading {args.output_dataset_id} -> {dest}")
    written = client.download(args.output_dataset_id, dest)
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
