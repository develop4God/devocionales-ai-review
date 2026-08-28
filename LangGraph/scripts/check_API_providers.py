"""
Fires one cheap live call at every enabled provider in providers.yml (or a
--config override) and reports pass/fail -- a fast way to check which
API keys/accounts currently have quota before launching a real worker on them.

Nothing here is hardcoded to a specific provider: it reads whatever
providers.yml declares and resolves each one through the same
domain.providers.get_model() every real graph node uses, so a new provider
added to the yml is checkable with no script changes.

Usage:
    python3 scripts/check_providers.py
    python3 scripts/check_providers.py --config /path/to/other_providers.yml
    python3 scripts/check_providers.py --only groq_gpt_oss_20b,ollama_local
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_PER_CALL_TIMEOUT_S = 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a providers.yml to use instead of config/providers.yml.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated provider ids to check instead of every enabled one.",
    )
    args = parser.parse_args()

    if args.config:
        os.environ["CONTENT_BATCH_GRAPH_PROVIDERS_CONFIG"] = args.config

    from content_batch_graph.domain.providers import _load_config, get_model

    config = _load_config()
    providers = [p for p in config["providers"] if p.get("enabled", True)]
    if args.only:
        wanted = {p.strip() for p in args.only.split(",")}
        providers = [p for p in providers if p["id"] in wanted]

    results = []
    for provider in providers:
        provider_id = provider["id"]
        if provider.get("client_type") == "batch":
            print(f"{provider_id}: SKIP (batch-only, no live chat model)")
            continue

        start = time.time()
        try:
            model = get_model(provider_id)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    model.invoke, "Reply with exactly one word: pong"
                )
                resp = future.result(timeout=_PER_CALL_TIMEOUT_S)
            elapsed = time.time() - start
            content = (resp.content or "").strip().replace("\n", " ")[:40]
            print(f"{provider_id}: OK ({elapsed:.1f}s) -> {content!r}")
            results.append((provider_id, True))
        except FutureTimeoutError:
            # The underlying request keeps running in its worker thread (no clean
            # cancel for an in-flight HTTP call) but this script moves on rather
            # than blocking the whole check on one slow/hung provider.
            elapsed = time.time() - start
            print(f"{provider_id}: FAIL ({elapsed:.1f}s) -> timed out")
            results.append((provider_id, False))
        except Exception as e:  # noqa: BLE001 -- deliberate: report every
            # provider's failure without one bad key stopping the whole check.
            elapsed = time.time() - start
            msg = str(e).replace("\n", " ")[:150]
            print(f"{provider_id}: FAIL ({elapsed:.1f}s) -> {msg}")
            results.append((provider_id, False))

    ok = sum(1 for _, passed in results if passed)
    print(f"\n{ok}/{len(results)} providers responded.")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
