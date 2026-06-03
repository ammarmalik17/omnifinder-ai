"""
Benchmark module - Tests free models for responsiveness and latency.

Extracted from llm_gateway.py to keep responsibilities separated.
Uses asyncio + httpx.AsyncClient for concurrent HTTP calls.
The sync wrapper (run_benchmark_sync) creates a fresh event loop to play
nicely with Streamlit's running event loop.
"""

import asyncio
import time
from typing import List, Optional, Tuple

import httpx

# How many concurrent benchmark requests to allow — kept low to avoid
# triggering OpenRouter's free-tier rate limits (50 requests/day total).
_BENCHMARK_MAX_CONCURRENT = 3

# Simple prompt for benchmarking (must be fast to generate)
_BENCHMARK_PROMPT = "Say exactly: ok"

# Timeout per model (seconds) — short, we're just testing "Say exactly: ok"
_BENCHMARK_TIMEOUT = 4

# OpenRouter API base URL
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


async def _test_single_model(
    client: httpx.AsyncClient,
    api_key: str,
    model_id: str,
) -> Optional[Tuple[str, float]]:
    """Test a single model asynchronously. Returns None if it fails."""
    try:
        start = time.perf_counter()
        response = await client.post(
            f"{_OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": _BENCHMARK_PROMPT}],
                "max_tokens": 10,
                "temperature": 0,
            },
            timeout=_BENCHMARK_TIMEOUT,
        )
        elapsed = time.perf_counter() - start

        if response.status_code != 200:
            body = response.text[:200]
            print(f"[Benchmark] ⚠ {model_id:<55} HTTP {response.status_code} — {body}")
            return None

        data = response.json()
        if data.get("choices") and data["choices"][0].get("message", {}).get("content"):
            return (model_id, round(elapsed, 2))

        print(f"[Benchmark] ⚠ {model_id:<55} unexpected response shape: {str(data)[:150]}")
        return None
    except Exception as e:
        print(f"[Benchmark] ⚠ {model_id:<55} exception: {e}")
        return None


async def run_benchmark_async(
    api_key: str, models: List[str]
) -> List[Tuple[str, float]]:
    """
    Benchmark a list of models concurrently using asyncio.

    Args:
        api_key: OpenRouter API key.
        models: List of model IDs to test.

    Returns:
        List of (model_id, latency) for responsive models, sorted by latency.
    """
    total = len(models)
    responsive: List[Tuple[str, float]] = []

    print(
        f"\n[Benchmark] Testing {total} models "
        f"(max {_BENCHMARK_MAX_CONCURRENT} concurrent)..."
    )

    connector = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=_BENCHMARK_MAX_CONCURRENT,
            max_keepalive_connections=_BENCHMARK_MAX_CONCURRENT,
            keepalive_expiry=30.0,
        ),
    )

    semaphore = asyncio.Semaphore(_BENCHMARK_MAX_CONCURRENT)

    async with connector as client:
        async def bounded_test(model_id: str) -> Tuple[str, Optional[Tuple[str, float]]]:
            """Rate-limited wrapper around _test_single_model."""
            async with semaphore:
                result = await _test_single_model(client, api_key, model_id)
                return (model_id, result)

        # Create all tasks
        tasks = {asyncio.create_task(bounded_test(m)): m for m in models}
        completed = 0

        for future in asyncio.as_completed(tasks):
            completed += 1
            model_id, result = await future
            if result is not None:
                responsive.append(result)
                print(
                    f"[Benchmark] ✓ {result[0]:<55} "
                    f"{result[1]:.1f}s  ({completed}/{total})"
                )
            else:
                print(
                    f"[Benchmark] ✗ {model_id:<55} "
                    f"failed     ({completed}/{total})"
                )

    responsive.sort(key=lambda x: x[1])
    if responsive:
        print(
            f"[Benchmark] Done — {len(responsive)}/{total} responsive, "
            f"fastest: {responsive[0][0]} ({responsive[0][1]:.1f}s)"
        )
    else:
        print(f"[Benchmark] Done — 0/{total} responsive")
    return responsive


def run_benchmark_sync(
    api_key: str, models: List[str]
) -> List[Tuple[str, float]]:
    """
    Synchronous wrapper around run_benchmark_async.

    Handles two scenarios:
    - No running event loop: uses asyncio.run() directly.
    - Already in an event loop (e.g. Streamlit): spawns a new thread with
      its own event loop, so the benchmark doesn't block Streamlit's loop.
    """
    async def runner():
        return await run_benchmark_async(api_key, models)

    try:
        asyncio.get_running_loop()
        # Already inside an event loop — run in a fresh thread
        import threading

        result_container: List[List[Tuple[str, float]]] = []

        def _run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result_container.append(loop.run_until_complete(runner()))
            finally:
                loop.close()

        t = threading.Thread(target=_run_in_thread, daemon=True)
        t.start()
        t.join()
        return result_container[0]

    except RuntimeError:
        # No running event loop
        return asyncio.run(runner())
