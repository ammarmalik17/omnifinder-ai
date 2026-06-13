"""
Benchmark module - Tests models for responsiveness and latency.

Provider-agnostic: uses a BenchmarkConfig dataclass so callers can
benchmark models from any OpenAI-compatible API (OpenRouter, Groq, etc.).
The sync wrapper (run_benchmark_sync) creates a fresh event loop to play
nicely with Streamlit's running event loop.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import httpx


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking models from an OpenAI-compatible API.

    Attributes:
        base_url: API base URL (e.g. https://openrouter.ai/api/v1)
        api_key: API key for authentication
        max_concurrent: Maximum concurrent benchmark requests
        timeout: Timeout in seconds per model test
    """
    base_url: str
    api_key: str
    max_concurrent: int = 3
    timeout: int = 4


# Simple prompt for benchmarking (must be fast to generate)
_BENCHMARK_PROMPT = "Say exactly: ok"


async def _test_single_model(
    client: httpx.AsyncClient,
    config: BenchmarkConfig,
    model_id: str,
) -> Optional[Tuple[str, float]]:
    """Test a single model asynchronously. Returns None if it fails."""
    try:
        start = time.perf_counter()
        response = await client.post(
            f"{config.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {config.api_key}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": _BENCHMARK_PROMPT}],
                "max_tokens": 10,
                "temperature": 0,
            },
            timeout=config.timeout,
        )
        elapsed = time.perf_counter() - start

        if response.status_code != 200:
            return None

        data = response.json()
        if data.get("choices") and data["choices"][0].get("message", {}).get("content"):
            return (model_id, round(elapsed, 2))

        return None
    except Exception as e:
        return None


async def run_benchmark_async(
    config: BenchmarkConfig, models: List[str]
) -> List[Tuple[str, float]]:
    """
    Benchmark a list of models concurrently using asyncio.

    Args:
        config: Benchmark configuration (URL, key, concurrency, timeout).
        models: List of model IDs to test.

    Returns:
        List of (model_id, latency) for responsive models, sorted by latency.
    """
    total = len(models)
    responsive: List[Tuple[str, float]] = []

    print(
        f"\n[Benchmark] Testing {total} models "
        f"(max {config.max_concurrent} concurrent) on {config.base_url}..."
    )

    connector = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=config.max_concurrent,
            max_keepalive_connections=config.max_concurrent,
            keepalive_expiry=30.0,
        ),
    )

    semaphore = asyncio.Semaphore(config.max_concurrent)

    async with connector as client:
        async def bounded_test(model_id: str) -> Tuple[str, Optional[Tuple[str, float]]]:
            """Rate-limited wrapper around _test_single_model."""
            async with semaphore:
                result = await _test_single_model(client, config, model_id)
                return (model_id, result)

        # Create all tasks
        tasks = {asyncio.create_task(bounded_test(m)): m for m in models}

        for future in asyncio.as_completed(tasks):
            model_id, result = await future
            if result is not None:
                responsive.append(result)

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
    config: BenchmarkConfig, models: List[str]
) -> List[Tuple[str, float]]:
    """
    Synchronous wrapper around run_benchmark_async.

    Handles two scenarios:
    - No running event loop: uses asyncio.run() directly.
    - Already in an event loop (e.g. Streamlit): spawns a new thread with
      its own event loop, so the benchmark doesn't block Streamlit's loop.
    """
    async def runner():
        return await run_benchmark_async(config, models)

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
