"""
Model Benchmarking Tool for OmniFinder AI (async).

Tests each available free model concurrently using asyncio for:
- Availability (is it responding?)
- Latency (how fast is the first token?)
"""
import asyncio
import os
import time
from typing import List, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

TEST_PROMPT = "Say exactly: ok"
TIMEOUT_SECONDS = 30
MAX_CONCURRENT = 5  # Limit concurrent requests to avoid rate-limit spikes


async def test_model(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model_id: str,
) -> Tuple[str, float | None, str | None]:
    """
    Test a single model's availability and latency using asyncio.

    Returns:
        Tuple of (model_id, latency_in_seconds_or_None, error_message_or_None)
    """
    async with sem:
        print(f"  Testing {model_id} ... ", end="", flush=True)
        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=10,
                temperature=0,
                timeout=TIMEOUT_SECONDS,
            )
            elapsed = time.perf_counter() - start
            content = response.choices[0].message.content or ""
            print(f"✓ {elapsed:.1f}s")
            return model_id, round(elapsed, 2), None
        except Exception as e:
            elapsed = time.perf_counter() - start
            error_msg = str(e)[:120]
            print(f"✗ {error_msg}")
            return model_id, None, error_msg


async def main():
    print("=" * 60)
    print("OmniFinder AI - Model Benchmark (async)")
    print("=" * 60)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment or .env file")
        return

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print("\nFetching available free models...")
    models_response = await client.models.list()
    models = sorted(
        [model.id for model in models_response.data if model.id.endswith(":free")]
    )
    print(f"Found {len(models)} free models\n")

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    print(f"Benchmarking (max {MAX_CONCURRENT} concurrent requests)...\n")
    tasks = [test_model(client, sem, model) for model in models]
    results = await asyncio.gather(*tasks)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    responsive = [(m, l) for m, l, e in results if l is not None]
    failed = [(m, e) for m, l, e in results if l is None]

    responsive.sort(key=lambda x: x[1])  # Sort by latency

    print(f"\n✓ Responsive models ({len(responsive)}):")
    print(f"   {'Model':<55} {'Latency':<10}")
    print(f"   {'-'*55} {'-'*10}")
    for model, latency in responsive:
        print(f"   {model:<55} {latency:<8.1f}s")

    if failed:
        print(f"\n✗ Failed models ({len(failed)}):")
        for model, error in failed:
            print(f"   {model:<55} {error or 'unknown error'}")

    print(f"\n{'─'*60}")
    print(f"Total models: {len(results)}")
    print(f"Responsive:   {len(responsive)}")
    print(f"Failed:       {len(failed)}")


if __name__ == "__main__":
    asyncio.run(main())