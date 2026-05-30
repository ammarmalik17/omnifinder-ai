"""
LLM Gateway - Centralized LLM management and configuration for OmniFinder AI.

This module provides a unified interface for managing LLM configurations,
model selection, and API key management across the application.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Number of seconds before benchmark results are considered stale
_BENCHMARK_CACHE_TTL = 300  # 5 minutes

# How many concurrent benchmark requests to allow (avoid rate-limit spikes)
_BENCHMARK_MAX_CONCURRENT = 5

# Simple prompt for benchmarking (must be fast to generate)
_BENCHMARK_PROMPT = "Say exactly: ok"

# Timeout per model (seconds) — short, we're just testing "Say exactly: ok"
_BENCHMARK_TIMEOUT = 8


class LLMGateway:
    """Gateway for managing LLM configurations and model selection."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM Gateway.

        Args:
            api_key: OpenRouter API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")

        self._client = None
        self._available_models = None
        self._benchmark_results = None  # (timestamp, [(model_id, latency), ...])
        self._benchmark_in_progress = False

    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI client configured for OpenRouter."""
        if self._client is None:
            # Reuse HTTP connections across all requests via a shared httpx transport.
            # Connection reuse shaves hundreds of ms per request by avoiding TLS handshakes.
            http_client = httpx.Client(
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=50,
                    keepalive_expiry=30.0,
                ),
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=30.0,
                    write=10.0,
                    pool=10.0,
                ),
            )
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                http_client=http_client,
            )
        return self._client

    def get_available_models(self) -> List[str]:
        """
        Fetch available free models from OpenRouter API.

        Returns:
            List of available model IDs ending with ':free'
        """
        if self._available_models is not None:
            return self._available_models

        try:
            models_response = self.client.models.list()
            free_models = [
                model.id for model in models_response.data if model.id.endswith(":free")
            ]
            self._available_models = sorted(free_models)
            return self._available_models
        except Exception as e:
            print(f"Error fetching models: {str(e)}")
            return []

    def create_llm(
        self,
        model: Optional[str] = None,
        temperature: float = 0,
        streaming: bool = True,
        **kwargs,
    ) -> ChatOpenRouter:
        """
        Create an LLM instance with the specified configuration.

        Args:
            model: Model ID to use. If None, uses the first available free model.
            temperature: Temperature for response generation
            streaming: Whether to enable streaming responses
            **kwargs: Additional arguments to pass to ChatOpenRouter

        Returns:
            Configured ChatOpenRouter instance
        """
        if model is None:
            model = self.get_default_model()
        elif not self.validate_model(model):
            print(
                f"Warning: Model {model} not found in available models. Using default."
            )
            model = self.get_default_model()

        return ChatOpenRouter(
            model=model,
            temperature=temperature,
            api_key=self.api_key,
            streaming=streaming,
            **kwargs,
        )

    def validate_model(self, model_id: str) -> bool:
        """
        Validate if a model ID is available and valid.

        Args:
            model_id: Model ID to validate

        Returns:
            True if model is available, False otherwise
        """
        available_models = self.get_available_models()
        return model_id in available_models

    def get_default_model(self) -> str:
        """Get the default model ID."""
        available_models = self.get_available_models()
        return (
            available_models[0]
            if available_models
            else "arcee-ai/trinity-large-preview:free"
        )

    # ------------------------------------------------------------------
    # Benchmarking — test which free models are responsive, sorted by latency
    # Uses ThreadPoolExecutor (same pattern as SearchAgent._execute_search_tools)
    # to work reliably in Streamlit's mixed sync/async environment.
    # ------------------------------------------------------------------

    def benchmark_models(self, force: bool = False) -> List[Tuple[str, float]]:
        """
        Benchmark all available free models and return only responsive ones
        sorted by latency (fastest first).

        Uses ThreadPoolExecutor for concurrency (sync client) — avoids
        asyncio.run() issues in Streamlit's running event loop.

        Results are cached for _BENCHMARK_CACHE_TTL seconds to avoid
        hammering the API on every page load.

        Args:
            force: If True, bypass cache and re-benchmark immediately.

        Returns:
            List of (model_id, latency_seconds) tuples, sorted fastest-first.
            Only models that responded successfully are included.
        """
        now = time.time()

        # Return cached results if still fresh and not forced
        if not force and self._benchmark_results is not None:
            cached_time, cached_data = self._benchmark_results
            if now - cached_time < _BENCHMARK_CACHE_TTL:
                return cached_data

        # Guard against concurrent benchmark calls
        if self._benchmark_in_progress:
            if self._benchmark_results is not None:
                return self._benchmark_results[1]
            return []

        self._benchmark_in_progress = True
        try:
            models = self.get_available_models()
            if not models:
                self._benchmark_results = (now, [])
                return []

            results = self._run_benchmark_sync(models)

            # Cache and return
            self._benchmark_results = (now, results)
            return results
        finally:
            self._benchmark_in_progress = False

    def _run_benchmark_sync(
        self, models: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Benchmark a list of models concurrently using ThreadPoolExecutor.

        Args:
            models: List of model IDs to test.

        Returns:
            List of (model_id, latency) for responsive models, sorted by latency.
        """
        client = self.client  # Get the shared sync client
        total = len(models)
        responsive = []
        completed = 0
        lock = __import__("threading").Lock()

        def test_single(model_id: str) -> Optional[Tuple[str, float]]:
            """Test a single model. Returns None if it fails."""
            try:
                start = time.perf_counter()
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": _BENCHMARK_PROMPT}],
                    max_tokens=10,
                    temperature=0,
                    timeout=_BENCHMARK_TIMEOUT,
                )
                elapsed = time.perf_counter() - start
                # Confirm we actually got text back
                if response.choices and response.choices[0].message.content:
                    return (model_id, round(elapsed, 2))
                return None
            except Exception:
                return None

        print(f"\n[Benchmark] Testing {total} models (max {_BENCHMARK_MAX_CONCURRENT} concurrent)...")

        with ThreadPoolExecutor(max_workers=_BENCHMARK_MAX_CONCURRENT) as executor:
            future_to_model = {
                executor.submit(test_single, m): m for m in models
            }
            for future in as_completed(future_to_model):
                completed += 1
                result = future.result()
                if result is not None:
                    with lock:
                        responsive.append(result)
                    print(f"[Benchmark] ✓ {result[0]:<55} {result[1]:.1f}s  ({completed}/{total})")
                else:
                    print(f"[Benchmark] ✗ {future_to_model[future]:<55} failed     ({completed}/{total})")

        # Sort by latency (fastest first)
        responsive.sort(key=lambda x: x[1])
        print(f"[Benchmark] Done — {len(responsive)}/{total} responsive, fastest: {responsive[0][0] if responsive else 'N/A'} ({responsive[0][1]:.1f}s)" if responsive else f"[Benchmark] Done — 0/{total} responsive")
        return responsive

    def get_benchmarked_models(self) -> List[str]:
        """
        Convenience method: return just the model IDs (no latencies)
        of responsive models, sorted fastest-first.
        """
        return [m for m, _ in self.benchmark_models()]


# Global LLM Gateway instance
_llm_gateway = None


def get_llm_gateway(api_key: Optional[str] = None) -> LLMGateway:
    """
    Get the global LLM Gateway instance.

    Args:
        api_key: Optional API key. If provided, will create new instance.

    Returns:
        LLMGateway instance
    """
    global _llm_gateway
    if _llm_gateway is None or api_key is not None:
        _llm_gateway = LLMGateway(api_key)
    return _llm_gateway


def create_default_llm() -> ChatOpenRouter:
    """Create a default LLM instance using environment configuration."""
    gateway = get_llm_gateway()
    return gateway.create_llm()


def create_llm_with_model(model_id: str) -> ChatOpenRouter:
    """Create an LLM instance with a specific model."""
    gateway = get_llm_gateway()
    return gateway.create_llm(model=model_id)