"""
LLM Gateway - Centralized LLM management and configuration for OmniFinder AI.

Supports multiple providers (OpenRouter, Groq) with a unified model list.
Model selection is provider-agnostic — users see a single merged dropdown.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from groq import Groq
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter
from openai import OpenAI

from .benchmark import BenchmarkConfig, run_benchmark_sync

# Load environment variables from .env file
load_dotenv()

# Number of seconds before benchmark results are considered stale
_BENCHMARK_CACHE_TTL = 300  # 5 minutes


class LLMGateway:
    """Gateway for managing LLM configurations and model selection.

    Aggregates models from all configured providers into a single list.
    Internally tracks which provider a model belongs to so the correct
    LLM instance is created at inference time.
    """

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ):
        """
        Initialize the LLM Gateway.

        At least one provider API key must be available, otherwise
        a ValueError is raised.

        Args:
            openrouter_api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
            groq_api_key: Groq API key. Falls back to GROQ_API_KEY env var.
        """
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key and not self.groq_api_key:
            raise ValueError(
                "At least one API key is required: "
                "OPENROUTER_API_KEY and/or GROQ_API_KEY"
            )

        # OpenAI client used for OpenRouter model listing
        self._client: Optional[OpenAI] = None
        self._groq_client: Optional[Groq] = None

        # Cached model data
        self._available_models: Optional[List[str]] = None
        self._model_provider: Dict[str, str] = {}  # model_id -> "openrouter" | "groq"
        self._benchmark_results: Optional[Tuple[float, List[Tuple[str, float]]]] = None
        self._benchmark_in_progress = False
        self._last_benchmark_was_fresh = False

    @property
    def client(self) -> Optional[OpenAI]:
        """Get or create OpenAI client configured for OpenRouter."""
        if self._client is None and self.api_key:
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

    @property
    def groq_client(self) -> Optional[Groq]:
        """Get or create Groq client."""
        if self._groq_client is None and self.groq_api_key:
            self._groq_client = Groq(api_key=self.groq_api_key)
        return self._groq_client

    # ------------------------------------------------------------------
    # Model listing — merge models from all configured providers
    # ------------------------------------------------------------------

    def _fetch_openrouter_models(self) -> List[str]:
        """Fetch free models from OpenRouter API."""
        if not self.client:
            return []
        try:
            models_response = self.client.models.list()
            model_ids = []
            for model in models_response.data:
                mid = model.id
                if mid.endswith(":free"):
                    model_ids.append(mid)
                    self._model_provider[mid] = "openrouter"
            return model_ids
        except Exception as e:
            print(f"Error fetching OpenRouter models: {str(e)}")
            return []

    def _fetch_groq_models(self) -> List[str]:
        """Fetch models from Groq API."""
        gc = self.groq_client
        if not gc:
            return []
        try:
            groq_models = gc.models.list()
            model_ids = []
            for model in groq_models.data:
                mid = model.id
                model_ids.append(mid)
                self._model_provider[mid] = "groq"
            return model_ids
        except Exception as e:
            print(f"Error fetching Groq models: {str(e)}")
            return []

    def get_available_models(self) -> List[str]:
        """
        Fetch available models from all configured providers, merged into one list.

        Returns:
            Sorted list of available model IDs.
        """
        if self._available_models is not None:
            return self._available_models

        all_models = []
        all_models.extend(self._fetch_openrouter_models())
        all_models.extend(self._fetch_groq_models())

        # Exclude non-conventional models: guardrails, image gen, embeddings, audio/speech
        exclude_patterns = [
            # Guardrails / moderation
            "prompt-guard",
            "safeguard",
            # Image generation
            "flux",
            "dall-e",
            "stable-diffusion",
            "sdxl",
            # Embeddings
            "text-embedding",
            "ada",
            # Audio transcription
            "whisper",
            # Text-to-speech / audio output
            "tts",
            "orpheus",
        ]
        self._available_models = sorted(
            m for m in set(all_models)
            if not any(p in m.lower() for p in exclude_patterns)
        )
        return self._available_models

    # ------------------------------------------------------------------
    # LLM creation — provider-agnostic, routes by model_id
    # ------------------------------------------------------------------

    def create_llm(
        self,
        model: Optional[str] = None,
        temperature: float = 0,
        streaming: bool = True,
        **kwargs,
    ) -> BaseChatModel:
        """
        Create an LLM instance for the given model.

        Automatically routes to the correct provider based on the model ID.

        Args:
            model: Model ID to use. If None, uses the default model.
            temperature: Temperature for response generation
            streaming: Whether to enable streaming responses
            **kwargs: Additional arguments to pass to the LLM constructor

        Returns:
            A LangChain BaseChatModel instance (ChatOpenRouter or ChatGroq).
        """
        if model is None:
            model = self.get_default_model()
        elif not self.validate_model(model):
            print(
                f"Warning: Model {model} not found in available models. Using default."
            )
            model = self.get_default_model()

        provider = self._model_provider.get(model, "openrouter")

        if provider == "groq":
            return ChatGroq(
                model=model,
                temperature=temperature,
                api_key=self.groq_api_key,
                streaming=streaming,
                **kwargs,
            )

        return ChatOpenRouter(
            model=model,
            temperature=temperature,
            api_key=self.api_key,
            streaming=streaming,
            **kwargs,
        )

    def validate_model(self, model_id: str) -> bool:
        """Check if a model ID exists in the current available list."""
        available_models = self.get_available_models()
        return model_id in available_models

    def get_default_model(self) -> str:
        """Get the default model ID (first available)."""
        available_models = self.get_available_models()
        return (
            available_models[0]
            if available_models
            else "openai/gpt-4o-mini:free"
        )

    # ------------------------------------------------------------------
    # Benchmarking — test models from each provider separately,
    # then merge and sort results.
    # ------------------------------------------------------------------

    def benchmark_models(self, force: bool = False) -> List[Tuple[str, float]]:
        """
        Benchmark all available models and return responsive ones sorted by latency.

        Benchmarks each provider's models separately (different base URLs, keys,
        and rate limits), then merges the results into a single sorted list.

        Results are cached for _BENCHMARK_CACHE_TTL seconds.

        Args:
            force: If True, bypass cache and re-benchmark immediately.

        Returns:
            List of (model_id, latency_seconds) tuples, sorted fastest-first.
        """
        now = time.time()

        # Return cached results if still fresh
        if not force and self._benchmark_results is not None:
            cached_time, cached_data = self._benchmark_results
            if now - cached_time < _BENCHMARK_CACHE_TTL:
                self._last_benchmark_was_fresh = False
                return cached_data

        # Guard against concurrent benchmark calls
        if self._benchmark_in_progress:
            self._last_benchmark_was_fresh = False
            if self._benchmark_results is not None:
                return self._benchmark_results[1]
            return []

        self._benchmark_in_progress = True
        try:
            models = self.get_available_models()
            if not models:
                self._benchmark_results = (now, [])
                return []

            all_results: List[Tuple[str, float]] = []

            # Benchmark OpenRouter models
            or_models = [
                m for m in models
                if self._model_provider.get(m) == "openrouter"
            ]
            if or_models and self.api_key:
                or_config = BenchmarkConfig(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key,
                    max_concurrent=3,
                )
                all_results.extend(run_benchmark_sync(or_config, or_models))

            # Benchmark Groq models
            groq_models = [
                m for m in models
                if self._model_provider.get(m) == "groq"
            ]
            if groq_models and self.groq_api_key:
                groq_config = BenchmarkConfig(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.groq_api_key,
                    max_concurrent=5,
                )
                all_results.extend(run_benchmark_sync(groq_config, groq_models))

            all_results.sort(key=lambda x: x[1])
            self._benchmark_results = (now, all_results)
            self._last_benchmark_was_fresh = True
            return all_results
        finally:
            self._benchmark_in_progress = False

    def get_benchmarked_models(self) -> List[str]:
        """
        Convenience method: return just the model IDs (no latencies)
        of responsive models, sorted fastest-first.
        """
        return [m for m, _ in self.benchmark_models()]


# Global LLM Gateway instance
_llm_gateway = None


def get_llm_gateway(
    openrouter_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
) -> LLMGateway:
    """
    Get the global LLM Gateway instance.

    Args:
        openrouter_api_key: Optional OpenRouter API key.
        groq_api_key: Optional Groq API key.

    Returns:
        LLMGateway instance
    """
    global _llm_gateway
    recreate = openrouter_api_key is not None or groq_api_key is not None
    if _llm_gateway is None or recreate:
        _llm_gateway = LLMGateway(
            openrouter_api_key=openrouter_api_key,
            groq_api_key=groq_api_key,
        )
    return _llm_gateway


def create_default_llm() -> BaseChatModel:
    """Create a default LLM instance using environment configuration."""
    gateway = get_llm_gateway()
    return gateway.create_llm()


def create_llm_with_model(model_id: str) -> BaseChatModel:
    """Create an LLM instance with a specific model."""
    gateway = get_llm_gateway()
    return gateway.create_llm(model=model_id)
