"""
LLM Gateway - Centralized LLM management and configuration for OmniFinder AI.

This module provides a unified interface for managing LLM configurations, 
model selection, and API key management across the application.
"""

import os
from typing import Optional, List, Dict, Any
from langchain_openrouter import ChatOpenRouter
from openai import OpenAI


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
    
    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI client configured for OpenRouter."""
        if self._client is None:
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
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
                model.id for model in models_response.data 
                if model.id.endswith(':free')
            ]
            self._available_models = sorted(free_models)
            return self._available_models
        except Exception as e:
            print(f"Error fetching models: {str(e)}")
            # Return empty list - let get_default_model handle the fallback
            return []
    
    def create_llm(
        self, 
        model: Optional[str] = None,
        temperature: float = 0,
        streaming: bool = True,
        **kwargs
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
        # Use first available model if none specified
        if model is None:
            model = self.get_default_model()
        # Validate model if provided
        elif not self.validate_model(model):
            print(f"Warning: Model {model} not found in available models. Using default.")
            model = self.get_default_model()
            
        return ChatOpenRouter(
            model=model,
            temperature=temperature,
            api_key=self.api_key,
            streaming=streaming,
            **kwargs
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
        return available_models[0] if available_models else "arcee-ai/trinity-large-preview:free"


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
    """
    Create a default LLM instance using environment configuration.
    
    Returns:
        Configured ChatOpenRouter instance
    """
    gateway = get_llm_gateway()
    return gateway.create_llm()


def create_llm_with_model(model_id: str) -> ChatOpenRouter:
    """
    Create an LLM instance with a specific model.
    
    Args:
        model_id: The model ID to use for the LLM
        
    Returns:
        Configured ChatOpenRouter instance
    """
    gateway = get_llm_gateway()
    return gateway.create_llm(model=model_id)
