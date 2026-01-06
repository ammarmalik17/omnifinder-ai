"""
Configuration module for OmniFinder AI agents.

This module defines configuration classes and settings for the search agent system,
following industry best practices for maintainability and scalability.
"""
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class SearchToolType(Enum):
    """Enumeration of available search tools."""
    WIKIPEDIA = "wikipedia"
    ARXIV = "arxiv"
    DUCKDUCKGO = "duckduckgo"
    WEB_SEARCH = "web_search"


@dataclass
class AgentConfig:
    """Configuration for the SearchAgent."""
    
    # Core settings
    model_name: str = "llama3-70b-8192"
    temperature: float = 0.1
    max_workers: int = 4
    use_react_for_complex: bool = True
    
    # Tool configuration
    enabled_tools: List[SearchToolType] = field(default_factory=lambda: [
        SearchToolType.WIKIPEDIA,
        SearchToolType.ARXIV,
        SearchToolType.DUCKDUCKGO,
        SearchToolType.WEB_SEARCH
    ])
    
    # Memory settings
    max_token_limit: int = 3000
    max_history_messages: int = 10
    
    # Search settings
    wikipedia_results: int = 5
    arxiv_max_results: int = 5
    duckduckgo_max_results: int = 5
    web_search_max_results: int = 7
    
    # ReAct settings
    max_react_iterations: int = 10
    complex_query_indicators: List[str] = field(default_factory=lambda: [
        "compare", "contrast", "analyze", "evaluate", "how does.*relate", 
        "what is the relationship", "why did", "what caused", "effects of",
        "impact of", "pros and cons", "advantages and disadvantages"
    ])
    
    def get_tool_config(self, tool_type: SearchToolType) -> Dict[str, Any]:
        """Get configuration for a specific tool."""
        tool_configs = {
            SearchToolType.WIKIPEDIA: {
                "results": self.wikipedia_results
            },
            SearchToolType.ARXIV: {
                "max_results": self.arxiv_max_results
            },
            SearchToolType.DUCKDUCKGO: {
                "max_results": self.duckduckgo_max_results
            },
            SearchToolType.WEB_SEARCH: {
                "max_results": self.web_search_max_results
            }
        }
        return tool_configs.get(tool_type, {})


@dataclass
class ConversationConfig:
    """Configuration for conversation management."""
    
    max_token_limit: int = 3000
    max_history_messages: int = 10
    summary_threshold: int = 500  # Number of tokens before summarization
    enable_summary_memory: bool = False


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    
    wikipedia_results: int = 5
    arxiv_max_results: int = 5
    duckduckgo_max_results: int = 5
    web_search_max_results: int = 7
    timeout_seconds: int = 30
    retry_attempts: int = 3


def get_default_config() -> AgentConfig:
    """Get the default agent configuration."""
    return AgentConfig()


def get_production_config() -> AgentConfig:
    """Get a production-ready configuration."""
    return AgentConfig(
        temperature=0.05,  # Lower temperature for more consistent results
        max_workers=2,     # More conservative for production
        use_react_for_complex=True,
        max_token_limit=4000,  # Higher limits for production
        max_history_messages=20
    )