from typing import Optional

from ddgs import DDGS
from langchain_core.tools import BaseTool

from src.config.agent_config import AgentConfig


class WebSearchTool(BaseTool):
    """Tool for searching the web using DuckDuckGo."""

    name: str = "web_search"
    description: str = "Search the web using DuckDuckGo. Use for current events, real-time information, and general web queries."

    config: AgentConfig = AgentConfig()

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """Initialize with optional config. Falls back to default AgentConfig."""
        super().__init__(**kwargs)
        if config is not None:
            self.config = config

    def _run(self, query: str) -> str:
        try:
            config = self.config

            with DDGS() as ddgs:
                results = list(
                    ddgs.text(query, max_results=config.web_search_max_results)
                )

            if not results:
                return f"No results found for '{query}'"

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"Title: {result['title']}\n"
                    f"Body: {result['body']}\n"
                    f"URL: {result['href']}\n"
                )

            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error performing web search: {str(e)}"
