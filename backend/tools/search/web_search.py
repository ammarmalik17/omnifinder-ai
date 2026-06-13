from ddgs import DDGS
from langchain_core.tools import BaseTool

from backend.config.agent_config import AgentConfig
from backend.tools.search import BaseSearchTool


class WebSearchTool(BaseSearchTool):
    """Tool for searching the web using DuckDuckGo."""

    name: str = "web_search"
    description: str = "Search the web using DuckDuckGo. Use for current events, real-time information, and general web queries."

    config: AgentConfig = AgentConfig()

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
