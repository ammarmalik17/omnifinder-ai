from typing import List, Optional

from langchain_core.tools import BaseTool

from backend.config.agent_config import AgentConfig

from .arxiv import ArxivSearchTool
from .web_search import WebSearchTool
from .wikipedia import WikipediaSearchTool


def get_all_tools(config: Optional[AgentConfig] = None) -> List[BaseTool]:
    """Returns all search tools, optionally configured with an AgentConfig.

    Args:
        config: Optional AgentConfig to pass to each tool instance.
               If None, tools will use default AgentConfig values.
    """
    return [
        WikipediaSearchTool(config=config),
        ArxivSearchTool(config=config),
        WebSearchTool(config=config),
    ]
