from typing import Any, List, Optional

from langchain_core.tools import BaseTool

from backend.config.agent_config import AgentConfig


class BaseSearchTool(BaseTool):
    """Base class for all search tools, providing shared constructor logic.

    All search tools (Wikipedia, ArXiv, Web) share the same __init__ signature:
    accepting an optional AgentConfig and forwarding kwargs to BaseTool.
    """

    config: AgentConfig = AgentConfig()

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs: Any):
        super().__init__(**kwargs)
        if config is not None:
            self.config = config


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
