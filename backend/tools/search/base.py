from typing import Any, Optional

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
