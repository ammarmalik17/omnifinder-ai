from .wikipedia import WikipediaSearchTool
from .arxiv import ArxivSearchTool
from .web_search import WebSearchTool


def get_all_tools():
    """Returns all search tools."""
    return [
        WikipediaSearchTool(),
        ArxivSearchTool(),
        WebSearchTool()
    ]