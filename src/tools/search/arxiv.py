import arxiv
from langchain_core.tools import BaseTool

from src.config.agent_config import AgentConfig


class ArxivSearchTool(BaseTool):
    """Tool for searching academic papers on Arxiv."""

    name: str = "arxiv"
    description: str = "Search for academic papers on Arxiv. Use for research papers and academic queries."

    def _run(self, query: str) -> str:
        try:
            # Use default config values if no config is passed
            config = getattr(self, "config", AgentConfig())

            # Search for papers
            search = arxiv.Search(
                query=query,
                max_results=config.arxiv_max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            results = []
            for result in search.results():
                results.append(
                    {
                        "title": result.title,
                        "summary": result.summary[:500] + "..."
                        if len(result.summary) > 500
                        else result.summary,
                        "authors": [author.name for author in result.authors],
                        "published": result.published.strftime("%Y-%m-%d"),
                        "url": result.entry_id,
                        "pdf_url": result.pdf_url
                        if result.pdf_url
                        else "No PDF available",
                    }
                )

            if not results:
                return f"No arXiv papers found for '{query}'"

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"Title: {result['title']}\n"
                    f"Authors: {', '.join(result['authors'])}\n"
                    f"Published: {result['published']}\n"
                    f"Summary: {result['summary']}\n"
                    f"URL: {result['url']}\n"
                    f"PDF: {result['pdf_url']}\n"
                )

            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error searching arXiv: {str(e)}"
