import warnings

import wikipedia
from bs4 import GuessedAtParserWarning
from langchain_core.tools import BaseTool

from src.config.agent_config import AgentConfig

# Suppress the BeautifulSoup warning about parser guessing
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)


class WikipediaSearchTool(BaseTool):
    """Tool for searching Wikipedia."""

    name: str = "wikipedia"
    description: str = (
        "Search for information on Wikipedia. Use for general knowledge queries."
    )

    def _run(self, query: str) -> str:
        try:
            # Use default config values if no config is passed
            config = getattr(self, "config", AgentConfig())

            # Search for pages
            search_results = wikipedia.search(query, results=config.wikipedia_results)
            if not search_results:
                return f"No Wikipedia pages found for '{query}'"

            # Get summaries for top results
            results = []
            for page_title in search_results[:3]:  # Get top 3 results
                try:
                    summary = wikipedia.summary(page_title, sentences=3)
                    results.append(
                        {
                            "title": page_title,
                            "summary": summary,
                            "url": wikipedia.page(page_title).url,
                        }
                    )
                except wikipedia.exceptions.DisambiguationError as e:
                    # If there's a disambiguation, pick the first option
                    if e.options:
                        summary = wikipedia.summary(e.options[0], sentences=3)
                        results.append(
                            {
                                "title": e.options[0],
                                "summary": summary,
                                "url": wikipedia.page(e.options[0]).url,
                            }
                        )
                except wikipedia.exceptions.PageError:
                    continue
                except Exception as e:
                    return f"Error searching Wikipedia: {str(e)}"

            if not results:
                return f"No detailed Wikipedia pages found for '{query}'"

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"Title: {result['title']}\n"
                    f"Summary: {result['summary']}\n"
                    f"URL: {result['url']}\n"
                )

            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"
