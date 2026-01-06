from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
import wikipedia
import arxiv
from duckduckgo_search import DDGS
import requests
from pydantic import BaseModel, Field


class WikipediaSearchTool(BaseTool):
    """Tool for searching Wikipedia."""
    
    name = "wikipedia"
    description = "Search for information on Wikipedia. Use for general knowledge queries."
    
    def _run(self, query: str) -> str:
        try:
            # Search for pages
            search_results = wikipedia.search(query, results=5)
            if not search_results:
                return f"No Wikipedia pages found for '{query}'"
            
            # Get summaries for top results
            results = []
            for page_title in search_results[:3]:  # Get top 3 results
                try:
                    summary = wikipedia.summary(page_title, sentences=3)
                    results.append({
                        "title": page_title,
                        "summary": summary,
                        "url": wikipedia.page(page_title).url
                    })
                except wikipedia.exceptions.DisambiguationError as e:
                    # If there's a disambiguation, pick the first option
                    if e.options:
                        summary = wikipedia.summary(e.options[0], sentences=3)
                        results.append({
                            "title": e.options[0],
                            "summary": summary,
                            "url": wikipedia.page(e.options[0]).url
                        })
                except wikipedia.exceptions.PageError:
                    continue
                except Exception as e:
                    # Handle other Wikipedia errors
                    continue
            
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


class ArxivSearchTool(BaseTool):
    """Tool for searching academic papers on Arxiv."""
    
    name = "arxiv"
    description = "Search for academic papers on Arxiv. Use for research papers and academic queries."
    
    def _run(self, query: str) -> str:
        try:
            # Search for papers
            search = arxiv.Search(
                query=query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for result in search.results():
                results.append({
                    "title": result.title,
                    "summary": result.summary[:500] + "..." if len(result.summary) > 500 else result.summary,
                    "authors": [author.name for author in result.authors],
                    "published": result.published.strftime('%Y-%m-%d'),
                    "url": result.entry_id,
                    "pdf_url": result.pdf_url if result.pdf_url else "No PDF available"
                })
            
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


class DuckDuckGoSearchTool(BaseTool):
    """Tool for searching with DuckDuckGo."""
    
    name = "duckduckgo"
    description = "Search the web using DuckDuckGo. Use for current events and real-time information."
    
    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            
            if not results:
                return f"No results found for '{query}' on DuckDuckGo"
            
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
            return f"Error searching DuckDuckGo: {str(e)}"


class WebSearchTool(BaseTool):
    """Tool for general web search using a search API."""
    
    name = "web_search"
    description = "General web search for comprehensive information across multiple domains."
    
    def _run(self, query: str) -> str:
        # For this implementation, we'll use DuckDuckGo as the web search provider
        # In a production environment, you might use Google Custom Search, SerpAPI, etc.
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=7))
            
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


def get_all_tools():
    """Returns all search tools."""
    return [
        WikipediaSearchTool(),
        ArxivSearchTool(),
        DuckDuckGoSearchTool(),
        WebSearchTool()
    ]