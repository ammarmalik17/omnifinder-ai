import asyncio
from typing import Any, AsyncGenerator, Dict, List

from langchain_core.language_models import BaseChatModel

from backend.prompts import synthesis_prompt


class ResultSynthesizer:
    """Component to synthesize results from multiple search tools into a coherent answer."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.chain = synthesis_prompt | self.llm

    @staticmethod
    def _format_search_results(search_results: List[Dict[str, Any]]) -> str:
        """Format search results into a single string for the LLM prompt.

        Shared between sync synthesize() and async stream_synthesize_async().
        """
        formatted_results = []
        for i, result in enumerate(search_results):
            tool_name = result.get("tool_name", "Unknown Tool")
            content = result.get("content", "")
            # Ensure content is a string and handle potential None values
            if content is None:
                content = "No content returned from tool"
            elif not isinstance(content, str):
                content = str(content)
            formatted_results.append(f"Source {i+1} ({tool_name}):\n{content}\n---\n")
        return "\n".join(formatted_results)

    def synthesize(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Synthesize search results into a coherent answer.

        Args:
            query: The original user query
            search_results: List of search results from various tools

        Returns:
            A synthesized response combining all results
        """
        formatted_results_str = self._format_search_results(search_results)

        # Invoke the LLM to synthesize results
        response = self.chain.invoke({
            "query": query,
            "search_results": formatted_results_str,
        })

        return response.content

    async def stream_synthesize_async(
        self, query: str, search_results: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """
        Stream the synthesized answer using real LLM token streaming.

        Uses model.astream() to yield actual tokens as the LLM generates them,
        instead of faking it by chunking a pre-generated response.

        Args:
            query: The original user query
            search_results: List of search results from various tools

        Yields:
            Real LLM token chunks as they arrive from the API
        """
        try:
            formatted_results_str = self._format_search_results(search_results)

            # Stream real tokens from the LLM via astream()
            async for chunk in self.chain.astream({
                "query": query,
                "search_results": formatted_results_str,
            }):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            yield f"Error during synthesis streaming: {str(e)}"
