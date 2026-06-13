import asyncio
from typing import Any, AsyncGenerator, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel


class ResultSynthesizer:
    """Component to synthesize results from multiple search tools into a coherent answer."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research synthesizer. Your task is to combine information from multiple sources into a coherent, well-structured answer.

Guidelines:
1. Organize information logically, grouping related points together
2. Identify and highlight the most important/relevant information
3. Resolve any contradictions between sources (if any)
4. Maintain source attribution when possible
5. Structure the response with clear sections
6. If sources disagree, present different perspectives with their respective sources
7. Include relevant links when available"""),
            ("human", """Original Query: {query}

Search Results:
{search_results}

Please synthesize these results into a comprehensive answer to the original query. Make sure to:
- Provide a clear, informative response
- Attribute information to sources where possible
- Include relevant URLs from the search results
- Structure the information logically"""),
        ])
        self.chain = self.synthesis_prompt | self.llm

    def synthesize(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Synthesize search results into a coherent answer.

        Args:
            query: The original user query
            search_results: List of search results from various tools

        Returns:
            A synthesized response combining all results
        """
        # Format search results for the prompt
        formatted_results = []
        for i, result in enumerate(search_results):
            tool_name = result.get('tool_name', 'Unknown Tool')
            content = result.get('content', '')
            # Ensure content is a string and handle potential None values
            if content is None:
                content = "No content returned from tool"
            elif not isinstance(content, str):
                content = str(content)
            formatted_results.append(f"Source {i+1} ({tool_name}):\n{content}\n---\n")

        formatted_results_str = "\n".join(formatted_results)

        # Invoke the LLM to synthesize results
        response = self.chain.invoke({
            "query": query,
            "search_results": formatted_results_str
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
            # Format search results for the prompt (same as synthesize())
            formatted_results = []
            for i, result in enumerate(search_results):
                tool_name = result.get('tool_name', 'Unknown Tool')
                content = result.get('content', '')
                if content is None:
                    content = "No content returned from tool"
                elif not isinstance(content, str):
                    content = str(content)
                formatted_results.append(f"Source {i+1} ({tool_name}):\n{content}\n---\n")

            formatted_results_str = "\n".join(formatted_results)

            # Stream real tokens from the LLM via astream()
            async for chunk in self.chain.astream({
                "query": query,
                "search_results": formatted_results_str
            }):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            yield f"❌ Error during synthesis streaming: {str(e)}"
