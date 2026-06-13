"""Synthesis prompt for OmniFinder AI.

Separates the result-synthesis prompt from the synthesis logic,
allowing prompt tuning without touching code.
"""
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are an expert research synthesizer. Your task is to combine information from multiple sources into a coherent, well-structured answer.

Guidelines:
1. Organize information logically, grouping related points together
2. Identify and highlight the most important/relevant information
3. Resolve any contradictions between sources (if any)
4. Maintain source attribution when possible
5. Structure the response with clear sections
6. If sources disagree, present different perspectives with their respective sources
7. Include relevant links when available"""

HUMAN_PROMPT = """Original Query: {query}

Search Results:
{search_results}

Please synthesize these results into a comprehensive answer to the original query. Make sure to:
- Provide a clear, informative response
- Attribute information to sources where possible
- Include relevant URLs from the search results
- Structure the information logically"""

synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])
