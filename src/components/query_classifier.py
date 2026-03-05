from typing import Literal, List
import json
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter


class QueryClassification(BaseModel):
    """Classification of a user query to determine the appropriate search tools."""
    
    primary_tool: Literal["wikipedia", "arxiv", "duckduckgo", "web_search"] = Field(
        ...,
        description="Primary search tool to use based on query characteristics"
    )
    secondary_tools: List[Literal["wikipedia", "arxiv", "duckduckgo", "web_search"]] = Field(
        default=[],
        description="Additional tools that might be helpful for the query"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why these tools were selected for the query"
    )


class QueryClassifier:
    """Classifies user queries to determine appropriate search tools."""
    
    def __init__(self, llm: ChatOpenRouter):
        self.llm = llm
        
        # Use text-based classification to avoid tool calling requirements
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at classifying user queries to determine the most appropriate search tools.
            
Your task is to analyze the query and select the best search tools based on these criteria:
- Academic queries (containing terms like 'research', 'paper', 'study', 'methodology', 'academic', 'scholarly', 'peer-reviewed', 'journal') → prioritize arxiv
- Current events and news queries (time-sensitive language, recent dates, breaking news) → prioritize duckduckgo
- Technical documentation and multi-domain queries → use web_search as primary or supplementary
- General knowledge queries → prioritize wikipedia

Consider these indicators:
- Academic: research, paper, study, methodology, academic, scholarly, peer-reviewed, journal, conference, proceedings
- Current events: today, yesterday, recently, latest, breaking, news, current, recent, announced, happened
- Technical: how to, tutorial, documentation, technical, guide, implementation, code, development
- General knowledge: what is, who is, explain, describe, definition, meaning, history

Available tools: wikipedia, arxiv, duckduckgo, web_search

Respond ONLY with a JSON object in this exact format:
{{
  "primary_tool": "tool_name",
  "secondary_tools": ["tool1", "tool2"],
  "reasoning": "your reasoning here"
}}

Do not include any other text. Just the JSON."""),
            ("human", "{query}")
        ])
        self.chain = self.prompt | self.llm
    
    def classify(self, query: str) -> QueryClassification:
        """Classify a query to determine appropriate search tools."""
        # Get the LLM response
        response = self.chain.invoke({"query": query})
        
        # Extract the content from the response
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse the JSON response
        try:
            # Find JSON in the response (in case there's extra text)
            import re
            json_match = re.search(r'\{.*?\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = response_content
            
            data = json.loads(json_str)
            
            # Create and return the QueryClassification object
            return QueryClassification(
                primary_tool=data["primary_tool"],
                secondary_tools=data.get("secondary_tools", []),
                reasoning=data["reasoning"]
            )
        except Exception as e:
            # Fallback to default classification on parsing error
            print(f"Warning: Failed to parse classification response: {e}")
            print(f"Raw response: {response_content}")
            return QueryClassification(
                primary_tool="web_search",
                secondary_tools=[],
                reasoning="Fallback classification due to parsing error"
            )