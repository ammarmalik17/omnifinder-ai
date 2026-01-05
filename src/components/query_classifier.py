from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


class QueryClassification(BaseModel):
    """Classification of a user query to determine the appropriate search tools."""
    
    primary_tool: Literal["wikipedia", "arxiv", "duckduckgo", "web_search"] = Field(
        ...,
        description="Primary search tool to use based on query characteristics"
    )
    secondary_tools: list[Literal["wikipedia", "arxiv", "duckduckgo", "web_search"]] = Field(
        default=[],
        description="Additional tools that might be helpful for the query"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why these tools were selected for the query"
    )


class QueryClassifier:
    """Classifies user queries to determine appropriate search tools."""
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.classifier = self.llm.with_structured_output(QueryClassification)
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

Respond with the appropriate tool classification and reasoning."""),
            ("human", "{query}")
        ])
        self.chain = self.prompt | self.classifier
    
    def classify(self, query: str) -> QueryClassification:
        """Classify a query to determine appropriate search tools."""
        return self.chain.invoke({"query": query})