"""Query classification schemas for OmniFinder AI."""
from typing import List, Literal

from pydantic import BaseModel, Field


class QueryClassification(BaseModel):
    """Classification of a user query with multi-intent detection and confidence scoring."""

    # Intent type: conversational or search-based
    intent_type: Literal["conversational", "search"] = Field(
        ...,
        description="Type of intent: 'conversational' for greetings/small talk, 'search' for knowledge queries",
    )

    # Conversational intents (when intent_type is "conversational")
    conversational_intent: Literal[
        "greeting", "small_talk", "thanks", "farewell", "help_request", None
    ] = Field(
        default=None,
        description="Specific conversational intent: greeting (hello/hi), small_talk (how are you), thanks (thank you), farewell (goodbye), help_request (help)",
    )

    # Search intents (when intent_type is "search")
    primary_tool: Literal["wikipedia", "arxiv", "web_search", None] = Field(
        default=None,
        description="Primary search tool (only for search intents). Choose from: wikipedia (general knowledge), arxiv (academic papers), web_search (current events/news)",
    )
    secondary_tools: List[Literal["wikipedia", "arxiv", "web_search"]] = Field(
        default=[],
        description="Additional search tools that might be helpful (can be empty)",
    )

    # Common fields
    reasoning: str = Field(
        ...,
        description="Brief explanation citing specific query terms or characteristics",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0). High >0.8 for clear intents, low <0.5 for ambiguous queries",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if confidence is low (<0.5) or query is ambiguous - should ask clarifying question",
    )

    # Compound query detection
    is_compound: bool = Field(
        default=False,
        description="True if the query contains multiple distinct sub-questions requiring different tools",
    )
    sub_queries: List[str] = Field(
        default=[],
        description="Decomposed sub-questions when is_compound=True, each targeting a specific search tool",
    )

    class Config:
        json_schema_extra = {"additionalProperties": False}
