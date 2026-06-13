"""Agent response schemas for OmniFinder AI."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .query import QueryClassification


class AgentResponse(BaseModel):
    """Typed response from SearchAgent.process_query()."""

    query: str = Field(..., description="The original user query")
    classification: Optional[QueryClassification] = Field(
        default=None, description="Query classification result"
    )
    search_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Raw search results from tools (tool_name + content dicts)",
    )
    synthesized_answer: str = Field(..., description="The final synthesized answer")
    conversational: bool = Field(
        default=False, description="Whether this was a conversational response"
    )
    needs_clarification: bool = Field(
        default=False, description="Whether clarification was requested from the user"
    )
    intent_handled: Optional[str] = Field(
        default=None, description="Which conversational intent was handled"
    )
    react_steps: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="ReAct reasoning steps for UI display"
    )
    react_iterations: Optional[int] = Field(
        default=None, description="Number of ReAct iterations performed"
    )
