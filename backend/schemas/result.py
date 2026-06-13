"""Search result schemas for OmniFinder AI."""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single search result returned by a search tool."""

    tool_name: str = Field(..., description="Name of the tool that produced this result")
    content: str = Field(..., description="The result content from the tool")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the result (timing, URLs, etc.)",
    )


class ToolResponse(BaseModel):
    """Response from a single tool execution with success/error status."""

    tool_name: str = Field(..., description="Name of the tool executed")
    content: str = Field(..., description="Tool response content")
    success: bool = Field(default=True, description="Whether the tool executed successfully")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    execution_time: Optional[float] = Field(
        default=None, description="Time taken for execution in seconds"
    )
