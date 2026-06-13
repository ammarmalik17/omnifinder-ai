"""Prompt templates for OmniFinder AI - centralized prompt engineering."""
from .classifier_prompt import classifier_prompt, fallback_classifier_prompt
from .conversational_prompt import conversational_prompt
from .react_prompt import react_prompt, build_tool_descriptions
from .synthesis_prompt import synthesis_prompt

__all__ = [
    "classifier_prompt",
    "fallback_classifier_prompt",
    "conversational_prompt",
    "react_prompt",
    "build_tool_descriptions",
    "synthesis_prompt",
]
