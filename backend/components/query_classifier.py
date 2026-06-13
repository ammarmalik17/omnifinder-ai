import asyncio
import json
import re
from typing import AsyncGenerator, List

from langchain_core.language_models import BaseChatModel

from backend.prompts import classifier_prompt, fallback_classifier_prompt
from backend.schemas import QueryClassification


class QueryClassifier:
    """Classifies user queries to determine appropriate search tools using structured output."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        # Initialize structured output classifier (industry standard approach)
        self.structured_llm = self.llm.with_structured_output(
            QueryClassification,
            method="json_schema",  # Use JSON Schema method for OpenRouter compatibility
        )

        # Create chain with structured output (prompts imported from backend.prompts)
        self.chain = classifier_prompt | self.structured_llm

        # Fallback chain for models that don't support json_schema structured output
        # (e.g. some Groq models). Uses text-based JSON extraction instead.
        self._fallback_chain = fallback_classifier_prompt | self.llm

    def classify(self, query: str) -> QueryClassification:
        """Classify a query to determine appropriate search tools.

        Uses structured output (industry standard) for type-safe, validated classification.
        Falls back to text-based JSON extraction for models that don't support
        json_schema structured output (e.g. some Groq models)
        or return empty responses.

        Args:
            query: The user's query string

        Returns:
            QueryClassification object with tool selection, reasoning, and confidence score
        """
        # Primary path: structured output with Pydantic validation
        try:
            result = self.chain.invoke({"query": query})
            return result
        except Exception as e:
            error_str = str(e)
            if "json_schema" not in error_str and "response_format" not in error_str:
                raise

        # Fallback path: retry up to 2 times with text-based JSON
        for attempt in range(2):
            response = self._fallback_chain.invoke({"query": query})
            raw = response.content if hasattr(response, "content") else str(response)
            result = self._parse_fallback_json(raw)
            if result is not None:
                return result

        # Last resort: return a reasonable default
        return QueryClassification(
            intent_type="search",
            primary_tool="web_search",
            secondary_tools=[],
            reasoning="Fallback default - model returned empty or invalid JSON",
            confidence=0.5,
            needs_clarification=False,
            is_compound=False,
            sub_queries=[],
        )

    @staticmethod
    def _parse_fallback_json(raw: str) -> QueryClassification | None:
        """Parse JSON from a fallback LLM response, handling common edge cases.

        Returns a QueryClassification if parsing succeeds, None otherwise.
        """
        if not raw or not raw.strip():
            return None

        cleaned = raw.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline:]
            closing = cleaned.rfind("```")
            if closing != -1:
                cleaned = cleaned[:closing]
            cleaned = cleaned.strip()

        # Try direct JSON parsing first
        try:
            return QueryClassification(**json.loads(cleaned))
        except (json.JSONDecodeError, Exception):
            pass

        # Fallback: use regex to find a JSON object anywhere in the text
        brace_match = re.search(r"\{[^{}]*((\{[^{}]*\}[^{}]*)*)\}[^{}]*\}", cleaned)
        if not brace_match:
            brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                return QueryClassification(**json.loads(brace_match.group()))
            except (json.JSONDecodeError, Exception):
                pass

        return None

    async def stream_classify(self, query: str) -> AsyncGenerator[str, None]:
        """
        Stream the classification process with progressive rendering.

        Args:
            query: The user's query string

        Yields:
            Chunks of the classification reasoning as it's being processed
        """
        try:
            # Stream the classification reasoning
            yield "🔍 Analyzing query intent...\n\n"
            await asyncio.sleep(0.1)

            # Stream the intent detection process
            yield "🧠 Detecting intent type...\n\n"
            await asyncio.sleep(0.1)

            # Get the classification result
            classification = self.classify(query)

            # Stream the classification details
            yield f"📊 **Intent Type**: {classification.intent_type}\n\n"
            await asyncio.sleep(0.05)

            if classification.intent_type == "conversational":
                yield f"💬 **Conversational Intent**: {classification.conversational_intent}\n\n"
                yield f"💡 **Reasoning**: {classification.reasoning}\n\n"
                yield f"🎯 **Confidence**: {classification.confidence:.2f}\n\n"
            else:
                yield f"🔍 **Primary Tool**: {classification.primary_tool}\n\n"
                if classification.secondary_tools:
                    yield f"🔧 **Secondary Tools**: {', '.join(classification.secondary_tools)}\n\n"

                if classification.is_compound:
                    yield f"📋 **Compound Query Detected**: {len(classification.sub_queries)} sub-questions\n\n"
                    for i, sq in enumerate(classification.sub_queries, 1):
                        yield f"   {i}. {sq}\n"
                    yield "\n"

                yield f"💡 **Reasoning**: {classification.reasoning}\n\n"
                yield f"🎯 **Confidence**: {classification.confidence:.2f}\n\n"

                if classification.needs_clarification:
                    yield "❓ **Note**: Query needs clarification due to low confidence.\n\n"

            yield "✅ Classification complete!\n\n"

        except Exception as e:
            yield f"❌ Error during classification: {str(e)}"
