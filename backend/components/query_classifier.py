import asyncio
import json
import re
from typing import AsyncGenerator, List, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field


class QueryClassification(BaseModel):
    """Classification of a user query with multi-intent detection and confidence scoring."""

    # Intent type: conversational or search-based
    intent_type: Literal["conversational", "search"] = Field(
        ...,
        description="Type of intent: 'conversational' for greetings/small talk, 'search' for knowledge queries"
    )

    # Conversational intents (when intent_type is "conversational")
    conversational_intent: Literal["greeting", "small_talk", "thanks", "farewell", "help_request", None] = Field(
        default=None,
        description="Specific conversational intent: greeting (hello/hi), small_talk (how are you), thanks (thank you), farewell (goodbye), help_request (help)"
    )

    # Search intents (when intent_type is "search")
    primary_tool: Literal["wikipedia", "arxiv", "web_search", None] = Field(
        default=None,
        description="Primary search tool (only for search intents). Choose from: wikipedia (general knowledge), arxiv (academic papers), web_search (current events/news)"
    )
    secondary_tools: List[Literal["wikipedia", "arxiv", "web_search"]] = Field(
        default=[],
        description="Additional search tools that might be helpful (can be empty)"
    )

    # Common fields
    reasoning: str = Field(
        ...,
        description="Brief explanation citing specific query terms or characteristics"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0). High >0.8 for clear intents, low <0.5 for ambiguous queries"
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if confidence is low (<0.5) or query is ambiguous - should ask clarifying question"
    )

    # Compound query detection
    is_compound: bool = Field(
        default=False,
        description="True if the query contains multiple distinct sub-questions requiring different tools"
    )
    sub_queries: List[str] = Field(
        default=[],
        description="Decomposed sub-questions when is_compound=True, each targeting a specific search tool"
    )

    class Config:
        json_schema_extra = {
            "additionalProperties": False
        }


class QueryClassifier:
    """Classifies user queries to determine appropriate search tools using structured output."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        # Initialize structured output classifier (industry standard approach)
        self.structured_llm = self.llm.with_structured_output(
            QueryClassification,
            method="json_schema"  # Use JSON Schema method for OpenRouter compatibility
        )

        # Create the classification prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at detecting user intent and classifying queries.

## STEP 1: Detect Intent Type

First, determine if this is a CONVERSATIONAL or SEARCH intent:

### CONVERSATIONAL Intents (do NOT search):
- **greeting**: hello, hi, hey, good morning, greetings
- **small_talk**: how are you, what's up, how's it going, nice to meet you
- **thanks**: thank you, thanks, appreciate it, grateful
- **farewell**: goodbye, bye, see you later, take care
- **help_request**: help, can you help me, I need assistance (without specific topic)

### SEARCH Intents (require knowledge retrieval):
- **Academic**: research, paper, study, methodology, academic, scholarly, peer-reviewed, journal, conference, proceedings → use arxiv
- **Current events/news**: today, yesterday, recently, latest, breaking, news, current, recent, announced, happened → use web_search
- **Technical**: how to, tutorial, documentation, technical, guide, implementation, code, development → use web_search
- **General knowledge**: what is, who is, explain, describe, definition, meaning, history → use wikipedia

## STEP 2: Detect Compound Queries

Check if the query contains MULTIPLE distinct sub-questions that need different tools:
- Multiple question marks: "What is X? Find papers on Y."
- Numbered or bullet-pointed lists: "1. Explain RAG. 2. Find latest papers."
- Multiple topics connected by "and" requiring different domains: "Compare X and Y"
- Query has several unrelated information needs

If compound:
- Set is_compound=true
- Break the query into sub_queries, each as a complete search query
- Set secondary_tools based on the tools needed for all sub-queries
- Set needs_clarification=false (we can handle this)

## STEP 3: Apply Guardrails

Before finalizing classification:
- If confidence < 0.5 OR query is ambiguous/vague → set needs_clarification=true
- For conversational intents → confidence should be high (>0.8) for clear greetings
- For unclear queries like "tell me about stuff" → needs_clarification=true

## STEP 4: Response Guidelines

For CONVERSATIONAL intents:
- Set intent_type="conversational"
- Set appropriate conversational_intent (greeting, small_talk, etc.)
- Set primary_tool=null (NO SEARCH NEEDED)
- Provide friendly reasoning

For SEARCH intents:
- Set intent_type="search"
- Select appropriate primary_tool based on query type
- Add secondary_tools if multiple sources would help
- Provide specific reasoning citing query terms
- If compound, set is_compound=true and populate sub_queries

Respond with structured classification including all fields."""),
            ("human", "{query}")
        ])

        # Create chain with structured output
        self.chain = self.prompt | self.structured_llm

        # Fallback chain for models that don't support json_schema structured output
        # (e.g. some Groq models). Uses text-based JSON extraction instead.
        self._fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at detecting user intent and classifying queries.

Follow these steps exactly:

## STEP 1: Detect Intent Type

CONVERSATIONAL Intents (do NOT search): greeting, small_talk, thanks, farewell, help_request

SEARCH Intents (require knowledge retrieval):
- arxiv for academic papers
- web_search for current events, news, technical topics
- wikipedia for general knowledge, definitions

## STEP 2: Detect Compound Queries

Check for multiple distinct sub-questions that need different tools.
If compound, decompose into sub_queries.

## STEP 3: Respond with ONLY valid JSON

{{
  "intent_type": "conversational" or "search",
  "conversational_intent": "greeting" or "small_talk" etc. or null,
  "primary_tool": "wikipedia" or "arxiv" or "web_search" or null,
  "secondary_tools": ["tool_name", ...] or [],
  "reasoning": "brief explanation",
  "confidence": 0.0-1.0,
  "needs_clarification": true or false,
  "is_compound": true or false,
  "sub_queries": ["sub-query 1", "sub-query 2"] or []
}}

Output ONLY the JSON object. No markdown, no code fences, no extra text."""),
            ("human", "{query}")
        ])
        self._fallback_chain = self._fallback_prompt | self.llm

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
