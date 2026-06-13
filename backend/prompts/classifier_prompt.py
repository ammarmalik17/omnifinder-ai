"""Classifier prompts for OmniFinder AI.

Separates prompt engineering from classification logic,
allowing prompt tuning without touching code.
"""
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------------------------
# Primary structured-output classifier prompt
# ------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = """You are an expert at detecting user intent and classifying queries.

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

Respond with structured classification including all fields."""

classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", CLASSIFIER_SYSTEM_PROMPT),
    ("human", "{query}"),
])

# ------------------------------------------------------------------
# Fallback text-based prompt (for models without json_schema support)
# ------------------------------------------------------------------

FALLBACK_SYSTEM_PROMPT = """You are an expert at detecting user intent and classifying queries.

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

Output ONLY the JSON object. No markdown, no code fences, no extra text."""

fallback_classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", FALLBACK_SYSTEM_PROMPT),
    ("human", "{query}"),
])
