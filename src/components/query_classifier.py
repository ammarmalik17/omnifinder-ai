from typing import Literal, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter


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
    
    class Config:
        json_schema_extra = {
            "additionalProperties": False
        }


class QueryClassifier:
    """Classifies user queries to determine appropriate search tools using structured output."""
    
    def __init__(self, llm: ChatOpenRouter):
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

## STEP 2: Apply Guardrails

Before finalizing classification:
- If confidence < 0.5 OR query is ambiguous/vague → set needs_clarification=true
- For conversational intents → confidence should be high (>0.8) for clear greetings
- For unclear queries like "tell me about stuff" → needs_clarification=true

## STEP 3: Response Guidelines

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

Respond with structured classification including all fields."""),
            ("human", "{query}")
        ])
        
        # Create chain with structured output
        self.chain = self.prompt | self.structured_llm
    
    def classify(self, query: str) -> QueryClassification:
        """Classify a query to determine appropriate search tools.
        
        Uses structured output (industry standard) for type-safe, validated classification.
        
        Args:
            query: The user's query string
            
        Returns:
            QueryClassification object with tool selection, reasoning, and confidence score
            
        Note:
            This method uses structured output which requires model support.
            If the model doesn't support structured output, an error will be raised.
        """
        # Industry standard approach - structured output with Pydantic validation
        result = self.chain.invoke({"query": query})
        # Structured output returns already-parsed Pydantic model with validation
        return result