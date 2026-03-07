"""
Conversational Response Handler for OmniFinder AI.

Handles greetings, small talk, thanks, and other conversational intents
without triggering unnecessary web searches.
"""
from typing import Dict, Any
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate


class ConversationalHandler:
    """Handles conversational intents with appropriate responses."""
    
    def __init__(self, llm: ChatOpenRouter):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are OmniFinder AI, a friendly and helpful search assistant.
            
Respond naturally and conversationally to the user's message.

For **greetings** (hello, hi, good morning):
- Respond warmly and offer assistance
- Example: "Hello! How can I help you today?"

For **small talk** (how are you, what's up):
- Respond briefly and pivot to offering help
- Example: "I'm doing well, thank you! Is there anything I can help you with?"

For **thanks** (thank you, thanks):
- Acknowledge graciously and offer further assistance
- Example: "You're welcome! Feel free to ask if you need anything else."

For **farewells** (goodbye, bye):
- Respond politely and invite return
- Example: "Goodbye! Have a great day and come back anytime!"

For **help requests** (can you help me):
- Express willingness to help and ask for specifics
- Example: "Of course! I'd be happy to help. What would you like assistance with?"

Keep responses concise, friendly, and natural. Avoid being overly verbose."""),
            ("human", "{message}")
        ])
        self.chain = self.prompt | self.llm
    
    def handle(self, intent_type: str, query: str) -> Dict[str, Any]:
        """Generate appropriate conversational response.
        
        Args:
            intent_type: Type of conversational intent (greeting, small_talk, thanks, farewell, help_request)
            query: The user's original message
            
        Returns:
            Dictionary with response and metadata
        """
        # Generate conversational response
        response = self.chain.invoke({"message": query})
        
        # Customize based on intent type
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "response": response_text,
            "intent_handled": intent_type,
            "requires_no_search": True,
            "metadata": {
                "conversational": True,
                "intent_type": intent_type
            }
        }


def get_conversational_response_templates() -> Dict[str, str]:
    """Return template responses for common conversational intents.
    
    These can be used as fallbacks or for faster response without LLM call.
    """
    return {
        "greeting": "Hello! 👋 How can I help you today? I can search Wikipedia, academic papers, or the web for information.",
        "small_talk": "I'm doing well, thank you! Is there anything I can help you with? I can answer questions using various search tools.",
        "thanks": "You're welcome! 😊 Feel free to ask if you need anything else!",
        "farewell": "Goodbye! Have a great day! Come back anytime you need help with research or information.",
        "help_request": "Of course! I'd be happy to help. What would you like assistance with? You can ask me to search for information from various sources."
    }
