"""
Conversational Response Handler for OmniFinder AI.

Handles greetings, small talk, thanks, and other conversational intents
without triggering unnecessary web searches.
"""

from typing import Any, AsyncGenerator, Dict

from langchain_core.language_models import BaseChatModel

from backend.prompts import conversational_prompt


class ConversationalHandler:
    """Handles conversational intents with appropriate responses."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.chain = conversational_prompt | self.llm

    def handle(self, intent_type: str, query: str) -> Dict[str, Any]:
        """Generate appropriate conversational response (synchronous, for memory storage).

        Args:
            intent_type: Type of conversational intent (greeting, small_talk, thanks, farewell, help_request)
            query: The user's original message

        Returns:
            Dictionary with response and metadata
        """
        # Generate conversational response
        response = self.chain.invoke({"message": query})

        # Customize based on intent type
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        return {
            "response": response_text,
            "intent_handled": intent_type,
            "requires_no_search": True,
            "metadata": {"conversational": True, "intent_type": intent_type},
        }

    async def astream_response(
        self, intent_type: str, query: str
    ) -> AsyncGenerator[str, None]:
        """Stream a conversational response using real LLM token streaming.

        Uses chain.astream() to yield actual tokens as the LLM generates them,
        instead of faking it by chunking a pre-generated response.

        Args:
            intent_type: Type of conversational intent (not used directly, but available for future customization)
            query: The user's original message

        Yields:
            Real LLM token chunks as they arrive from the API
        """
        try:
            async for chunk in self.chain.astream({"message": query}):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"Error generating response: {str(e)}"


def get_conversational_response_templates() -> Dict[str, str]:
    """Return template responses for common conversational intents.

    These can be used as fallbacks or for faster response without LLM call.
    """
    return {
        "greeting": "Hello! 👋 How can I help you today? I can search Wikipedia, academic papers, or the web for information.",
        "small_talk": "I'm doing well, thank you! Is there anything I can help you with? I can answer questions using various search tools.",
        "thanks": "You're welcome! 😊 Feel free to ask if you need anything else!",
        "farewell": "Goodbye! Have a great day! Come back anytime you need help with research or information.",
        "help_request": "Of course! I'd be happy to help. What would you like assistance with? You can ask me to search for information from various sources.",
    }
