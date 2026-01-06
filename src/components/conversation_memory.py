from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Optional
import tiktoken


class ConversationBufferWindowMemory:
    """Manages conversation history with a sliding window to stay within token limits."""
    
    def __init__(self, llm: ChatGroq, max_token_limit: int = 3000, max_history_messages: int = 10):
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.max_history_messages = max_history_messages
        self.messages: List[BaseMessage] = []
        self.token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Using a common encoder
        
    def add_message(self, message: BaseMessage):
        """Add a message to the conversation history."""
        self.messages.append(message)
        
        # Trim messages if needed to stay within limits
        self._trim_messages()
    
    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.add_message(HumanMessage(content=content))
    
    def add_ai_message(self, content: str):
        """Add an AI message to the conversation."""
        self.add_message(AIMessage(content=content))
    
    def get_messages(self) -> List[BaseMessage]:
        """Get the current conversation history."""
        return self.messages.copy()
    
    def _trim_messages(self):
        """Trim messages to stay within token and message count limits."""
        # First, limit by message count
        if len(self.messages) > self.max_history_messages:
            self.messages = self.messages[-self.max_history_messages:]
        
        # Then, trim by token count if necessary
        total_tokens = self._count_tokens()
        if total_tokens > self.max_token_limit:
            # Use LangChain's trim_messages function to reduce tokens
            prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="messages")
            ])
            
            # Create a trim function
            trimmer = trim_messages(
                prompt,
                token_counter=self._count_tokens_for_trim,
                max_tokens=self.max_token_limit,
                start_on="human"  # Start keeping messages from the most recent human message
            )
            
            # Apply trimming
            trimmed_messages = trimmer.invoke({"messages": self.messages})
            self.messages = trimmed_messages
    
    def _count_tokens(self) -> int:
        """Count the total number of tokens in the conversation."""
        total_tokens = 0
        for message in self.messages:
            content = message.content
            if isinstance(content, str):
                try:
                    total_tokens += len(self.token_encoder.encode(content))
                except Exception:
                    # Fallback if token encoding fails
                    total_tokens += len(content)
            elif isinstance(content, list):
                # Handle list of content (for complex message types)
                for item in content:
                    if isinstance(item, str):
                        try:
                            total_tokens += len(self.token_encoder.encode(item))
                        except Exception:
                            # Fallback if token encoding fails
                            total_tokens += len(item)
            elif content is None:
                continue
            else:
                # Handle other content types
                try:
                    content_str = str(content)
                    total_tokens += len(self.token_encoder.encode(content_str))
                except Exception:
                    # Fallback if token encoding fails
                    total_tokens += len(str(content))
        return total_tokens
    
    def _count_tokens_for_trim(self, messages: List[BaseMessage]) -> int:
        """Count tokens for the trim function."""
        total_tokens = 0
        for message in messages:
            content = message.content
            if isinstance(content, str):
                try:
                    total_tokens += len(self.token_encoder.encode(content))
                except Exception:
                    # Fallback if token encoding fails
                    total_tokens += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        try:
                            total_tokens += len(self.token_encoder.encode(item))
                        except Exception:
                            # Fallback if token encoding fails
                            total_tokens += len(item)
            elif content is None:
                continue
            else:
                # Handle other content types
                try:
                    content_str = str(content)
                    total_tokens += len(self.token_encoder.encode(content_str))
                except Exception:
                    # Fallback if token encoding fails
                    total_tokens += len(str(content))
        return total_tokens
    
    def clear(self):
        """Clear the conversation history."""
        self.messages = []


class ConversationSummaryMemory:
    """Maintains conversation history by summarizing older interactions."""
    
    def __init__(self, llm: ChatGroq, max_token_limit: int = 3000):
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.summary = ""
        self.earlier_messages: List[BaseMessage] = []
        self.recent_messages: List[BaseMessage] = []
        self.token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Create a prompt for summarizing conversations
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at summarizing conversations. Create a concise summary that captures the key points and context of the conversation."),
            ("human", "Please summarize the following conversation:\n\n{conversation}")
        ])
        self.summary_chain = self.summary_prompt | self.llm
    
    def add_message(self, message: BaseMessage):
        """Add a message to the conversation history."""
        self.recent_messages.append(message)
        
        # Check if we need to summarize older messages
        if self._count_total_tokens() > self.max_token_limit:
            self._create_summary()
    
    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.add_message(HumanMessage(content=content))
    
    def add_ai_message(self, content: str):
        """Add an AI message to the conversation."""
        self.add_message(AIMessage(content=content))
    
    def get_messages(self) -> List[BaseMessage]:
        """Get the current conversation history with summary if applicable."""
        result = []
        
        # Add summary if we have one
        if self.summary:
            result.append(SystemMessage(content=f"Summary of earlier conversation: {self.summary}"))
        
        # Add the earlier messages (that were summarized)
        result.extend(self.earlier_messages)
        
        # Add the recent messages
        result.extend(self.recent_messages)
        
        return result
    
    def _create_summary(self):
        """Create a summary of the earlier messages and clear them."""
        if not self.recent_messages:
            return
        
        # Move recent messages to earlier messages
        self.earlier_messages.extend(self.recent_messages)
        self.recent_messages = []
        
        # Create summary of earlier messages
        conversation_text = ""
        for msg in self.earlier_messages:
            role = "User" if isinstance(msg, HumanMessage) else "AI"
            conversation_text += f"{role}: {msg.content}\n"
        
        # Generate summary using the LLM
        try:
            summary_response = self.summary_chain.invoke({"conversation": conversation_text})
            self.summary = summary_response.content
        except Exception:
            # Fallback to a simple token-based summary if LLM fails
            self.summary = self._simple_token_summary(conversation_text)
        
        # Keep only the most recent messages to stay within limits
        while len(self.earlier_messages) > 2:  # Keep only last 2 messages in earlier_messages
            self.earlier_messages.pop(0)
    
    def _simple_token_summary(self, text: str) -> str:
        """Create a simple summary by truncating text based on tokens."""
        tokens = self.token_encoder.encode(text)
        max_summary_tokens = 500  # Limit summary to 500 tokens
        if len(tokens) > max_summary_tokens:
            tokens = tokens[:max_summary_tokens]
        return self.token_encoder.decode(tokens)
    
    def _count_total_tokens(self) -> int:
        """Count the total number of tokens in the conversation."""
        total_tokens = 0
        
        # Count tokens in summary
        if self.summary:
            total_tokens += len(self.token_encoder.encode(self.summary))
        
        # Count tokens in earlier messages
        for message in self.earlier_messages:
            content = message.content
            if isinstance(content, str):
                total_tokens += len(self.token_encoder.encode(content))
        
        # Count tokens in recent messages
        for message in self.recent_messages:
            content = message.content
            if isinstance(content, str):
                total_tokens += len(self.token_encoder.encode(content))
        
        return total_tokens
    
    def clear(self):
        """Clear the conversation history."""
        self.summary = ""
        self.earlier_messages = []
        self.recent_messages = []
        self.messages = []


