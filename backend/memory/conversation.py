import asyncio
import logging
import threading
from typing import Dict, List, Optional

import tiktoken
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter

logger = logging.getLogger(__name__)


class ConversationBufferWindowMemory:
    """Manages conversation history with a sliding window to stay within token limits."""

    def __init__(
        self,
        llm: ChatOpenRouter,
        max_token_limit: int = 3000,
        max_history_messages: int = 10,
    ):
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.max_history_messages = max_history_messages
        self.messages: List[BaseMessage] = []
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self._lock = threading.RLock()
        self._token_cache: Dict[int, int] = {}  # Cache token counts by message id

    def __len__(self) -> int:
        """Return the number of messages in the conversation."""
        with self._lock:
            return len(self.messages)

    def add_message(self, message: BaseMessage):
        """Add a message to the conversation history."""
        with self._lock:
            self.messages.append(message)
            # Cache token count for new message
            self._cache_message_tokens(message)
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
        with self._lock:
            return self.messages.copy()

    def _cache_message_tokens(self, message: BaseMessage):
        """Cache token count for a message."""
        msg_id = id(message)
        if msg_id not in self._token_cache:
            self._token_cache[msg_id] = self._count_message_tokens(message)

    def _count_message_tokens(self, message: BaseMessage) -> int:
        """Count tokens for a single message."""
        content = message.content
        if isinstance(content, str):
            try:
                return len(self.token_encoder.encode(content))
            except Exception:
                return len(content)
        elif isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, str):
                    try:
                        total += len(self.token_encoder.encode(item))
                    except Exception:
                        total += len(item)
            return total
        elif content is None:
            return 0
        else:
            try:
                content_str = str(content)
                return len(self.token_encoder.encode(content_str))
            except Exception:
                return len(str(content))

    def _trim_messages(self):
        """Trim messages to stay within token and message count limits."""
        # First, limit by message count
        if len(self.messages) > self.max_history_messages:
            self.messages = self.messages[-self.max_history_messages :]
            # Simplified cache invalidation: clear and repopulate lazily
            self._token_cache.clear()

        # Then, trim by token count if necessary
        total_tokens = self._count_tokens_cached()
        if total_tokens > self.max_token_limit:
            # Use LangChain's trim_messages function with correct parameters
            trimmed_messages = trim_messages(
                self.messages,
                token_counter=self._count_tokens_for_trim_cached,
                max_tokens=self.max_token_limit,
                strategy="last",  # Keep messages from the end
                start_on="human",  # Start keeping messages from the most recent human message
                include_system=True,  # Always include system messages
            )

            self.messages = trimmed_messages
            # Simplified cache invalidation: clear and repopulate lazily
            self._token_cache.clear()

    def _count_tokens_cached(self) -> int:
        """Count total tokens using cache."""
        total = 0
        for msg in self.messages:
            msg_id = id(msg)
            if msg_id in self._token_cache:
                total += self._token_cache[msg_id]
            else:
                # Fallback if not cached - lazy population
                token_count = self._count_message_tokens(msg)
                self._token_cache[msg_id] = token_count
                total += token_count
        return total

    def _count_tokens_for_trim_cached(self, messages: List[BaseMessage]) -> int:
        """Count tokens for the trim function using cache."""
        total = 0
        for msg in messages:
            msg_id = id(msg)
            if msg_id in self._token_cache:
                total += self._token_cache[msg_id]
            else:
                # Fallback if not cached - lazy population
                token_count = self._count_message_tokens(msg)
                self._token_cache[msg_id] = token_count
                total += token_count
        return total

    def _count_tokens(self) -> int:
        """Count the total number of tokens in the conversation (legacy method)."""
        return self._count_tokens_cached()

    def _count_tokens_for_trim(self, messages: List[BaseMessage]) -> int:
        """Count tokens for the trim function (legacy method)."""
        return self._count_tokens_for_trim_cached(messages)

    def clear(self):
        """Clear the conversation history."""
        with self._lock:
            self.messages = []
            self._token_cache.clear()


class ConversationSummaryMemory:
    """Maintains conversation history by summarizing older interactions."""

    # Number of recent messages to keep as buffer (not summarized)
    RECENT_BUFFER_SIZE = 5

    def __init__(
        self,
        llm: ChatOpenRouter,
        max_token_limit: int = 3000,
        max_summary_tokens: int = 500,
    ):
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.max_summary_tokens = max_summary_tokens  # Configurable summary token limit
        self.summary = ""
        self.recent_messages: List[BaseMessage] = []
        self.token_encoder = tiktoken.get_encoding(
            "cl100k_base"
        )  # Fixed: use cl100k_base
        self._lock = threading.RLock()  # Thread-safe lock for all state access
        self._token_cache: Dict[int, int] = {}  # Cache token counts by message id
        self._summarizing = False  # Guard against concurrent summarization
        self._summary_token_count: Optional[int] = None  # Cache summary token count

        # Create a prompt for summarizing conversations
        self.summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at summarizing conversations. Create a concise summary that captures the key points and context of the conversation.",
                ),
                (
                    "human",
                    "Please summarize the following conversation:\n\n{conversation}",
                ),
            ]
        )
        self.summary_chain = self.summary_prompt | self.llm

    def __len__(self) -> int:
        """Return the total number of messages in the conversation."""
        with self._lock:
            return len(self.recent_messages)

    async def aadd_message(self, message: BaseMessage):
        """Add a message to the conversation history (async version).

        This is the primary method for adding messages. Use this in async contexts.
        For synchronous code, use add_message() but note it cannot be called from
        within an async context (use aadd_message directly instead).
        """
        # Acquire lock, update state, check if summary needed, then release lock
        with self._lock:
            self.recent_messages.append(message)
            # Cache token count for new message
            self._cache_message_tokens(message)
            needs_summary = self._count_total_tokens_cached() > self.max_token_limit

        # Summary generation happens outside the lock to avoid blocking
        if needs_summary and not self._summarizing:
            await self._create_summary_async()

    def add_message(self, message: BaseMessage):
        """Add a message to the conversation history (sync wrapper).

        Note: This method uses asyncio.run() and cannot be called from within
        an async context. If you're in an async function, use aadd_message() directly.
        """
        asyncio.run(self.aadd_message(message))

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        """Add an AI message to the conversation."""
        self.add_message(AIMessage(content=content))

    def get_messages(self) -> List[BaseMessage]:
        """Get the current conversation history with summary if applicable."""
        with self._lock:
            result = []

            # Add summary if we have one
            if self.summary:
                result.append(
                    SystemMessage(
                        content=f"Summary of earlier conversation: {self.summary}"
                    )
                )

            # Add the recent messages (earlier messages are now in the summary)
            result.extend(self.recent_messages)

            return result

    def _cache_message_tokens(self, message: BaseMessage):
        """Cache token count for a message."""
        msg_id = id(message)
        if msg_id not in self._token_cache:
            self._token_cache[msg_id] = self._count_message_tokens(message)

    def _count_message_tokens(self, message: BaseMessage) -> int:
        """Count tokens for a single message."""
        content = message.content
        if isinstance(content, str):
            try:
                return len(self.token_encoder.encode(content))
            except Exception:
                return len(content)
        elif isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, str):
                    try:
                        total += len(self.token_encoder.encode(item))
                    except Exception:
                        total += len(item)
            return total
        elif content is None:
            return 0
        else:
            try:
                content_str = str(content)
                return len(self.token_encoder.encode(content_str))
            except Exception:
                return len(str(content))

    async def _create_summary_async(self):
        """Create a summary of the earlier messages asynchronously."""
        # Set the summarizing flag under lock
        with self._lock:
            if self._summarizing:
                return
            self._summarizing = True

        try:
            # Determine which messages to summarize (keep a buffer of recent messages)
            with self._lock:
                if not self.recent_messages:
                    return

                buffer_size = min(self.RECENT_BUFFER_SIZE, len(self.recent_messages))
                if len(self.recent_messages) <= buffer_size:
                    return  # Not enough messages to summarize

                # Get messages to summarize and keep buffered messages
                messages_to_summarize = self.recent_messages[:-buffer_size]
                self.recent_messages = self.recent_messages[-buffer_size:]

                # Get IDs of messages being summarized for cache cleanup
                msg_ids_to_remove = {id(msg) for msg in messages_to_summarize}

            # Build conversation text directly from messages_to_summarize (no self.earlier_messages)
            conversation_text = ""
            for msg in messages_to_summarize:
                role = "User" if isinstance(msg, HumanMessage) else "AI"
                conversation_text += f"{role}: {msg.content}\n"

            # Generate summary using the LLM asynchronously
            try:
                # Run synchronous invoke in a thread to avoid blocking
                summary_response = await asyncio.to_thread(
                    self.summary_chain.invoke, {"conversation": conversation_text}
                )
                with self._lock:
                    self.summary = summary_response.content
                    # Invalidate cached summary token count
                    self._summary_token_count = None
            except Exception as e:
                logger.warning(
                    f"LLM summarization failed: {e}. Falling back to truncation."
                )
                with self._lock:
                    self.summary = self._truncate_text(conversation_text)
                    # Invalidate cached summary token count
                    self._summary_token_count = None

            # Update token cache for removed messages
            with self._lock:
                for msg_id in msg_ids_to_remove:
                    if msg_id in self._token_cache:
                        del self._token_cache[msg_id]

        finally:
            # Always reset the summarizing flag
            with self._lock:
                self._summarizing = False

    def _truncate_text(self, text: str) -> str:
        """Truncate text based on token limit (renamed from _simple_token_summary)."""
        tokens = self.token_encoder.encode(text)
        if len(tokens) > self.max_summary_tokens:
            tokens = tokens[: self.max_summary_tokens]
        return self.token_encoder.decode(tokens)

    def _simple_token_summary(self, text: str) -> str:
        """Create a simple summary by truncating text based on tokens (legacy method)."""
        return self._truncate_text(text)

    def _count_total_tokens_cached(self) -> int:
        """Count the total number of tokens in the conversation using cache."""
        total_tokens = 0

        # Count tokens in summary using cached value
        if self.summary:
            if self._summary_token_count is not None:
                total_tokens += self._summary_token_count
            else:
                # Compute and cache the summary token count
                self._summary_token_count = len(self.token_encoder.encode(self.summary))
                total_tokens += self._summary_token_count

        # Count tokens in recent messages using cache
        for message in self.recent_messages:
            msg_id = id(message)
            if msg_id in self._token_cache:
                total_tokens += self._token_cache[msg_id]
            else:
                token_count = self._count_message_tokens(message)
                self._token_cache[msg_id] = token_count
                total_tokens += token_count

        return total_tokens

    def _count_total_tokens(self) -> int:
        """Count the total number of tokens in the conversation (legacy method)."""
        return self._count_total_tokens_cached()

    def clear(self):
        """Clear the conversation history."""
        with self._lock:
            self.summary = ""
            self.recent_messages = []
            self._token_cache.clear()
            self._summary_token_count = None
