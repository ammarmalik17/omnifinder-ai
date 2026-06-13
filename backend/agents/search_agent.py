import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, AsyncGenerator, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from backend.components.conversational_handler import ConversationalHandler
from backend.components.query_classifier import QueryClassifier
from backend.components.result_synthesizer import ResultSynthesizer
from backend.config.agent_config import AgentConfig
from backend.core.react_agent import create_omnifinder_react_agent
from backend.memory.conversation import ConversationBufferWindowMemory
from backend.schemas.agent_response import AgentResponse
from backend.schemas.result import ToolResponse
from backend.tools.search import get_all_tools
from backend.utils.logger import AgentLogger


class SearchAgent:
    """Main search agent that integrates all components."""

    def __init__(self, llm: BaseChatModel, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.llm = llm
        self.max_workers = self.config.max_workers
        self.use_react_for_complex = self.config.use_react_for_complex
        self.logger = AgentLogger("SearchAgent")

        self.query_classifier = QueryClassifier(llm)
        self.result_synthesizer = ResultSynthesizer(llm)
        self.memory = ConversationBufferWindowMemory(
            llm,
            max_token_limit=self.config.max_token_limit,
            max_history_messages=self.config.max_history_messages,
        )

        # Initialize conversational handler
        self.conversational_handler = ConversationalHandler(llm)

        # Initialize tools with the agent's config
        all_tools = get_all_tools(config=self.config)
        self.tools = {tool.name: tool for tool in all_tools}

        # Create a mapping from old names to new names for backward compatibility
        self.tool_name_mapping = {
            "wikipedia_search": "wikipedia",
            "arxiv_search": "arxiv",
            "duckduckgo_search": "web_search",
        }

        # Reverse mapping for forward compatibility
        self.reverse_tool_name_mapping = {
            v: k for k, v in self.tool_name_mapping.items()
        }

        # Initialize ReAct agent for complex queries
        self.react_agent = create_omnifinder_react_agent(llm, list(self.tools.values()))

        # Thread lock for thread safety
        self.lock = threading.Lock()

    def process_query(
        self, query: str, enabled_tools: List[str] = None, use_react: bool = None
    ) -> AgentResponse:
        """
        Process a user query through the full pipeline with intent detection.

        Follows industry-standard flowchart:
        1. Intent Detection → Conversational vs Search
        2. For Conversational: Direct response (NO SEARCH)
        3. For Search: Route to appropriate tools
        4. Apply guardrails (confidence checking, clarification)

        Args:
            query: The user's query string
            enabled_tools: List of tool names to enable (None = all tools)
            use_react: Whether to use ReAct mode for complex queries (None = use default logic)

        Returns:
            Dictionary containing classification, search results, and synthesized answer
        """
        # Start comprehensive query timing
        self.logger.start_query_timing(query)

        try:
            # STEP 1: Intent Detection
            with self.logger.log_timing_context(
                "Query Classification", f"Query: {query[:100]}"
            ):
                classification = self.query_classifier.classify(query)

            self.logger.log_classification(
                classification.primary_tool,
                classification.secondary_tools,
                classification.confidence,
            )
            self.logger.log_query_processing(
                query,
                "conversational"
                if classification.intent_type == "conversational"
                else ("react" if self.use_react_for_complex else "traditional"),
            )

            # STEP 2: Handle Conversational Intents (NO SEARCH)
            if classification.intent_type == "conversational":
                self.logger.log_step(
                    "Handling conversational intent",
                    f"Intent type: {classification.conversational_intent}",
                )
                return self._handle_conversational_intent(query, classification)

            # STEP 3: Check for Low Confidence / Need Clarification
            if classification.needs_clarification or classification.confidence < 0.5:
                self.logger.log_step(
                    "Low confidence detected",
                    f"Confidence: {classification.confidence:.2f}, Needs clarification: {classification.needs_clarification}",
                )
                return self._handle_low_confidence(query, classification)

            # STEP 4: Determine if we should use ReAct for complex queries
            should_use_react = (
                use_react if use_react is not None else self.use_react_for_complex
            )

            if should_use_react and self._is_complex_query(query, classification):
                self.logger.log_step("Complex query detected - using ReAct pattern")
                self.logger.info(
                    f"Using ReAct pattern for complex query: {query[:50]}..."
                )
                return self._process_with_react(query, classification)

            # STEP 5: Traditional search routing
            self.logger.log_step("Traditional search routing")
            return self._process_search_query(query, classification, enabled_tools)

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            self.logger.end_query_timing(success=False)
            raise

    def _handle_conversational_intent(
        self, query: str, classification
    ) -> AgentResponse:
        """Handle conversational intents without search.

        Args:
            query: User's query
            classification: Classification result with conversational intent

        Returns:
            AgentResponse with conversational response
        """
        # Use handler to generate appropriate response
        response_data = self.conversational_handler.handle(
            classification.conversational_intent, query
        )

        # Add to memory
        with self.lock:
            self.memory.add_user_message(query)
            self.memory.add_ai_message(response_data["response"])

        return AgentResponse(
            query=query,
            classification=classification,
            synthesized_answer=response_data["response"],
            conversational=True,
            intent_handled=classification.conversational_intent,
        )

    def _handle_low_confidence(self, query: str, classification) -> AgentResponse:
        """Handle low confidence queries by asking for clarification.

        Args:
            query: User's query
            classification: Classification result with low confidence

        Returns:
            AgentResponse with clarifying question
        """
        clarification_response = f'''I want to make sure I understand your question correctly.

Could you please provide more details about what you're looking for? For example:
- Are you asking about a specific topic or concept?
- Do you need current/recent information or general knowledge?
- Are you looking for academic research or general information?

Your query: "{query}"'''

        # Add to memory
        with self.lock:
            self.memory.add_user_message(query)
            self.memory.add_ai_message(clarification_response)

        return AgentResponse(
            query=query,
            classification=classification,
            synthesized_answer=clarification_response,
            needs_clarification=True,
        )

    def _process_search_query(
        self, query: str, classification, enabled_tools: List[str] = None
    ) -> AgentResponse:
        """Process a search query through tool execution and synthesis.

        Args:
            query: User's query
            classification: Classification result
            enabled_tools: List of enabled tools

        Returns:
            AgentResponse with search results and synthesized answer
        """
        # Determine which tools to use based on classification and enabled_tools filter
        tools_to_use = (
            [classification.primary_tool] + classification.secondary_tools
            if classification.primary_tool
            else []
        )
        self.logger.log_step(
            "Tool selection",
            f"Primary: {classification.primary_tool}, Secondary: {classification.secondary_tools}",
        )

        # Filter tools based on enabled_tools parameter
        if enabled_tools:
            tools_to_use = [tool for tool in tools_to_use if tool in enabled_tools]
            # Ensure at least one tool is available
            if not tools_to_use:
                tools_to_use = [enabled_tools[0]] if enabled_tools else ["web_search"]

        self.logger.info(f"📋 Tools to execute: {tools_to_use}")

        # Execute search tools concurrently
        with self.logger.log_timing_context(
            "Concurrent Tool Execution", f"Tools: {', '.join(tools_to_use)}"
        ):
            search_results = self._execute_search_tools(query, tools_to_use)

        self.logger.log_step(
            "Tool execution complete", f"Got results from {len(search_results)} tools"
        )

        # Synthesize results into a comprehensive answer
        with self.logger.log_timing_context(
            "Result Synthesis", f"Synthesizing from {len(search_results)} sources"
        ):
            self.logger.log_synthesis_start(len(search_results))
            synthesized_answer = self.result_synthesizer.synthesize(
                query, search_results
            )
            self.logger.log_synthesis_complete(0, len(synthesized_answer))

        # Add to memory
        with self.lock:
            self.memory.add_user_message(query)
            self.memory.add_ai_message(synthesized_answer)
            msg_count = len(self.memory.get_messages())
            self.logger.log_memory_operation("add_messages", msg_count)

        # End query timing
        self.logger.end_query_timing(success=True)

        return AgentResponse(
            query=query,
            classification=classification,
            search_results=search_results,
            synthesized_answer=synthesized_answer,
        )

    def _is_complex_query(self, query: str, classification) -> bool:
        """
        Determine if a query is complex enough to benefit from ReAct reasoning.

        Args:
            query: The user's query
            classification: The query classification result

        Returns:
            True if the query should use ReAct reasoning, False otherwise
        """
        # Compound queries (multiple sub-questions) always go to ReAct
        if hasattr(classification, "is_compound") and classification.is_compound:
            return True

        # Consider a query complex if it:
        # - Contains multiple sub-questions
        # - Requires synthesis of information from multiple sources
        # - Contains complex reasoning terms
        query_lower = query.lower()
        for indicator in self.config.complex_query_indicators:
            if indicator in query_lower:
                return True

        # If multiple tools are suggested, it might be a complex query
        if len([classification.primary_tool] + classification.secondary_tools) > 2:
            return True

        return False

    def _process_with_react(self, query: str, classification=None) -> AgentResponse:
        """
        Process a query using the ReAct agent for complex reasoning.

        Args:
            query: The user's query
            classification: Optional classification result (for sub_queries)

        Returns:
            AgentResponse containing the response from the ReAct agent
        """
        sub_queries = None
        if classification and hasattr(classification, "is_compound") and classification.is_compound:
            sub_queries = classification.sub_queries

        react_result = self.react_agent.process_query(query, sub_queries=sub_queries)

        return AgentResponse(
            query=query,
            synthesized_answer=react_result["final_answer"],
            react_steps=react_result["steps"],
            react_iterations=react_result["iterations"],
        )

    def _execute_search_tools(
        self, query: str, tools_to_use: List[str]
    ) -> List[ToolResponse]:
        """
        Execute search tools concurrently to improve performance.

        Args:
            query: The user's query
            tools_to_use: List of tool names to execute

        Returns:
            List of ToolResponse from each tool
        """
        search_results = []
        tool_timings = {}

        # Use ThreadPoolExecutor to run tools concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks to the executor
            future_to_tool = {}
            for tool_name in tools_to_use:
                if tool_name in self.tools:
                    self.logger.log_tool_usage(tool_name, query)
                    future = executor.submit(
                        self._execute_single_tool_with_timing, tool_name, query
                    )
                    future_to_tool[future] = tool_name
                    tool_timings[tool_name] = {"start": time.time()}

            # Collect results as they complete
            for future in as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                tool_start = tool_timings[tool_name]["start"]
                tool_duration = time.time() - tool_start

                try:
                    result = future.result()
                    self.logger.log_tool_result(tool_name, tool_duration, len(result))
                    search_results.append(
                        ToolResponse(
                            tool_name=tool_name,
                            content=result,
                            execution_time=tool_duration,
                        )
                    )
                except Exception as e:
                    self.logger.log_tool_error(tool_name, str(e), tool_duration)
                    error_msg = f"Error with {tool_name}: {str(e)}"
                    search_results.append(
                        ToolResponse(
                            tool_name=tool_name,
                            content=error_msg,
                            success=False,
                            error=str(e),
                            execution_time=tool_duration,
                        )
                    )

        return search_results

    def _execute_single_tool(self, tool_name: str, query: str) -> str:
        """
        Execute a single search tool.

        Args:
            tool_name: Name of the tool to execute
            query: The user's query

        Returns:
            Tool execution result
        """
        # Check if the tool name exists in the current tools
        if tool_name in self.tools:
            tool = self.tools[tool_name]
        # If not found, check if it's an old name that needs mapping
        elif tool_name in self.reverse_tool_name_mapping:
            mapped_name = self.reverse_tool_name_mapping[tool_name]
            tool = self.tools[mapped_name]
        else:
            raise ValueError(f"Tool '{tool_name}' not found")

        return tool.invoke(query)

    def _execute_single_tool_with_timing(self, tool_name: str, query: str) -> str:
        """
        Execute a single search tool with detailed timing.

        Args:
            tool_name: Name of the tool to execute
            query: The user's query

        Returns:
            Tool execution result
        """
        start_time = time.time()
        self.logger.info(f"▶️ Executing tool: {tool_name}")

        try:
            result = self._execute_single_tool(tool_name, query)
            elapsed = time.time() - start_time
            self.logger.info(
                f"✔️ Tool {tool_name} completed in {elapsed:.3f}s (result: {len(result)} chars)"
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(
                f"✖️ Tool {tool_name} failed after {elapsed:.3f}s: {str(e)}"
            )
            raise

    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the current conversation history."""
        return self.memory.get_messages()

    def clear_conversation(self):
        """Clear the conversation history."""
        with self.lock:
            self.memory.clear()

    def chat(
        self, query: str, enabled_tools: List[str] = None, use_react: bool = None
    ) -> str:
        """
        Simple chat interface that returns just the answer.

        Args:
            query: The user's query
            enabled_tools: List of tool names to enable (None = all tools)
            use_react: Whether to use ReAct mode for complex queries (None = use default logic)

        Returns:
            The synthesized answer
        """
        result = self.process_query(
            query, enabled_tools=enabled_tools, use_react=use_react
        )
        return result.synthesized_answer

    async def stream_synthesized_answer(
        self, query: str, result: AgentResponse
    ) -> AsyncGenerator[str, None]:
        """
        Stream the synthesized answer using real LLM token streaming.

        For search results, uses model.astream() to yield actual tokens as the LLM
        generates them. For conversational responses, chunks the pre-generated text.

        Args:
            query: The user's query
            result: The AgentResponse from process_query

        Yields:
            Chunks of the synthesized answer as they become available
        """
        try:
            # For conversational responses, stream directly (pre-generated text)
            if result.conversational:
                response = result.synthesized_answer
                for i in range(0, len(response), 3):
                    chunk = response[i : i + 3]
                    yield chunk
                    await asyncio.sleep(0.01)
                return

            # For search results, stream from the synthesizer's real LLM streaming
            search_results = result.search_results
            if not search_results:
                yield result.synthesized_answer
                return

            # Stream initial banner, then delegate to real LLM token streaming
            yield "\U0001f50d Analyzing search results...\n\n"
            async for chunk in self.result_synthesizer.stream_synthesize_async(
                query, search_results
            ):
                yield chunk

        except Exception as e:
            yield f"\u274c Error during streaming: {str(e)}"
