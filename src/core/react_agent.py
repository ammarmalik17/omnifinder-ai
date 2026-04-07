"""
ReAct (Reasoning and Acting) Agent Implementation for OmniFinder AI.

This module implements the ReAct pattern for search agents, which alternates between
reasoning about the problem and taking actions with tools to solve complex queries.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openrouter import ChatOpenRouter

from src.utils.logger import AgentLogger


class ReActSearchAgent:
    """
    A ReAct (Reasoning and Acting) based search agent that can dynamically
    reason about queries and take actions with multiple tools to find answers.
    """

    def __init__(self, llm: ChatOpenRouter, tools: List[BaseTool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.logger = AgentLogger("ReActAgent")

        # Create a system prompt that includes tool descriptions for manual tool selection
        # This avoids using bind_tools() which requires OpenRouter endpoint support
        self.react_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are OmniFinder AI, an intelligent search agent that uses the ReAct (Reasoning and Acting) framework to answer queries.

Your approach:
1. REASON: Think step-by-step about the query and what information is needed
2. ACT: Use tools to gather information when needed
3. OBSERVE: Analyze the results from tools
4. REPEAT: Continue reasoning and acting until you have sufficient information
5. FINALIZE: Provide a comprehensive answer

Available tools:
{tool_descriptions}

When using tools:
- Always think before acting
- Use the most appropriate tool for the specific information needed
- Wait for tool results before proceeding
- Combine information from multiple sources when relevant

To use a tool, respond with:
ACTION: <tool_name>
ARGS: <arguments as JSON>

Your response should be informative, well-structured, and cite sources when possible.""",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Create the chain WITHOUT bound tools - we'll handle tool selection manually
        self.chain = self.react_prompt | self.llm

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the ReAct pattern.

        Args:
            query: The user's query

        Returns:
            Dictionary containing the final response and intermediate steps
        """
        now = datetime.now()
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(
            f"🧠 REACT AGENT STARTING at {now.strftime('%H:%M:%S.%f')[:-3]}"
        )
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Available tools: {', '.join(self.tools.keys())}")
        self.logger.info(f"{'=' * 60}\n")

        react_start_time = time.time()

        # Initialize conversation with the user query
        messages = [HumanMessage(content=query)]

        # Track the steps for transparency
        steps = []

        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        # Build tool descriptions for the prompt
        tool_descriptions = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in self.tools.values()]
        )

        self.logger.info(f"📋 Tool Descriptions:\n{tool_descriptions}\n")

        while iteration < max_iterations:
            iteration += 1
            iter_start_time = time.time()
            iter_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.logger.info(f"\n{'─' * 40}")
            self.logger.info(f"🔄 ITERATION {iteration} starting at {iter_time}")

            try:
                # Get the next message from the LLM
                llm_start = time.time()
                self.logger.info(f"🧠 Invoking LLM with {len(messages)} messages...")
                response = self.chain.invoke(
                    {"messages": messages, "tool_descriptions": tool_descriptions}
                )
                llm_duration = time.time() - llm_start
                self.logger.info(f"✅ LLM response received in {llm_duration:.3f}s")

                # Add the AI's response to the conversation
                messages.append(response)

                # Parse the response to check for tool usage
                response_content = (
                    response.content if hasattr(response, "content") else str(response)
                )
                self.logger.info(
                    f"📝 Response content ({len(response_content)} chars):\n{response_content[:300]}..."
                    if len(response_content) > 300
                    else f"📝 Response content ({len(response_content)} chars):\n{response_content}"
                )

                # Check if the response contains an action (tool call)
                if "ACTION:" in response_content.upper():
                    self.logger.info("🎯 Detected ACTION keyword - parsing tool call")
                    self.logger.log_react_iteration(iteration, "tool_call_detected")
                    # Parse the action and arguments
                    action_line = None
                    args_line = None

                    lines = response_content.split("\n")
                    for i, line in enumerate(lines):
                        if line.strip().upper().startswith("ACTION:"):
                            action_line = line
                            # Look for ARGS in the next few lines
                            for j in range(i + 1, min(i + 4, len(lines))):
                                if lines[j].strip().upper().startswith("ARGS:"):
                                    args_line = lines[j]
                                    break
                            break

                    if action_line:
                        # Extract tool name
                        tool_name = action_line.split(":", 1)[1].strip().lower()
                        self.logger.info(f"🔧 Tool name extracted: {tool_name}")

                        # Extract arguments
                        tool_args = {}
                        if args_line:
                            import json

                            try:
                                args_str = args_line.split(":", 1)[1].strip()
                                tool_args = json.loads(args_str)
                                self.logger.info(f"📋 Arguments parsed: {tool_args}")
                            except Exception as e:
                                self.logger.warning(
                                    f"⚠️ JSON parsing failed: {e}, using default query"
                                )
                                tool_args = {"query": query}
                        else:
                            self.logger.info("📋 No ARGS found, using default query")
                            tool_args = {"query": query}

                        # Log the step
                        steps.append(
                            {
                                "step": iteration,
                                "action": "tool_call",
                                "tool": tool_name,
                                "arguments": tool_args,
                            }
                        )

                        # Execute the tool
                        tool_exec_start = time.time()
                        if tool_name in self.tools:
                            self.logger.info(f"▶️ Executing tool: {tool_name}")
                            tool = self.tools[tool_name]
                            try:
                                tool_result = tool._run(**tool_args)
                                tool_exec_duration = time.time() - tool_exec_start
                                self.logger.info(
                                    f"✅ Tool execution successful in {tool_exec_duration:.3f}s (result: {len(tool_result)} chars)"
                                )
                                self.logger.log_react_iteration(
                                    iteration, "tool_executed", tool_name
                                )
                            except Exception as e:
                                tool_exec_duration = time.time() - tool_exec_start
                                tool_result = f"Error executing {tool_name}: {str(e)}"
                                self.logger.error(
                                    f"❌ Tool execution error after {tool_exec_duration:.3f}s: {tool_result}"
                                )
                        else:
                            tool_exec_duration = time.time() - tool_exec_start
                            tool_result = f"Unknown tool: {tool_name}"
                            self.logger.error(f"❌ Tool not found: {tool_name}")

                        # Add the tool result as a HumanMessage (since it's new input for the model)
                        tool_message = HumanMessage(
                            content=f"Observation from {tool_name}: {tool_result}"
                        )
                        messages.append(tool_message)

                        # Log the result
                        steps.append(
                            {
                                "step": iteration,
                                "action": "tool_result",
                                "tool": tool_name,
                                "result": tool_result,
                            }
                        )
                        self.logger.info("💾 Tool result added to messages")
                    else:
                        self.logger.warning(
                            "⚠️ No valid action found, generating final response"
                        )
                        steps.append(
                            {
                                "step": iteration,
                                "action": "final_response",
                                "content": response_content,
                            }
                        )
                        break
                else:
                    self.logger.info(
                        "✅ No ACTION keyword found - assuming final answer"
                    )
                    steps.append(
                        {
                            "step": iteration,
                            "action": "final_response",
                            "content": response_content,
                        }
                    )
                    break

            except Exception as e:
                iter_duration = time.time() - iter_start_time
                self.logger.error(
                    f"\n❌ ERROR in ReAct loop at iteration {iteration} (after {iter_duration:.3f}s):"
                )
                self.logger.error(f"Error type: {type(e).__name__}")
                self.logger.error(f"Error message: {str(e)}")
                import traceback

                self.logger.error(f"\nFull traceback:\n{traceback.format_exc()}")
                raise

        total_duration = time.time() - react_start_time
        end_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"✅ REACT AGENT COMPLETED at {end_time}")
        self.logger.info(f"⏱️ Total duration: {total_duration:.3f}s")
        self.logger.info(f"📊 Total iterations: {iteration}")
        self.logger.info(f"📊 Steps taken: {len(steps)}")
        self.logger.info(f"{'=' * 60}\n")

        return {
            "final_answer": response.content,
            "steps": steps,
            "iterations": iteration,
            "messages": messages,
        }

    async def stream_process_query(self, query: str) -> AsyncGenerator[str, None]:
        """
        Stream the ReAct process with progressive rendering.

        Args:
            query: The user's query

        Yields:
            Chunks of the ReAct process as it unfolds
        """
        try:
            yield "🧠 Starting ReAct reasoning process...\n\n"
            await asyncio.sleep(0.1)

            # Initialize conversation with the user query
            messages = [HumanMessage(content=query)]
            steps = []
            max_iterations = 10
            iteration = 0

            # Build tool descriptions for the prompt
            tool_descriptions = "\n".join(
                [f"- {tool.name}: {tool.description}" for tool in self.tools.values()]
            )

            yield f"📋 Available tools: {', '.join(self.tools.keys())}\n\n"
            await asyncio.sleep(0.1)

            while iteration < max_iterations:
                iteration += 1
                yield f"🔄 **Iteration {iteration}** - Reasoning...\n\n"
                await asyncio.sleep(0.1)

                try:
                    # Get the next message from the LLM
                    response = self.chain.invoke(
                        {"messages": messages, "tool_descriptions": tool_descriptions}
                    )

                    # Add the AI's response to the conversation
                    messages.append(response)

                    # Parse the response to check for tool usage
                    response_content = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )

                    # Check if the response contains an action (tool call)
                    if "ACTION:" in response_content.upper():
                        yield f"🎯 **Step {iteration}**: Tool usage detected\n\n"
                        await asyncio.sleep(0.05)

                        # Parse the action and arguments
                        action_line = None
                        args_line = None

                        lines = response_content.split("\n")
                        for i, line in enumerate(lines):
                            if line.strip().upper().startswith("ACTION:"):
                                action_line = line
                                # Look for ARGS in the next few lines
                                for j in range(i + 1, min(i + 4, len(lines))):
                                    if lines[j].strip().upper().startswith("ARGS:"):
                                        args_line = lines[j]
                                        break
                                break

                        if action_line:
                            # Extract tool name
                            tool_name = action_line.split(":", 1)[1].strip().lower()

                            # Extract arguments
                            tool_args = {}
                            if args_line:
                                import json

                                try:
                                    args_str = args_line.split(":", 1)[1].strip()
                                    tool_args = json.loads(args_str)
                                except:
                                    tool_args = {"query": query}
                            else:
                                tool_args = {"query": query}

                            # Log the step
                            steps.append(
                                {
                                    "step": iteration,
                                    "action": "tool_call",
                                    "tool": tool_name,
                                    "arguments": tool_args,
                                }
                            )

                            # Execute the tool
                            if tool_name in self.tools:
                                tool = self.tools[tool_name]
                                try:
                                    tool_result = tool._run(**tool_args)
                                    yield f"🔧 **Tool Result**: {tool_name} returned:\n```\n{tool_result[:200]}{'...' if len(tool_result) > 200 else ''}\n```\n\n"
                                    await asyncio.sleep(0.1)
                                except Exception as e:
                                    tool_result = (
                                        f"Error executing {tool_name}: {str(e)}"
                                    )
                                    yield f"❌ **Tool Error**: {tool_result}\n\n"
                                    await asyncio.sleep(0.1)
                            else:
                                tool_result = f"Unknown tool: {tool_name}"
                                yield f"❓ **Unknown Tool**: {tool_result}\n\n"
                                await asyncio.sleep(0.1)

                            # Add the tool result as a HumanMessage
                            tool_message = HumanMessage(
                                content=f"Observation from {tool_name}: {tool_result}"
                            )
                            messages.append(tool_message)

                            # Log the result
                            steps.append(
                                {
                                    "step": iteration,
                                    "action": "tool_result",
                                    "tool": tool_name,
                                    "result": tool_result,
                                }
                            )
                        else:
                            yield f"📝 **Step {iteration}**: Generating final response\n\n"
                            await asyncio.sleep(0.1)
                            steps.append(
                                {
                                    "step": iteration,
                                    "action": "final_response",
                                    "content": response_content,
                                }
                            )
                            break
                    else:
                        yield f"📝 **Step {iteration}**: Final response generated\n\n"
                        await asyncio.sleep(0.1)
                        steps.append(
                            {
                                "step": iteration,
                                "action": "final_response",
                                "content": response_content,
                            }
                        )
                        break

                except Exception as e:
                    yield f"❌ **Error in iteration {iteration}**: {str(e)}\n\n"
                    await asyncio.sleep(0.1)
                    break

            yield f"✅ **ReAct Process Complete** - {iteration} iterations\n\n"
            await asyncio.sleep(0.1)

            # Stream the final answer
            final_answer = (
                response.content
                if hasattr(response, "content") and response.content
                else str(response)
            )
            yield "✨ **Final Answer**:\n\n"
            for k in range(0, len(final_answer), 6):
                chunk = final_answer[k : k + 6]
                yield chunk
                await asyncio.sleep(0.01)

        except Exception as e:
            yield f"❌ **ReAct Error**: {str(e)}"


def create_omnifinder_react_agent(
    llm: ChatOpenRouter, tools: List[BaseTool]
) -> ReActSearchAgent:
    """
    Factory function to create an OmniFinder ReAct agent with proper configuration.
    """
    return ReActSearchAgent(llm=llm, tools=tools)
