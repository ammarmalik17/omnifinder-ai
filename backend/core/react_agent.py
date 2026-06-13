"""
ReAct (Reasoning and Acting) Agent Implementation for OmniFinder AI.

This module implements the ReAct pattern for search agents, which alternates between
reasoning about the problem and taking actions with tools to solve complex queries.
"""

import time
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from backend.prompts import react_prompt, build_tool_descriptions

from backend.utils.logger import AgentLogger


class ReActSearchAgent:
    """
    A ReAct (Reasoning and Acting) based search agent that can dynamically
    reason about queries and take actions with multiple tools to find answers.
    """

    def __init__(self, llm: BaseChatModel, tools: List[BaseTool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.logger = AgentLogger("ReActAgent")

        # Track consecutive failures per tool to break retry loops
        self._tool_failures: Dict[str, int] = {}

        # Create a system prompt that includes tool descriptions for manual tool selection
        # This avoids using bind_tools() which requires OpenRouter endpoint support
        self.chain = react_prompt | self.llm

    @staticmethod
    def _filter_tool_args(tool: BaseTool, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strip unknown keyword arguments that the tool does not accept.

        Inspects the tool's args schema and keeps only the keys that match.
        This prevents the LLM from inventing parameters like 'domain' or 'source'
        that don't exist on the tool.

        Args:
            tool: The tool to validate arguments against
            args: The arguments dict from the LLM

        Returns:
            Filtered dict containing only valid arguments for this tool
        """
        valid_keys = set(tool.args.keys())
        filtered = {k: v for k, v in args.items() if k in valid_keys}
        if not filtered:
            # If nothing matched, fall back to a default 'query' argument
            filtered = {"query": args.get("query", "")}
        return filtered

    def process_query(self, query: str, sub_queries: List[str] = None) -> Dict[str, Any]:
        """
        Process a query using the ReAct pattern.

        Args:
            query: The user's query
            sub_queries: Optional list of sub-questions (for compound queries)

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

        # If sub_queries were provided by classifier, inject them as context
        if sub_queries:
            plan_hint = (
                "This query has been identified as a compound query with multiple sub-questions. "
                "Address each one systematically:\n"
            )
            for i, sq in enumerate(sub_queries, 1):
                plan_hint += f"  {i}. {sq}\n"
            plan_hint += "\nCreate a PLAN covering all sub-questions, then execute them one by one."
            messages.append(HumanMessage(content=plan_hint))
            self.logger.info(f"📋 Injected {len(sub_queries)} sub-questions for planning")

        # Track the steps for transparency
        steps = []

        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        # Track whether a PLAN has been emitted yet
        plan_emitted = False

        # Build tool descriptions with exact JSON schema for the prompt
        tool_descriptions = build_tool_descriptions(self.tools)

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

                        # Execute the tool with argument validation and failure tracking
                        tool_exec_start = time.time()
                        if tool_name in self.tools:
                            self.logger.info(f"▶️ Executing tool: {tool_name}")
                            tool = self.tools[tool_name]

                            # Validate args: strip any keys the tool doesn't accept
                            validated_args = self._filter_tool_args(tool, tool_args)
                            if validated_args != tool_args:
                                self.logger.info(
                                    f"🔧 Filtered invalid args: {tool_args} -> {validated_args}"
                                )

                            try:
                                tool_result = tool.invoke(validated_args)
                                tool_exec_duration = time.time() - tool_exec_start
                                self.logger.info(
                                    f"✅ Tool execution successful in {tool_exec_duration:.3f}s (result: {len(tool_result)} chars)"
                                )
                                self.logger.log_react_iteration(
                                    iteration, "tool_executed", tool_name
                                )
                                # Reset failure counter on success
                                self._tool_failures[tool_name] = 0
                            except Exception as e:
                                tool_exec_duration = time.time() - tool_exec_start
                                tool_result = f"Error executing {tool_name}: {str(e)}"
                                self.logger.error(
                                    f"❌ Tool execution error after {tool_exec_duration:.3f}s: {tool_result}"
                                )
                                # Increment failure counter
                                self._tool_failures[tool_name] = (
                                    self._tool_failures.get(tool_name, 0) + 1
                                )
                                # If same tool failed 3+ times, inject a hard constraint
                                if self._tool_failures[tool_name] >= 3:
                                    valid_keys = list(tool.args.keys())
                                    hard_stop = (
                                        f"Tool '{tool_name}' has failed {self._tool_failures[tool_name]} consecutive times. "
                                        f"It only accepts these arguments: {valid_keys}. "
                                        f"Do NOT retry '{tool_name}' — use a different tool or provide a final answer."
                                    )
                                    tool_result = f"{tool_result}\n\n{hard_stop}"
                                    self.logger.info(f"🛑 Failure cap reached for {tool_name}: {hard_stop}")
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
                elif "PLAN:" in response_content.upper() and "ACTION:" not in response_content.upper():
                    self.logger.info(f"📋 PLAN detected (iteration {iteration}) — storing plan and continuing")
                    plan_emitted = True
                    steps.append(
                        {
                            "step": iteration,
                            "action": "plan",
                            "content": response_content,
                        }
                    )
                elif "DONE" in response_content.upper():
                    self.logger.info("✅ DONE keyword found — all plan steps complete")
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
                        "✅ No ACTION/PLAN/DONE found — assuming final answer"
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

        # Post-loop evidence check: if ALL tool results were errors, override to disclaimer
        # This prevents hallucination when every tool call failed.
        tool_results = [s for s in steps if s["action"] == "tool_result"]
        if tool_results and all(
            r["result"].startswith("Error") or r["result"].startswith("Unknown tool")
            for r in tool_results
        ):
            disclaimer = (
                "I was unable to retrieve information from any search tools. "
                "All tool queries returned errors. Please check your query or try again later."
            )
            self.logger.info("🛡️ All tool results were errors — returning disclaimer")
            response_content = disclaimer
            # Also update the last message in the conversation
            if messages and hasattr(messages[-1], "content"):
                messages[-1].content = disclaimer
        else:
            response_content = response.content if hasattr(response, "content") else str(response)

        self.logger.info(f"{'-' * 40}")

        return {
            "final_answer": response_content,
            "steps": steps,
            "iterations": iteration,
            "messages": messages,
        }



def create_omnifinder_react_agent(
    llm: BaseChatModel, tools: List[BaseTool]
) -> ReActSearchAgent:
    """
    Factory function to create an OmniFinder ReAct agent with proper configuration.
    """
    return ReActSearchAgent(llm=llm, tools=tools)
