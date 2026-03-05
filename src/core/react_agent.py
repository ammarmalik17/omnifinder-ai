"""
ReAct (Reasoning and Acting) Agent Implementation for OmniFinder AI.

This module implements the ReAct pattern for search agents, which alternates between
reasoning about the problem and taking actions with tools to solve complex queries.
"""
from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter


class ReActSearchAgent:
    """
    A ReAct (Reasoning and Acting) based search agent that can dynamically
    reason about queries and take actions with multiple tools to find answers.
    """
    
    def __init__(self, llm: ChatOpenRouter, tools: List[BaseTool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        
        # Create a system prompt that includes tool descriptions for manual tool selection
        # This avoids using bind_tools() which requires OpenRouter endpoint support
        self.react_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are OmniFinder AI, an intelligent search agent that uses the ReAct (Reasoning and Acting) framework to answer queries.

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

Your response should be informative, well-structured, and cite sources when possible."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
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
        print("\n=== ReAct Agent Starting ===")
        print(f"Query: {query}")
        print(f"Available tools: {list(self.tools.keys())}")
        
        # Initialize conversation with the user query
        messages = [HumanMessage(content=query)]
        
        # Track the steps for transparency
        steps = []
        
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        # Build tool descriptions for the prompt
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in self.tools.values()
        ])
        
        print(f"\nTool Descriptions:\n{tool_descriptions}\n")
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            try:
                # Get the next message from the LLM
                print(f"Invoking LLM with {len(messages)} messages...")
                response = self.chain.invoke({
                    "messages": messages,
                    "tool_descriptions": tool_descriptions
                })
                print("LLM response received")
                
                # Add the AI's response to the conversation
                messages.append(response)
                
                # Parse the response to check for tool usage
                response_content = response.content if hasattr(response, 'content') else str(response)
                print(f"\nResponse content:\n{response_content[:500]}..." if len(response_content) > 500 else f"\nResponse content:\n{response_content}")
                
                # Check if the response contains an action (tool call)
                if "ACTION:" in response_content.upper():
                    print("\n✓ Detected ACTION keyword - parsing tool call")
                    # Parse the action and arguments
                    action_line = None
                    args_line = None
                    
                    lines = response_content.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().upper().startswith('ACTION:'):
                            action_line = line
                            # Look for ARGS in the next few lines
                            for j in range(i+1, min(i+4, len(lines))):
                                if lines[j].strip().upper().startswith('ARGS:'):
                                    args_line = lines[j]
                                    break
                            break
                    
                    if action_line:
                        # Extract tool name
                        tool_name = action_line.split(':', 1)[1].strip().lower()
                        print(f"Tool name extracted: {tool_name}")
                        
                        # Extract arguments
                        tool_args = {}
                        if args_line:
                            import json
                            try:
                                args_str = args_line.split(':', 1)[1].strip()
                                tool_args = json.loads(args_str)
                                print(f"Arguments parsed: {tool_args}")
                            except Exception as e:
                                print(f"JSON parsing failed: {e}, using default query")
                                # If JSON parsing fails, use the query as the argument
                                tool_args = {'query': query}
                        else:
                            print("No ARGS found, using default query")
                            tool_args = {'query': query}
                        
                        # Log the step
                        steps.append({
                            "step": iteration,
                            "action": "tool_call",
                            "tool": tool_name,
                            "arguments": tool_args
                        })
                        
                        # Execute the tool
                        if tool_name in self.tools:
                            print(f"Executing tool: {tool_name}")
                            tool = self.tools[tool_name]
                            try:
                                tool_result = tool._run(**tool_args)
                                print("Tool execution successful")
                            except Exception as e:
                                tool_result = f"Error executing {tool_name}: {str(e)}"
                                print(f"Tool execution error: {tool_result}")
                        else:
                            tool_result = f"Unknown tool: {tool_name}"
                            print(f"Tool not found: {tool_name}")
                        
                        # Add the tool result as a HumanMessage (since it's new input for the model)
                        tool_message = HumanMessage(
                            content=f"Observation from {tool_name}: {tool_result}"
                        )
                        messages.append(tool_message)
                        
                        # Log the result
                        steps.append({
                            "step": iteration,
                            "action": "tool_result",
                            "tool": tool_name,
                            "result": tool_result
                        })
                        print("Tool result added to messages")
                    else:
                        print("No valid action found, generating final response")
                        # No valid action found, generate final response
                        steps.append({
                            "step": iteration,
                            "action": "final_response",
                            "content": response_content
                        })
                        break
                else:
                    print("No ACTION keyword found - assuming final answer")
                    # No action keyword found, assume this is the final answer
                    steps.append({
                        "step": iteration,
                        "action": "final_response",
                        "content": response_content
                    })
                    break
                    
            except Exception as e:
                print(f"\n❌ ERROR in ReAct loop at iteration {iteration}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                import traceback
                print(f"\nFull traceback:\n{traceback.format_exc()}")
                raise
        
        print("\n=== ReAct Agent Completed ===")
        print(f"Total iterations: {iteration}")
        print(f"Steps taken: {len(steps)}")
        
        return {
            "final_answer": response.content,
            "steps": steps,
            "iterations": iteration,
            "messages": messages
        }


def create_omnifinder_react_agent(llm: ChatOpenRouter, tools: List[BaseTool]) -> ReActSearchAgent:
    """
    Factory function to create an OmniFinder ReAct agent with proper configuration.
    """
    return ReActSearchAgent(llm=llm, tools=tools)