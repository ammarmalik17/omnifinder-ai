"""
ReAct (Reasoning and Acting) Agent Implementation for OmniFinder AI.

This module implements the ReAct pattern for search agents, which alternates between
reasoning about the problem and taking actions with tools to solve complex queries.
"""
from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq


class ReActSearchAgent:
    """
    A ReAct (Reasoning and Acting) based search agent that can dynamically
    reason about queries and take actions with multiple tools to find answers.
    """
    
    def __init__(self, llm: ChatGroq, tools: List[BaseTool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        
        # Create a tool calling model with structured output
        self.tool_calling_llm = llm.bind_tools(list(self.tools.values()))
        
        # Create the ReAct prompt that guides the reasoning and acting process
        self.react_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are OmniFinder AI, an intelligent search agent that uses the ReAct (Reasoning and Acting) framework to answer queries.

Your approach:
1. REASON: Think step-by-step about the query and what information is needed
2. ACT: Use tools to gather information when needed
3. OBSERVE: Analyze the results from tools
4. REPEAT: Continue reasoning and acting until you have sufficient information
5. FINALIZE: Provide a comprehensive answer

Available tools: {tool_names}

When using tools:
- Always think before acting
- Use the most appropriate tool for the specific information needed
- Wait for tool results before proceeding
- Combine information from multiple sources when relevant

Your response should be informative, well-structured, and cite sources when possible."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Bind tools to the prompt
        self.chain = self.react_prompt | self.tool_calling_llm
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the ReAct pattern.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary containing the final response and intermediate steps
        """
        # Initialize conversation with the user query
        messages = [HumanMessage(content=query)]
        
        # Track the steps for transparency
        steps = []
        
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get the next message from the LLM
            response = self.chain.invoke({
                "messages": messages,
                "tool_names": list(self.tools.keys())
            })
            
            # Add the AI's response to the conversation
            messages.append(response)
            
            # Check if the response contains tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Process each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    # Log the step
                    steps.append({
                        "step": iteration,
                        "action": "tool_call",
                        "tool": tool_name,
                        "arguments": tool_args
                    })
                    
                    # Execute the tool
                    tool = self.tools[tool_name]
                    try:
                        tool_result = tool._run(**tool_args)
                    except Exception as e:
                        tool_result = f"Error executing {tool_name}: {str(e)}"
                    
                    # Add the tool result as a ToolMessage
                    tool_message = ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    )
                    messages.append(tool_message)
                    
                    # Log the result
                    steps.append({
                        "step": iteration,
                        "action": "tool_result",
                        "tool": tool_name,
                        "result": tool_result
                    })
            else:
                # No more tool calls, we have our final answer
                steps.append({
                    "step": iteration,
                    "action": "final_response",
                    "content": response.content
                })
                break
        
        return {
            "final_answer": response.content,
            "steps": steps,
            "iterations": iteration,
            "messages": messages
        }


def create_omnifinder_react_agent(llm: ChatGroq, tools: List[BaseTool]) -> ReActSearchAgent:
    """
    Factory function to create an OmniFinder ReAct agent with proper configuration.
    """
    return ReActSearchAgent(llm=llm, tools=tools)