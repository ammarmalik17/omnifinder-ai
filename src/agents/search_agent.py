from typing import Dict, Any, List
from langchain_groq import ChatGroq
from src.components.query_classifier import QueryClassifier
from src.tools.search_tools import get_all_tools
from src.components.result_synthesizer import ResultSynthesizer
from src.components.conversation_memory import ConversationBufferWindowMemory
from src.core.react_agent import create_omnifinder_react_agent
from src.config.agent_config import AgentConfig
from src.utils.logger import AgentLogger
from langchain_core.messages import BaseMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class SearchAgent:
    """Main search agent that integrates all components."""
    
    def __init__(self, llm: ChatGroq, config: AgentConfig = None):
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
            max_history_messages=self.config.max_history_messages
        )
        
        # Initialize tools
        all_tools = get_all_tools()
        self.tools = {tool.name: tool for tool in all_tools}
        
        # Create a mapping from old names to new names for backward compatibility
        self.tool_name_mapping = {
            "wikipedia_search": "wikipedia",
            "arxiv_search": "arxiv",
            "duckduckgo_search": "duckduckgo",
        }
        
        # Reverse mapping for forward compatibility
        self.reverse_tool_name_mapping = {v: k for k, v in self.tool_name_mapping.items()}
        
        # Initialize ReAct agent for complex queries
        self.react_agent = create_omnifinder_react_agent(llm, list(self.tools.values()))
        
        # Thread lock for thread safety
        self.lock = threading.Lock()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the full pipeline.
        
        Args:
            query: The user's query string
            
        Returns:
            Dictionary containing classification, search results, and synthesized answer
        """
        self.logger.log_query_processing(query, "react" if self._is_complex_query(query, self.query_classifier.classify(query)) else "traditional")
        
        # Classify the query to determine which tools to use
        classification = self.query_classifier.classify(query)
        
        # For complex queries that might benefit from ReAct reasoning, use the ReAct agent
        if self.use_react_for_complex and self._is_complex_query(query, classification):
            self.logger.info(f"Using ReAct pattern for complex query: {query[:50]}...")
            return self._process_with_react(query)
        
        # Determine which tools to use
        tools_to_use = [classification.primary_tool] + classification.secondary_tools
        
        # Execute search tools concurrently
        search_results = self._execute_search_tools(query, tools_to_use)
        
        # Synthesize results into a comprehensive answer
        synthesized_answer = self.result_synthesizer.synthesize(query, search_results)
        
        # Add to memory
        with self.lock:
            self.memory.add_user_message(query)
            self.memory.add_ai_message(synthesized_answer)
        
        return {
            "query": query,
            "classification": classification,
            "search_results": search_results,
            "synthesized_answer": synthesized_answer
        }
    
    def _is_complex_query(self, query: str, classification) -> bool:
        """
        Determine if a query is complex enough to benefit from ReAct reasoning.
        
        Args:
            query: The user's query
            classification: The query classification result
            
        Returns:
            True if the query should use ReAct reasoning, False otherwise
        """
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
    
    def _process_with_react(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the ReAct agent for complex reasoning.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary containing the response from the ReAct agent
        """
        react_result = self.react_agent.process_query(query)
        
        # Format the result to match the expected return format
        return {
            "query": query,
            "classification": None,  # Not applicable for ReAct
            "search_results": [],  # Results are handled within ReAct
            "synthesized_answer": react_result["final_answer"],
            "react_steps": react_result["steps"],  # Include steps for transparency
            "react_iterations": react_result["iterations"]
        }
    
    def _execute_search_tools(self, query: str, tools_to_use: List[str]) -> List[Dict[str, Any]]:
        """
        Execute search tools concurrently to improve performance.
        
        Args:
            query: The user's query
            tools_to_use: List of tool names to execute
            
        Returns:
            List of results from each tool
        """
        search_results = []
        
        # Use ThreadPoolExecutor to run tools concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks to the executor
            future_to_tool = {}
            for tool_name in tools_to_use:
                if tool_name in self.tools:
                    future = executor.submit(self._execute_single_tool, tool_name, query)
                    future_to_tool[future] = tool_name
            
            # Collect results as they complete
            for future in as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                try:
                    result = future.result()
                    search_results.append({
                        "tool_name": tool_name,
                        "content": result
                    })
                except Exception as e:
                    error_msg = f"Error with {tool_name}: {str(e)}"
                    search_results.append({
                        "tool_name": tool_name,
                        "content": error_msg
                    })
        
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
        
        return tool._run(query)
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the current conversation history."""
        return self.memory.get_messages()
    
    def clear_conversation(self):
        """Clear the conversation history."""
        with self.lock:
            self.memory.clear()
    
    def chat(self, query: str) -> str:
        """
        Simple chat interface that returns just the answer.
        
        Args:
            query: The user's query
            
        Returns:
            The synthesized answer
        """
        result = self.process_query(query)
        return result["synthesized_answer"]