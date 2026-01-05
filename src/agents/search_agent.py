from typing import Dict, Any, List
from langchain_groq import ChatGroq
from src.components.query_classifier import QueryClassifier
from src.tools.search_tools import get_all_tools
from src.components.result_synthesizer import ResultSynthesizer
from src.components.conversation_memory import ConversationBufferWindowMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class SearchAgent:
    """Main search agent that integrates all components."""
    
    def __init__(self, llm: ChatGroq, max_workers: int = 4):
        self.llm = llm
        self.max_workers = max_workers
        self.query_classifier = QueryClassifier(llm)
        self.result_synthesizer = ResultSynthesizer(llm)
        self.memory = ConversationBufferWindowMemory(llm)
        
        # Initialize tools
        all_tools = get_all_tools()
        self.tools = {tool.name: tool for tool in all_tools}
        
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
        # Classify the query to determine which tools to use
        classification = self.query_classifier.classify(query)
        
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
        tool = self.tools[tool_name]
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