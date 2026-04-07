"""
Logging utilities for OmniFinder AI system.

This module provides standardized logging for the search agent system,
following industry best practices for maintainability and debugging.
"""
import logging
import sys
from functools import wraps
import time
from datetime import datetime
from typing import Dict, Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with standardized formatting.
    
    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


def log_function_call(logger: logging.Logger = None):
    """
    Decorator to log function calls with execution time.
    
    Args:
        logger: Logger instance to use (default: creates new logger)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = setup_logger(func.__module__)
            
            start_time = time.time()
            logger.info(f"Calling function: {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"Function {func.__name__} completed in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {str(e)}")
                raise
                
        return wrapper
    return decorator


def log_error(logger: logging.Logger = None):
    """
    Decorator to log errors with detailed information.
    
    Args:
        logger: Logger instance to use (default: creates new logger)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = setup_logger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {str(e)}\n"
                    f"Args: {args}\n"
                    f"Kwargs: {kwargs}",
                    exc_info=True
                )
                raise
                
        return wrapper
    return decorator


class TimingContext:
    """Context manager for timing code blocks with detailed logging."""
    
    def __init__(self, logger: 'AgentLogger', operation: str, details: str = ""):
        self.logger = logger
        self.operation = operation
        self.details = details
        self.start_time = None
        self.start_datetime = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.start_datetime = datetime.now()
        self.logger.info(f"⏱️ START: {self.operation} at {self.start_datetime.strftime('%H:%M:%S.%f')[:-3]}")
        if self.details:
            self.logger.info(f"   Details: {self.details}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_datetime = datetime.now()
        duration = end_time - self.start_time
        self.logger.info(f"⏱️ END: {self.operation} completed in {duration:.3f}s at {end_datetime.strftime('%H:%M:%S.%f')[:-3]}")
        if exc_type:
            self.logger.error(f"   ❌ Failed with: {exc_type.__name__}: {exc_val}")
        return False


class AgentLogger:
    """
    A centralized logger for the OmniFinder AI agent system.
    """
    
    def __init__(self, name: str = "OmniFinder", level: int = logging.INFO):
        self.logger = setup_logger(name, level)
        self.step_counter = 0
        self.query_start_time = None
    
    def info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """Log an error message."""
        self.logger.error(message, exc_info=exc_info)
    
    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)
    
    def start_query_timing(self, query: str):
        """Start timing for a complete query processing cycle."""
        self.query_start_time = time.time()
        self.step_counter = 0
        query_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🚀 QUERY START at {query_time}")
        self.logger.info(f"Query: {query}")
        self.logger.info(f"{'='*60}\n")
    
    def end_query_timing(self, success: bool = True):
        """End timing for a complete query processing cycle."""
        if self.query_start_time:
            total_duration = time.time() - self.query_start_time
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            status = "✅ SUCCESS" if success else "❌ FAILED"
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"{status} at {end_time}")
            self.logger.info(f"⏱️ TOTAL QUERY TIME: {total_duration:.3f}s")
            self.logger.info(f"📊 Total steps: {self.step_counter}")
            self.logger.info(f"{'='*60}\n")
            self.query_start_time = None
    
    def log_step(self, step_name: str, details: str = ""):
        """Log a processing step with auto-incrementing counter."""
        self.step_counter += 1
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.logger.info(f"📍 STEP {self.step_counter:02d} [{timestamp}]: {step_name}")
        if details:
            self.logger.info(f"   {details}")
    
    def log_timing_context(self, operation: str, details: str = ""):
        """Return a timing context manager for code blocks."""
        return TimingContext(self, operation, details)
    
    def log_query_processing(self, query: str, method: str = "traditional"):
        """Log query processing information."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.logger.info(f"🔍 Processing query with {method} method at {timestamp}: {query[:50]}...")
    
    def log_tool_usage(self, tool_name: str, query: str):
        """Log tool usage."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.logger.info(f"🔧 Using tool '{tool_name}' at {timestamp} for query: {query[:50]}...")
    
    def log_tool_result(self, tool_name: str, duration: float, result_length: int = 0):
        """Log tool execution result with timing."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.logger.info(f"✅ Tool '{tool_name}' completed at {timestamp} in {duration:.3f}s (result: {result_length} chars)")
    
    def log_tool_error(self, tool_name: str, error: str, duration: float):
        """Log tool execution error with timing."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.logger.error(f"❌ Tool '{tool_name}' failed at {timestamp} after {duration:.3f}s: {error}")
    
    def log_search_results(self, tool_name: str, num_results: int):
        """Log search results information."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.logger.info(f"📊 Tool '{tool_name}' returned {num_results} results at {timestamp}")
    
    def log_classification(self, primary_tool: str, secondary_tools: list, confidence: float):
        """Log query classification results."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        tools_str = f"{primary_tool}" + (f" + {', '.join(secondary_tools)}" if secondary_tools else "")
        self.logger.info(f"🎯 Classification at {timestamp}: Primary={primary_tool}, Confidence={confidence:.2f}, Tools=[{tools_str}]")
    
    def log_synthesis_start(self, num_sources: int):
        """Log start of result synthesis."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.logger.info(f"📝 Starting result synthesis at {timestamp} with {num_sources} sources")
    
    def log_synthesis_complete(self, duration: float, answer_length: int):
        """Log completion of result synthesis."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.logger.info(f"✅ Synthesis completed at {timestamp} in {duration:.3f}s (answer: {answer_length} chars)")
    
    def log_react_iteration(self, iteration: int, action: str, tool: str = None):
        """Log ReAct iteration details."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        tool_info = f" using {tool}" if tool else ""
        self.logger.info(f"🔄 ReAct iteration {iteration} at {timestamp}: {action}{tool_info}")
    
    def log_memory_operation(self, operation: str, message_count: int = 0):
        """Log memory operations."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.logger.info(f"🧠 Memory {operation} at {timestamp} (messages: {message_count})")
