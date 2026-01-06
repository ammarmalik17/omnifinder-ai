"""
Logging utilities for OmniFinder AI system.

This module provides standardized logging for the search agent system,
following industry best practices for maintainability and debugging.
"""
import logging
import sys
from typing import Any
from functools import wraps
import time


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


class AgentLogger:
    """
    A centralized logger for the OmniFinder AI agent system.
    """
    
    def __init__(self, name: str = "OmniFinder", level: int = logging.INFO):
        self.logger = setup_logger(name, level)
    
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
    
    def log_query_processing(self, query: str, method: str = "traditional"):
        """Log query processing information."""
        self.logger.info(f"Processing query with {method} method: {query[:50]}...")
    
    def log_tool_usage(self, tool_name: str, query: str):
        """Log tool usage."""
        self.logger.info(f"Using tool '{tool_name}' for query: {query[:50]}...")
    
    def log_search_results(self, tool_name: str, num_results: int):
        """Log search results information."""
        self.logger.info(f"Tool '{tool_name}' returned {num_results} results")