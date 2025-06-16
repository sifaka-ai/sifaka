"""ToolCall model for Sifaka.

This module contains the ToolCall model that tracks tool execution
with timing and result information for observability.

Extracted from the monolithic thought.py file to improve maintainability.
"""

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel


class ToolCall(BaseModel):
    """Record of a tool call made during generation.

    Tracks tool execution with timing and result information for observability.
    
    This model tracks:
    - Tool identification and arguments
    - Execution results and timing
    - Iteration tracking for multi-round workflows
    - Timestamp for audit trail
    
    Example:
        ```python
        tool_call = ToolCall(
            iteration=1,
            tool_name="web_search",
            args={"query": "renewable energy statistics", "max_results": 5},
            result={"results": [...], "total_found": 1250},
            execution_time=2.35,
            timestamp=datetime.now()
        )
        ```
    """

    iteration: int
    tool_name: str
    args: Dict[str, Any]
    result: Any
    execution_time: float  # in seconds
    timestamp: datetime

    def get_execution_time_ms(self) -> float:
        """Get execution time in milliseconds.
        
        Returns:
            Execution time in milliseconds
        """
        return self.execution_time * 1000

    def is_slow_execution(self, threshold_seconds: float = 5.0) -> bool:
        """Check if this tool call was slow.
        
        Args:
            threshold_seconds: Threshold for considering execution slow
            
        Returns:
            True if execution time exceeds threshold
        """
        return self.execution_time > threshold_seconds

    def was_successful(self) -> bool:
        """Check if the tool call was successful.
        
        Returns:
            True if result doesn't contain error information
        """
        if self.result is None:
            return False
            
        # Check for common error patterns
        if isinstance(self.result, dict):
            error_fields = ["error", "exception", "failure", "failed"]
            return not any(field in self.result for field in error_fields)
            
        if isinstance(self.result, str):
            error_keywords = ["error", "exception", "failed", "failure"]
            result_lower = self.result.lower()
            return not any(keyword in result_lower for keyword in error_keywords)
            
        return True

    def get_error_message(self) -> str:
        """Get error message if the tool call failed.
        
        Returns:
            Error message, or empty string if successful
        """
        if self.was_successful():
            return ""
            
        if isinstance(self.result, dict):
            error_fields = ["error", "exception", "message", "failure"]
            for field in error_fields:
                if field in self.result:
                    return str(self.result[field])
                    
        if isinstance(self.result, str):
            return self.result
            
        return "Unknown error occurred"

    def get_result_size(self) -> Dict[str, Any]:
        """Get information about the result size.
        
        Returns:
            Dictionary with result size information
        """
        import sys
        
        size_bytes = sys.getsizeof(self.result)
        
        result_info = {
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2),
            "type": type(self.result).__name__
        }
        
        # Add type-specific information
        if isinstance(self.result, (list, tuple)):
            result_info["item_count"] = len(self.result)
        elif isinstance(self.result, dict):
            result_info["key_count"] = len(self.result)
        elif isinstance(self.result, str):
            result_info["character_count"] = len(self.result)
            
        return result_info

    def get_args_summary(self) -> Dict[str, Any]:
        """Get summary of tool arguments.
        
        Returns:
            Dictionary with argument summary
        """
        return {
            "arg_count": len(self.args),
            "arg_names": list(self.args.keys()),
            "has_complex_args": any(
                isinstance(v, (dict, list)) for v in self.args.values()
            )
        }

    def get_performance_category(self) -> str:
        """Get performance category based on execution time.
        
        Returns:
            Performance category (very_fast, fast, normal, slow, very_slow)
        """
        if self.execution_time < 0.1:
            return "very_fast"
        elif self.execution_time < 0.5:
            return "fast"
        elif self.execution_time < 2.0:
            return "normal"
        elif self.execution_time < 10.0:
            return "slow"
        else:
            return "very_slow"

    def extract_result_metrics(self) -> Dict[str, Any]:
        """Extract metrics from the tool result.
        
        Returns:
            Dictionary with extracted metrics
        """
        metrics = {}
        
        if not isinstance(self.result, dict):
            return metrics
            
        # Look for common metric patterns
        metric_patterns = [
            "count", "total", "size", "length", "duration",
            "score", "confidence", "accuracy", "precision", "recall"
        ]
        
        for key, value in self.result.items():
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in metric_patterns):
                if isinstance(value, (int, float)):
                    metrics[key] = value
                    
        return metrics

    def get_tool_category(self) -> str:
        """Get tool category based on tool name.
        
        Returns:
            Tool category (search, retrieval, analysis, generation, other)
        """
        tool_lower = self.tool_name.lower()
        
        if any(keyword in tool_lower for keyword in ["search", "query", "find"]):
            return "search"
        elif any(keyword in tool_lower for keyword in ["retrieve", "fetch", "get"]):
            return "retrieval"
        elif any(keyword in tool_lower for keyword in ["analyze", "process", "compute"]):
            return "analysis"
        elif any(keyword in tool_lower for keyword in ["generate", "create", "build"]):
            return "generation"
        else:
            return "other"

    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of this tool call.
        
        Returns:
            Dictionary with tool call summary
        """
        return {
            "tool_name": self.tool_name,
            "tool_category": self.get_tool_category(),
            "iteration": self.iteration,
            "execution_time_seconds": self.execution_time,
            "execution_time_ms": self.get_execution_time_ms(),
            "performance_category": self.get_performance_category(),
            "was_successful": self.was_successful(),
            "error_message": self.get_error_message() if not self.was_successful() else None,
            "args_count": len(self.args),
            "result_size": self.get_result_size(),
            "extracted_metrics": self.extract_result_metrics(),
            "timestamp": self.timestamp
        }
