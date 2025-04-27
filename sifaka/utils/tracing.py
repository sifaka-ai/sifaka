"""
Tracing utilities for Sifaka.
"""
from typing import Dict, Any, List, Optional
import json
import os
import datetime

class Tracer:
    """
    Tracer for Sifaka.
    
    A tracer records the steps of the reflection process for debugging and auditing.
    
    Args:
        trace_dir (Optional[str]): Directory to save trace files
        in_memory (bool): Whether to keep traces in memory
    """
    
    def __init__(
        self, 
        trace_dir: Optional[str] = None,
        in_memory: bool = True
    ):
        self.trace_dir = trace_dir
        self.in_memory = in_memory
        self.traces = {}
        
        if trace_dir and not os.path.exists(trace_dir):
            os.makedirs(trace_dir)
    
    def start_trace(self, trace_id: Optional[str] = None) -> str:
        """
        Start a new trace.
        
        Args:
            trace_id (Optional[str]): ID for the trace
            
        Returns:
            str: The trace ID
        """
        if trace_id is None:
            trace_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        self.traces[trace_id] = {
            "id": trace_id,
            "start_time": datetime.datetime.now().isoformat(),
            "events": []
        }
        
        return trace_id
    
    def add_event(
        self, 
        trace_id: str, 
        event_type: str, 
        data: Dict[str, Any]
    ) -> None:
        """
        Add an event to a trace.
        
        Args:
            trace_id (str): The trace ID
            event_type (str): The type of event
            data (Dict[str, Any]): The event data
        """
        if trace_id not in self.traces:
            self.start_trace(trace_id)
        
        event = {
            "type": event_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "data": data
        }
        
        self.traces[trace_id]["events"].append(event)
    
    def end_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        End a trace and save it if a trace directory is configured.
        
        Args:
            trace_id (str): The trace ID
            
        Returns:
            Dict[str, Any]: The trace data
        """
        if trace_id not in self.traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        self.traces[trace_id]["end_time"] = datetime.datetime.now().isoformat()
        
        if self.trace_dir:
            trace_path = os.path.join(self.trace_dir, f"{trace_id}.json")
            with open(trace_path, "w") as f:
                json.dump(self.traces[trace_id], f, indent=2)
        
        trace_data = self.traces[trace_id]
        
        if not self.in_memory:
            del self.traces[trace_id]
        
        return trace_data
    
    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        Get a trace by ID.
        
        Args:
            trace_id (str): The trace ID
            
        Returns:
            Dict[str, Any]: The trace data
        """
        if trace_id not in self.traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        return self.traces[trace_id]
