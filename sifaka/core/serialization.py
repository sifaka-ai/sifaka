"""Serialization utilities for Sifaka.

This module contains utilities for serializing and deserializing conversation history
and other complex data structures with memory optimization.

Extracted from the monolithic thought.py file to improve maintainability.
"""

import json
import pickle
import gzip
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel


class ConversationHistory:
    """Manages conversation history with memory optimization and serialization."""
    
    def __init__(self, max_size: Optional[int] = None, compress: bool = True):
        """Initialize conversation history manager.
        
        Args:
            max_size: Maximum number of messages to keep (None for unlimited)
            compress: Whether to compress stored data
        """
        self.messages: List[Union[Dict, str]] = []
        self.max_size = max_size
        self.compress = compress
        self._metadata = {
            "created_at": datetime.now(),
            "total_messages_processed": 0,
            "compression_enabled": compress
        }
    
    def add_message(self, message: Union[Dict, str]) -> None:
        """Add a message to the conversation history.
        
        Args:
            message: Message to add (dict or string)
        """
        self.messages.append(message)
        self._metadata["total_messages_processed"] += 1
        
        # Enforce size limit
        if self.max_size and len(self.messages) > self.max_size:
            self.messages = self.messages[-self.max_size:]
    
    def get_messages(self) -> List[Union[Dict, str]]:
        """Get all messages in the conversation history.
        
        Returns:
            List of messages
        """
        return self.messages.copy()
    
    def get_recent_messages(self, count: int) -> List[Union[Dict, str]]:
        """Get the most recent messages.
        
        Args:
            count: Number of recent messages to return
            
        Returns:
            List of recent messages
        """
        return self.messages[-count:] if count > 0 else []
    
    def clear(self) -> None:
        """Clear all messages from history."""
        self.messages.clear()
    
    def get_size_info(self) -> Dict[str, Any]:
        """Get information about conversation history size.
        
        Returns:
            Dictionary with size information
        """
        import sys
        
        total_size = sys.getsizeof(self.messages)
        for msg in self.messages:
            total_size += sys.getsizeof(msg)
            
        return {
            "message_count": len(self.messages),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 4),
            "avg_message_size": total_size / len(self.messages) if self.messages else 0,
            "compression_enabled": self.compress
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "messages": self.messages,
            "max_size": self.max_size,
            "compress": self.compress,
            "metadata": self._metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationHistory":
        """Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ConversationHistory instance
        """
        instance = cls(
            max_size=data.get("max_size"),
            compress=data.get("compress", True)
        )
        instance.messages = data.get("messages", [])
        instance._metadata = data.get("metadata", {})
        return instance


class ThoughtSerializer:
    """Handles serialization of SifakaThought objects with optimization."""
    
    @staticmethod
    def serialize_to_json(thought: BaseModel, pretty: bool = False) -> str:
        """Serialize thought to JSON string.
        
        Args:
            thought: SifakaThought object to serialize
            pretty: Whether to format JSON for readability
            
        Returns:
            JSON string representation
        """
        data = thought.model_dump()
        
        # Convert datetime objects to ISO strings
        data = ThoughtSerializer._convert_datetimes(data)
        
        if pretty:
            return json.dumps(data, indent=2, default=str)
        else:
            return json.dumps(data, default=str)
    
    @staticmethod
    def serialize_to_file(thought: BaseModel, file_path: Union[str, Path], 
                         format: str = "json", compress: bool = False) -> None:
        """Serialize thought to file.
        
        Args:
            thought: SifakaThought object to serialize
            file_path: Path to save file
            format: Serialization format ("json" or "pickle")
            compress: Whether to compress the file
        """
        file_path = Path(file_path)
        
        if format == "json":
            data = ThoughtSerializer.serialize_to_json(thought, pretty=True)
            content = data.encode('utf-8')
        elif format == "pickle":
            content = pickle.dumps(thought)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if compress:
            content = gzip.compress(content)
            file_path = file_path.with_suffix(file_path.suffix + ".gz")
        
        file_path.write_bytes(content)
    
    @staticmethod
    def deserialize_from_file(file_path: Union[str, Path], 
                            thought_class: type) -> BaseModel:
        """Deserialize thought from file.
        
        Args:
            file_path: Path to file
            thought_class: Class to deserialize to
            
        Returns:
            Deserialized thought object
        """
        file_path = Path(file_path)
        content = file_path.read_bytes()
        
        # Check if compressed
        if file_path.suffix == ".gz":
            content = gzip.decompress(content)
            file_path = file_path.with_suffix("")
        
        # Determine format from extension
        if file_path.suffix == ".json":
            data = json.loads(content.decode('utf-8'))
            return thought_class.model_validate(data)
        elif file_path.suffix in [".pkl", ".pickle"]:
            return pickle.loads(content)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    @staticmethod
    def _convert_datetimes(obj: Any) -> Any:
        """Convert datetime objects to ISO strings recursively.
        
        Args:
            obj: Object to convert
            
        Returns:
            Object with datetime objects converted to strings
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: ThoughtSerializer._convert_datetimes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ThoughtSerializer._convert_datetimes(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def get_serialization_stats(thought: BaseModel) -> Dict[str, Any]:
        """Get statistics about thought serialization.
        
        Args:
            thought: SifakaThought object
            
        Returns:
            Dictionary with serialization statistics
        """
        import sys
        
        # Get object size
        object_size = sys.getsizeof(thought.model_dump())
        
        # Get JSON size
        json_data = ThoughtSerializer.serialize_to_json(thought)
        json_size = len(json_data.encode('utf-8'))
        
        # Get compressed JSON size
        compressed_size = len(gzip.compress(json_data.encode('utf-8')))
        
        # Get pickle size
        pickle_size = len(pickle.dumps(thought))
        
        return {
            "object_size_bytes": object_size,
            "json_size_bytes": json_size,
            "compressed_json_size_bytes": compressed_size,
            "pickle_size_bytes": pickle_size,
            "compression_ratio": round(compressed_size / json_size, 3),
            "json_vs_pickle_ratio": round(json_size / pickle_size, 3),
            "recommended_format": "pickle" if pickle_size < json_size * 0.7 else "json"
        }


class MemoryOptimizer:
    """Utilities for optimizing memory usage in thought objects."""
    
    @staticmethod
    def optimize_conversation_history(history: List[Union[Dict, str]], 
                                    max_messages: int = 50) -> List[Union[Dict, str]]:
        """Optimize conversation history by keeping only recent messages.
        
        Args:
            history: Conversation history to optimize
            max_messages: Maximum number of messages to keep
            
        Returns:
            Optimized conversation history
        """
        if len(history) <= max_messages:
            return history
            
        # Keep the most recent messages
        return history[-max_messages:]
    
    @staticmethod
    def compress_large_strings(obj: Any, min_size: int = 1000) -> Any:
        """Compress large strings in an object.
        
        Args:
            obj: Object to process
            min_size: Minimum string size to compress
            
        Returns:
            Object with large strings compressed
        """
        if isinstance(obj, str) and len(obj) >= min_size:
            compressed = gzip.compress(obj.encode('utf-8'))
            return {
                "_compressed": True,
                "_original_size": len(obj),
                "_data": compressed
            }
        elif isinstance(obj, dict):
            return {k: MemoryOptimizer.compress_large_strings(v, min_size) 
                   for k, v in obj.items()}
        elif isinstance(obj, list):
            return [MemoryOptimizer.compress_large_strings(item, min_size) 
                   for item in obj]
        else:
            return obj
    
    @staticmethod
    def decompress_strings(obj: Any) -> Any:
        """Decompress strings that were compressed by compress_large_strings.
        
        Args:
            obj: Object to decompress
            
        Returns:
            Object with strings decompressed
        """
        if isinstance(obj, dict) and obj.get("_compressed"):
            compressed_data = obj["_data"]
            return gzip.decompress(compressed_data).decode('utf-8')
        elif isinstance(obj, dict):
            return {k: MemoryOptimizer.decompress_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [MemoryOptimizer.decompress_strings(item) for item in obj]
        else:
            return obj
