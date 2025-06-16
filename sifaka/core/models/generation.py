"""Generation model for Sifaka.

This module contains the Generation model that tracks text generation operations
with PydanticAI metadata for complete traceability.

Extracted from the monolithic thought.py file to improve maintainability.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Generation(BaseModel):
    """Record of a text generation operation.

    Captures the generated text along with PydanticAI metadata for complete traceability.
    
    This model tracks:
    - Generated text and metadata
    - Complete conversation history with PydanticAI integration
    - Cost and usage information
    - Iteration tracking for multi-round workflows
    
    Example:
        ```python
        generation = Generation(
            iteration=1,
            text="Renewable energy comes from natural sources...",
            model="openai:gpt-4",
            timestamp=datetime.now(),
            conversation_history=[...],
            cost=0.002,
            usage={"prompt_tokens": 50, "completion_tokens": 100}
        )
        ```
    """

    iteration: int
    text: str
    model: str
    timestamp: datetime
    conversation_history: List[Union[Dict, str]] = Field(
        default_factory=list,
        description="Complete conversation history including requests and responses",
    )
    cost: Optional[float] = None
    usage: Optional[Dict] = None

    def get_token_count(self) -> Dict[str, int]:
        """Extract token count information from usage data.
        
        Returns:
            Dictionary with token counts, or empty dict if not available
        """
        if not self.usage:
            return {}
            
        # Handle different usage formats
        token_info = {}
        if isinstance(self.usage, dict):
            # Common token field names
            for prompt_key in ["prompt_tokens", "input_tokens", "tokens_prompt"]:
                if prompt_key in self.usage:
                    token_info["prompt_tokens"] = self.usage[prompt_key]
                    break
                    
            for completion_key in ["completion_tokens", "output_tokens", "tokens_completion"]:
                if completion_key in self.usage:
                    token_info["completion_tokens"] = self.usage[completion_key]
                    break
                    
            for total_key in ["total_tokens", "tokens_total"]:
                if total_key in self.usage:
                    token_info["total_tokens"] = self.usage[total_key]
                    break
                    
        return token_info

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation history.
        
        Returns:
            Dictionary with conversation statistics
        """
        if not self.conversation_history:
            return {"message_count": 0, "total_length": 0}
            
        message_count = len(self.conversation_history)
        total_length = 0
        
        for msg in self.conversation_history:
            if isinstance(msg, str):
                total_length += len(msg)
            elif isinstance(msg, dict):
                # Try to get text content from various possible fields
                for text_field in ["content", "text", "message", "body"]:
                    if text_field in msg and isinstance(msg[text_field], str):
                        total_length += len(msg[text_field])
                        break
                        
        return {
            "message_count": message_count,
            "total_length": total_length,
            "avg_message_length": total_length / message_count if message_count > 0 else 0
        }

    def has_error(self) -> bool:
        """Check if this generation contains error information.
        
        Returns:
            True if conversation history contains error information
        """
        if not self.conversation_history:
            return False
            
        for msg in self.conversation_history:
            if isinstance(msg, dict) and "error" in msg:
                return True
                
        return False

    def get_errors(self) -> List[str]:
        """Extract error messages from conversation history.
        
        Returns:
            List of error messages found in conversation history
        """
        errors = []
        
        if not self.conversation_history:
            return errors
            
        for msg in self.conversation_history:
            if isinstance(msg, dict) and "error" in msg:
                errors.append(str(msg["error"]))
                
        return errors

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information for this generation.
        
        Returns:
            Dictionary with memory usage statistics
        """
        import sys
        
        text_size = sys.getsizeof(self.text)
        conversation_size = sys.getsizeof(self.conversation_history)
        
        # Calculate conversation history size more accurately
        for msg in self.conversation_history:
            conversation_size += sys.getsizeof(msg)
            
        usage_size = sys.getsizeof(self.usage) if self.usage else 0
        total_size = sys.getsizeof(self.model_dump())
        
        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 4),
            "components": {
                "text": {
                    "size_bytes": text_size,
                    "character_count": len(self.text)
                },
                "conversation_history": {
                    "size_bytes": conversation_size,
                    "message_count": len(self.conversation_history)
                },
                "usage": {
                    "size_bytes": usage_size
                }
            }
        }
