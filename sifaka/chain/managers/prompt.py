"""
Prompt manager module for Sifaka.

This module provides the PromptManager class which is responsible for
creating, modifying, and managing prompts.
"""

from typing import List, Optional

from ...utils.logging import get_logger

logger = get_logger(__name__)


class PromptManager:
    """
    Manages prompts for chains.
    
    This class is responsible for creating, modifying, and managing prompts
    for use in chains.
    """
    
    def create_prompt_with_feedback(self, original_prompt: str, feedback: str) -> str:
        """
        Create a new prompt with feedback.
        
        Args:
            original_prompt: The original prompt
            feedback: The feedback to include
            
        Returns:
            A new prompt with feedback
        """
        return f"{original_prompt}\n\nPrevious attempt feedback:\n{feedback}"
        
    def create_prompt_with_history(
        self, original_prompt: str, history: List[str]
    ) -> str:
        """
        Create a new prompt with history.
        
        Args:
            original_prompt: The original prompt
            history: The history to include
            
        Returns:
            A new prompt with history
        """
        history_text = "\n".join(history)
        return f"{original_prompt}\n\nPrevious attempts:\n{history_text}"
        
    def create_prompt_with_context(
        self, original_prompt: str, context: str
    ) -> str:
        """
        Create a new prompt with context.
        
        Args:
            original_prompt: The original prompt
            context: The context to include
            
        Returns:
            A new prompt with context
        """
        return f"Context:\n{context}\n\nPrompt:\n{original_prompt}"
        
    def create_prompt_with_examples(
        self, original_prompt: str, examples: List[str]
    ) -> str:
        """
        Create a new prompt with examples.
        
        Args:
            original_prompt: The original prompt
            examples: The examples to include
            
        Returns:
            A new prompt with examples
        """
        examples_text = "\n".join([f"Example {i+1}: {example}" for i, example in enumerate(examples)])
        return f"{original_prompt}\n\nExamples:\n{examples_text}"
