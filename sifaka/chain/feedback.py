"""
Feedback formatter module for Sifaka.

This module provides the FeedbackFormatter class which is responsible for
formatting feedback from critique details for inclusion in prompts.
"""

from typing import Dict, Any


class FeedbackFormatter:
    """
    Formats feedback for the model.
    
    This class is responsible for extracting and formatting feedback from
    critique details for inclusion in prompts.
    """
    
    def format_feedback(self, critique_details: Dict[str, Any]) -> str:
        """
        Format feedback from critique details.
        
        Args:
            critique_details: The critique details to extract feedback from
            
        Returns:
            Formatted feedback string
        """
        if isinstance(critique_details, dict) and "feedback" in critique_details:
            return critique_details["feedback"]
        return ""
        
    def create_prompt_with_feedback(self, original_prompt: str, feedback: str) -> str:
        """
        Create a new prompt with feedback.
        
        Args:
            original_prompt: The original prompt
            feedback: The feedback to include
            
        Returns:
            New prompt with feedback
        """
        return f"{original_prompt}\n\nPrevious attempt feedback:\n{feedback}"
