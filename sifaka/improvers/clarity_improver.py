"""
Clarity improver for Sifaka.

This module provides an improver that enhances the clarity and coherence of text.
"""

from typing import Dict, Any, Tuple, Optional

from sifaka.interfaces import Model, ImprovementResult
from sifaka.registry import register_improver


class ImprovementResultImpl:
    """Implementation of the ImprovementResult interface."""
    
    def __init__(
        self,
        original_text: str,
        improved_text: str,
        changes_made: bool,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an improvement result.
        
        Args:
            original_text: The original text before improvement.
            improved_text: The improved text.
            changes_made: Whether changes were made to the text.
            message: Message describing the result.
            details: Additional details about the improvement result.
        """
        self._original_text = original_text
        self._improved_text = improved_text
        self._changes_made = changes_made
        self._message = message
        self._details = details or {}
    
    @property
    def passed(self) -> bool:
        """Whether the improvement passed."""
        return True
    
    @property
    def original_text(self) -> str:
        """The original text before improvement."""
        return self._original_text
    
    @property
    def improved_text(self) -> str:
        """The improved text."""
        return self._improved_text
    
    @property
    def changes_made(self) -> bool:
        """Whether changes were made to the text."""
        return self._changes_made
    
    @property
    def message(self) -> str:
        """Message describing the result."""
        return self._message
    
    @property
    def details(self) -> Dict[str, Any]:
        """Additional details about the improvement result."""
        return self._details


class ClarityImprover:
    """Improver that enhances the clarity and coherence of text.
    
    This improver uses an LLM to improve the clarity and coherence of text.
    
    Attributes:
        model: The model to use for improvement.
        temperature: The temperature to use for generation.
        system_prompt: The system prompt to use for the model.
    """
    
    def __init__(
        self,
        model: Model,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        """Initialize a clarity improver.
        
        Args:
            model: The model to use for improvement.
            temperature: The temperature to use for generation.
            system_prompt: The system prompt to use for the model.
        """
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You are an expert editor who improves the clarity and coherence of text. "
            "Your goal is to make the text clearer, more coherent, and easier to understand "
            "while preserving the original meaning and intent."
        )
    
    def improve(self, text: str) -> Tuple[str, ImprovementResult]:
        """Improve text clarity and coherence.
        
        Args:
            text: The text to improve.
            
        Returns:
            A tuple of (improved_text, improvement_result).
        """
        if not text.strip():
            return text, ImprovementResultImpl(
                original_text=text,
                improved_text=text,
                changes_made=False,
                message="Empty text cannot be improved",
                details={"improver": "clarity"},
            )
        
        # Create the prompt for improvement
        prompt = (
            "Please improve the clarity and coherence of the following text. "
            "Make it clearer, more coherent, and easier to understand while "
            "preserving the original meaning and intent.\n\n"
            f"Text: {text}\n\n"
            "Improved text:"
        )
        
        # Generate improved text
        improved_text = self.model.generate(
            prompt,
            temperature=self.temperature,
            system_message=self.system_prompt,
        )
        
        # Check if text was actually improved
        changes_made = improved_text.strip() != text.strip()
        
        return improved_text, ImprovementResultImpl(
            original_text=text,
            improved_text=improved_text,
            changes_made=changes_made,
            message="Text clarity and coherence improved" if changes_made else "No changes needed",
            details={"improver": "clarity"},
        )


@register_improver("clarity")
def create_clarity_improver(
    model: Model,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> ClarityImprover:
    """Create a clarity improver.
    
    Args:
        model: The model to use for improvement.
        temperature: The temperature to use for generation.
        system_prompt: The system prompt to use for the model.
        
    Returns:
        A clarity improver.
    """
    return ClarityImprover(model, temperature, system_prompt)
