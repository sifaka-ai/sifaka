"""
Integration with Anthropic's API for text reflection and analysis.

This module provides integration with Anthropic's API for:
- Text reflection and analysis
- Content safety checks
- Style and tone analysis
"""

from typing import Any, Dict, Optional

from anthropic import Anthropic
from pydantic import BaseModel

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ReflectionResult(BaseModel):
    """Result of a text reflection operation."""

    text: str
    analysis: Dict[str, Any]
    suggestions: Optional[Dict[str, Any]] = None
    safety_score: Optional[float] = None


class AnthropicReflector:
    """Reflector that uses Anthropic's API for text analysis."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize the Anthropic reflector.

        Args:
            api_key: Anthropic API key
            model: Model to use for reflection
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def reflect(self, text: str) -> ReflectionResult:
        """
        Reflect on the given text using Anthropic's API.

        Args:
            text: Text to reflect on

        Returns:
            ReflectionResult containing analysis and suggestions
        """
        try:
            # Prepare the prompt
            prompt = f"""Please analyze the following text and provide:
1. A detailed analysis of its content, style, and tone
2. Suggestions for improvement
3. A safety score (0-1) indicating potential issues

Text to analyze:
{text}

Please provide your analysis in a structured format."""

            # Call the API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Parse the response
            analysis = {
                "content": response.content[0].text,
                "model": self.model,
                "temperature": self.temperature,
            }

            return ReflectionResult(
                text=text,
                analysis=analysis,
                safety_score=0.8,  # Placeholder - would need to parse from response
            )

        except Exception as e:
            logger.error(f"Error in Anthropic reflection: {str(e)}")
            raise