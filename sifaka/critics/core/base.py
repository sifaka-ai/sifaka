"""Simplified base critic implementation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from ...core.models import CritiqueResult, SifakaResult
from ...core.interfaces import Critic
from ...core.llm_client import LLMClient, LLMManager, Provider
from ...core.cache import get_global_cache

from .config import CriticConfig
from .response_parser import ResponseParser, CriticResponse
from .confidence import ConfidenceCalculator


class BaseCritic(Critic, ABC):
    """Simplified base critic with clear separation of concerns."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = None,
        config: Optional[CriticConfig] = None,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize critic with configuration."""
        self.config = config or CriticConfig()
        self.model = model
        self.temperature = temperature or self.config.temperature
        self.provider = provider
        self._api_key = api_key
        self._client: Optional[LLMClient] = None
        
        # Components
        self._parser = ResponseParser()
        self._confidence_calc = ConfidenceCalculator(self.config.base_confidence)
    
    @property
    def client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._client is None:
            self._client = LLMManager.get_client(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                api_key=self._api_key
            )
        return self._client
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the critic's name."""
        pass
    
    @abstractmethod
    async def _create_messages(self, text: str, result: SifakaResult) -> List[Dict[str, str]]:
        """Create messages for the LLM.
        
        Args:
            text: Text to critique
            result: Current result with history
            
        Returns:
            List of messages for LLM
        """
        pass
    
    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Critique the text with caching support.
        
        Args:
            text: Text to critique
            result: Current result with history
            
        Returns:
            CritiqueResult with feedback and suggestions
        """
        # Check cache
        cache = get_global_cache()
        if cache:
            cached = await cache.get_critique(
                text=text,
                critic_name=self.name,
                model=self.model,
                temperature=self.temperature,
                iteration=result.iteration
            )
            if cached:
                return CritiqueResult(**cached)
        
        try:
            # Generate critique
            messages = await self._create_messages(text, result)
            
            # Add JSON format instruction
            messages.append({
                "role": "system",
                "content": self._get_format_instruction()
            })
            
            # Call LLM
            response = await self.client.complete(
                messages,
                timeout=self.config.timeout_seconds
            )
            
            # Parse response
            critic_response = self._parser.parse(response.content, "json")
            
            # Calculate confidence if not provided
            if critic_response.confidence == 0.7:  # Default value
                critic_response.confidence = self._confidence_calc.calculate(
                    feedback=critic_response.feedback,
                    suggestions=critic_response.suggestions,
                    response_length=len(response.content),
                    metadata=critic_response.metadata
                )
            
            # Create result
            result = CritiqueResult(
                critic=self.name,
                feedback=critic_response.feedback,
                suggestions=critic_response.suggestions,
                needs_improvement=critic_response.needs_improvement,
                confidence=critic_response.confidence,
                metadata=critic_response.metadata
            )
            
            # Cache result
            if cache:
                await cache.set_critique(
                    text=text,
                    critic_name=self.name,
                    model=self.model,
                    temperature=self.temperature,
                    critique_data=result.model_dump(),
                    iteration=result.iteration
                )
            
            return result
            
        except Exception as e:
            # Return error result
            return CritiqueResult(
                critic=self.name,
                feedback=f"Error during critique: {str(e)}",
                suggestions=["Please review the text manually"],
                needs_improvement=True,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _get_format_instruction(self) -> str:
        """Get JSON format instruction."""
        return """Please provide your response in the following JSON format:
{
    "feedback": "Your main feedback about the text",
    "suggestions": ["Specific suggestion 1", "Specific suggestion 2", ...],
    "needs_improvement": true/false,
    "confidence": 0.0-1.0  // Your confidence in this assessment
}"""
    
    def _get_previous_context(self, result: SifakaResult) -> str:
        """Get context from previous critiques."""
        if not result.critiques:
            return ""
        
        # Get recent critiques from this critic
        recent = []
        for critique in list(result.critiques)[-self.config.context_window:]:
            if critique.critic == self.name:
                recent.append(f"- {critique.feedback}")
        
        if not recent:
            return ""
        
        return f"\n\nPrevious feedback:\n" + "\n".join(recent) + \
               "\n\nPlease provide NEW insights and avoid repetition."


def create_prompt_with_format(
    base_prompt: str, response_format: str = "json", include_examples: bool = True
) -> str:
    """Create a prompt with JSON format instructions.
    
    Args:
        base_prompt: The base prompt text
        response_format: Format type (json only)
        include_examples: Whether to include examples
        
    Returns:
        Formatted prompt with instructions
    """
    format_instruction = """
Please provide your response in the following JSON format:
{
    "feedback": "Main feedback about the text",
    "suggestions": ["Suggestion 1", "Suggestion 2", ...],
    "needs_improvement": true/false,
    "confidence": 0.0-1.0  // Your confidence in this assessment
}"""
    
    prompt = base_prompt + "\n\n" + format_instruction
    
    if include_examples:
        prompt += """

Example response:
{
    "feedback": "The text provides a good overview but lacks specific examples",
    "suggestions": ["Add concrete examples", "Include data to support claims"],
    "needs_improvement": true,
    "confidence": 0.85
}"""
    
    return prompt