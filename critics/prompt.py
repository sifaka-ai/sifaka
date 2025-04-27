"""Implementation of prompt-based critics."""

from typing import Dict, Any, Optional
from pydantic import Field

from .base import Critic
from .protocols import LLMProvider, PromptFactory


class DefaultPromptFactory(PromptFactory):
    """Default implementation of prompt factory."""

    def create_critique_prompt(self, text: str) -> str:
        return f"Critique the following text: {text}"

    def create_validation_prompt(self, text: str) -> str:
        return f"Validate the following text: {text}"

    def create_improvement_prompt(self, text: str, feedback: str) -> str:
        return f"Improve the following text based on feedback:\nText: {text}\nFeedback: {feedback}"


class PromptCritic(Critic):
    """A critic that uses prompts and an LLM to perform its functions."""

    llm_provider: LLMProvider = Field(..., description="Language model provider")
    prompt_factory: PromptFactory = Field(
        default_factory=DefaultPromptFactory, description="Factory for creating prompts"
    )

    def validate(self, text: str) -> bool:
        """Validate text using the LLM."""
        prompt = self.prompt_factory.create_validation_prompt(text)
        result = self.llm_provider.invoke(prompt)
        return self._parse_validation_result(result)

    def critique(self, text: str) -> Dict[str, Any]:
        """Critique text using the LLM."""
        prompt = self.prompt_factory.create_critique_prompt(text)
        result = self.llm_provider.invoke(prompt)
        return self._parse_critique_result(result)

    def improve(self, text: str, feedback: str) -> str:
        """Improve text using the LLM."""
        prompt = self.prompt_factory.create_improvement_prompt(text, feedback)
        result = self.llm_provider.invoke(prompt)
        return self._parse_improvement_result(result)

    async def avalidate(self, text: str) -> bool:
        """Async version of validate."""
        prompt = self.prompt_factory.create_validation_prompt(text)
        result = await self.llm_provider.ainvoke(prompt)
        return self._parse_validation_result(result)

    async def acritique(self, text: str) -> Dict[str, Any]:
        """Async version of critique."""
        prompt = self.prompt_factory.create_critique_prompt(text)
        result = await self.llm_provider.ainvoke(prompt)
        return self._parse_critique_result(result)

    async def aimprove(self, text: str, feedback: str) -> str:
        """Async version of improve."""
        prompt = self.prompt_factory.create_improvement_prompt(text, feedback)
        result = await self.llm_provider.ainvoke(prompt)
        return self._parse_improvement_result(result)

    def _parse_validation_result(self, result: Dict[str, Any]) -> bool:
        """Parse validation result from LLM response."""
        # Implementation depends on LLM response format
        return bool(result.get("valid", False))

    def _parse_critique_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse critique result from LLM response."""
        # Implementation depends on LLM response format
        return {
            "score": float(result.get("score", 0.0)),
            "feedback": str(result.get("feedback", "")),
            "issues": list(result.get("issues", [])),
            "suggestions": list(result.get("suggestions", [])),
        }

    def _parse_improvement_result(self, result: Dict[str, Any]) -> str:
        """Parse improvement result from LLM response."""
        # Implementation depends on LLM response format
        return str(result.get("improved_text", ""))
