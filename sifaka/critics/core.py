"""
Core critic implementation.

This module provides the CriticCore class which is the main interface
for critics, delegating to specialized components.
"""

from typing import Any, Dict, List, Optional

from .base import BaseCritic
from .models import CriticConfig, CriticMetadata
from .managers.memory import MemoryManager
from .managers.prompt import DefaultPromptManager, PromptManager
from .managers.response import ResponseParser
from .services.critique import CritiqueService
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CriticCore(BaseCritic):
    """
    Core critic implementation that delegates to specialized components.

    This class implements the BaseCritic interface but delegates most of its
    functionality to specialized components for better separation of concerns.
    """

    def __init__(
        self,
        config: CriticConfig,
        llm_provider: Any,
        prompt_manager: Optional[PromptManager] = None,
        response_parser: Optional[ResponseParser] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        """
        Initialize a CriticCore instance.

        Args:
            config: The critic configuration
            llm_provider: The language model provider to use
            prompt_manager: Optional prompt manager to use
            response_parser: Optional response parser to use
            memory_manager: Optional memory manager to use
        """
        super().__init__(config)

        # Create managers
        self._prompt_manager = prompt_manager or self._create_prompt_manager()
        self._response_parser = response_parser or ResponseParser()
        self._memory_manager = memory_manager

        # Create services
        self._critique_service = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=self._prompt_manager,
            response_parser=self._response_parser,
            memory_manager=self._memory_manager,
        )

        # Store the language model provider
        self._model = llm_provider

    def validate(self, text: str) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty
        """
        return self._critique_service.validate(text)

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Improve text based on violations.

        Args:
            text: The text to improve
            violations: List of rule violations

        Returns:
            The improved text

        Raises:
            ValueError: If text is empty
        """
        return self._critique_service.improve(text, violations)

    def critique(self, text: str) -> CriticMetadata:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details

        Raises:
            ValueError: If text is empty
        """
        result = self._critique_service.critique(text)

        # Convert dict to CriticMetadata
        return CriticMetadata(
            score=float(result.get("score", 0.0)),
            feedback=str(result.get("feedback", "")),
            issues=list(result.get("issues", [])),
            suggestions=list(result.get("suggestions", [])),
        )

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text is empty
        """
        return self._critique_service.improve(text, feedback)

    async def avalidate(self, text: str) -> bool:
        """
        Asynchronously validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty
        """
        return await self._critique_service.avalidate(text)

    async def acritique(self, text: str) -> CriticMetadata:
        """
        Asynchronously critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details

        Raises:
            ValueError: If text is empty
        """
        result = await self._critique_service.acritique(text)

        # Convert dict to CriticMetadata
        return CriticMetadata(
            score=float(result.get("score", 0.0)),
            feedback=str(result.get("feedback", "")),
            issues=list(result.get("issues", [])),
            suggestions=list(result.get("suggestions", [])),
        )

    async def aimprove(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Asynchronously improve text based on violations.

        Args:
            text: The text to improve
            violations: List of rule violations

        Returns:
            The improved text

        Raises:
            ValueError: If text is empty
        """
        return await self._critique_service.aimprove(text, violations)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """
        Asynchronously improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text is empty
        """
        return await self._critique_service.aimprove(text, feedback)

    def _create_prompt_manager(self) -> PromptManager:
        """
        Create a prompt manager.

        Returns:
            A prompt manager
        """
        return DefaultPromptManager(self.config)
