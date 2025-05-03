"""
Prompt manager for critics.

This module provides the PromptManager class which is responsible for
creating and managing prompts for critics.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import CriticConfig
from ...utils.logging import get_logger

logger = get_logger(__name__)


class PromptManager(ABC):
    """
    Manages prompt creation for critics.

    This class is responsible for creating prompts for validation, critique,
    improvement, and reflection.
    """

    def __init__(self, config: CriticConfig):
        """
        Initialize a PromptManager instance.

        Args:
            config: The critic configuration
        """
        self._config = config

    def create_validation_prompt(self, text: str) -> str:
        """
        Create a prompt for validating text.

        Args:
            text: The text to validate

        Returns:
            A prompt for validating the text
        """
        return self._create_validation_prompt_impl(text)

    def create_critique_prompt(self, text: str) -> str:
        """
        Create a prompt for critiquing text.

        Args:
            text: The text to critique

        Returns:
            A prompt for critiquing the text
        """
        return self._create_critique_prompt_impl(text)

    def create_improvement_prompt(
        self, text: str, feedback: str, reflections: Optional[List[str]] = None
    ) -> str:
        """
        Create a prompt for improving text.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement
            reflections: Optional reflections to include in the prompt

        Returns:
            A prompt for improving the text
        """
        return self._create_improvement_prompt_impl(text, feedback, reflections)

    def create_reflection_prompt(
        self, original_text: str, feedback: str, improved_text: str
    ) -> str:
        """
        Create a prompt for reflecting on an improvement.

        Args:
            original_text: The original text
            feedback: The feedback received
            improved_text: The improved text

        Returns:
            A prompt for reflecting on the improvement
        """
        return self._create_reflection_prompt_impl(original_text, feedback, improved_text)

    @abstractmethod
    def _create_validation_prompt_impl(self, text: str) -> str:
        """
        Implementation of create_validation_prompt.

        Args:
            text: The text to validate

        Returns:
            A prompt for validating the text
        """
        pass

    @abstractmethod
    def _create_critique_prompt_impl(self, text: str) -> str:
        """
        Implementation of create_critique_prompt.

        Args:
            text: The text to critique

        Returns:
            A prompt for critiquing the text
        """
        pass

    @abstractmethod
    def _create_improvement_prompt_impl(
        self, text: str, feedback: str, reflections: Optional[List[str]] = None
    ) -> str:
        """
        Implementation of create_improvement_prompt.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement
            reflections: Optional reflections to include in the prompt

        Returns:
            A prompt for improving the text
        """
        pass

    @abstractmethod
    def _create_reflection_prompt_impl(
        self, original_text: str, feedback: str, improved_text: str
    ) -> str:
        """
        Implementation of create_reflection_prompt.

        Args:
            original_text: The original text
            feedback: The feedback received
            improved_text: The improved text

        Returns:
            A prompt for reflecting on the improvement
        """
        pass


class DefaultPromptManager(PromptManager):
    """
    Default implementation of PromptManager.

    This class provides default implementations of prompt creation methods.
    """

    def _create_validation_prompt_impl(self, text: str) -> str:
        """
        Implementation of create_validation_prompt.

        Args:
            text: The text to validate

        Returns:
            A prompt for validating the text
        """
        return f"""Please Validate the following text:

        TEXT TO VALIDATE:
        {text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        VALID: [true/false]
        REASON: [reason for validation result]

        VALIDATION:"""

    def _create_critique_prompt_impl(self, text: str) -> str:
        """
        Implementation of create_critique_prompt.

        Args:
            text: The text to critique

        Returns:
            A prompt for critiquing the text
        """
        return f"""Please critique the following text:

        TEXT TO CRITIQUE:
        {text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        SCORE: [number between 0 and 1]
        FEEDBACK: [your general feedback]
        ISSUES:
        - [issue 1]
        - [issue 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]

        CRITIQUE:"""

    def _create_improvement_prompt_impl(
        self, text: str, feedback: str, reflections: Optional[List[str]] = None
    ) -> str:
        """
        Implementation of create_improvement_prompt.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement
            reflections: Optional reflections to include in the prompt

        Returns:
            A prompt for improving the text
        """
        prompt = f"""Please improve the following text based on the feedback:

        TEXT TO IMPROVE:
        {text}

        FEEDBACK:
        {feedback}
        """

        if reflections and len(reflections) > 0:
            prompt += "\n\nREFLECTIONS FROM PREVIOUS IMPROVEMENTS:\n"
            for i, reflection in enumerate(reflections):
                prompt += f"{i+1}. {reflection}\n"

        prompt += """
        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        IMPROVED_TEXT: [your improved version of the text]

        IMPROVED VERSION:"""

        return prompt

    def _create_reflection_prompt_impl(
        self, original_text: str, feedback: str, improved_text: str
    ) -> str:
        """
        Implementation of create_reflection_prompt.

        Args:
            original_text: The original text
            feedback: The feedback received
            improved_text: The improved text

        Returns:
            A prompt for reflecting on the improvement
        """
        return f"""Please reflect on the following text improvement:

        ORIGINAL TEXT:
        {original_text}

        FEEDBACK:
        {feedback}

        IMPROVED TEXT:
        {improved_text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        REFLECTION: [your reflection on what was improved and why]

        REFLECTION:"""
