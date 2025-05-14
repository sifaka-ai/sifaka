"""
Prompt Factory Module for Sifaka.

This module provides specialized prompt manager implementations for different
critic types, including:
- SelfRefineCriticPromptManager: For self-refine critics
- ConstitutionalCriticPromptManager: For constitutional critics
- PromptCriticPromptManager: For prompt critics

These prompt managers extend the base PromptManager class and provide
specialized prompt creation methods for their respective critic types.

## Usage Examples
```python
from sifaka.core.managers.prompt_factories import SelfRefineCriticPromptManager
from sifaka.utils.config.critics import SelfRefineCriticConfig

# Create a config
config = SelfRefineCriticConfig(
    name="self_refine_critic",
    description="Improves text through iterative self-critique and revision",
    max_iterations=3
)

# Create a prompt manager
prompt_manager = SelfRefineCriticPromptManager(config)

# Create a critique prompt
critique_prompt = prompt_manager.create_critique_prompt("This is a test.") if prompt_manager else ""
```
"""

from typing import Any, Dict, List, Optional, Union
import time

from pydantic import Field, ConfigDict, PrivateAttr

from .prompt import CriticPromptManager, DefaultPromptManager, PromptCriticPromptManager
from ...utils.state import StateManager
from ...utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "SelfRefineCriticPromptManager",
    "ConstitutionalCriticPromptManager",
    "PromptCriticPromptManager",
]


class SelfRefineCriticPromptManager(DefaultPromptManager):
    """
    Prompt manager for self-refine critics.

    This class provides specialized prompt creation methods for self-refine critics,
    which enable language models to iteratively critique and revise their own outputs.

    ## Lifecycle Management

    The SelfRefineCriticPromptManager manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up templates
       - Initializes state
       - Allocates resources

    2. **Operation**
       - Creates critique prompts
       - Creates revision prompts
       - Creates reflection prompts
       - Manages prompt history

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the prompt manager."""
        super().__init__(config)
        if self._state_manager:
            self._state_manager.update(
                "critique_prompt_template", getattr(config, "critique_prompt_template", None)
            )
            self._state_manager.update(
                "revision_prompt_template", getattr(config, "revision_prompt_template", None)
            )

    def create_critique_prompt(self, text: str) -> str:
        """
        Create a prompt for critiquing text.

        Args:
            text: The text to critique

        Returns:
            A prompt for critiquing the text

        Raises:
            ValueError: If text is empty
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        # Use custom template if available
        template = (
            self._state_manager.get("critique_prompt_template") if self._state_manager else None
        )
        if template:
            return template.format(response=text, task="Critique the following text")

        # Default template
        return f"""Please critique the following text. Focus on accuracy, clarity, and completeness.

Text:
{text}

Critique:"""

    def create_improvement_prompt(
        self, text: str, feedback: str, reflections: Optional[List[str]] = None
    ) -> str:
        """
        Create a prompt for improving text based on feedback.

        Args:
            text: The text to improve
            feedback: The feedback to use for improvement
            reflections: Optional list of reflections to include

        Returns:
            A prompt for improving the text

        Raises:
            ValueError: If text or feedback is empty
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        if not feedback or not isinstance(feedback, str):
            raise ValueError("Feedback must be a non-empty string")

        # Use custom template if available
        template = (
            self._state_manager.get("revision_prompt_template") if self._state_manager else None
        )
        if template:
            return template.format(
                response=text, critique=feedback, task="Improve the following text"
            )

        # Default template
        prompt = f"""Please revise the following text based on the critique.

Text:
{text}

Critique:
{feedback}

Revised text:"""

        # Add reflections if available
        if reflections and len(reflections) > 0:
            reflections_text = "\n".join(reflections)
            prompt += f"\n\nPrevious reflections:\n{reflections_text}"

        return prompt


class ConstitutionalCriticPromptManager(DefaultPromptManager):
    """
    Prompt manager for constitutional critics.

    This class provides specialized prompt creation methods for constitutional critics,
    which evaluate and improve text based on a set of principles.

    ## Lifecycle Management

    The ConstitutionalCriticPromptManager manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up principles
       - Initializes state
       - Allocates resources

    2. **Operation**
       - Creates critique prompts
       - Creates improvement prompts
       - Creates validation prompts
       - Manages prompt history

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the prompt manager."""
        super().__init__(config)
        if self._state_manager:
            self._state_manager.update("principles", getattr(config, "principles", []))
            self._state_manager.update(
                "critique_prompt_template", getattr(config, "critique_prompt_template", None)
            )
            self._state_manager.update(
                "improvement_prompt_template", getattr(config, "improvement_prompt_template", None)
            )

    def _format_principles(self) -> str:
        """
        Format principles as a bulleted list.

        Returns:
            Formatted principles as a string
        """
        principles = self._state_manager.get("principles", []) if self._state_manager else []
        return "\n".join(f"- {p}" for p in principles)

    def create_critique_prompt(self, text: str) -> str:
        """
        Create a prompt for critiquing text against principles.

        Args:
            text: The text to critique

        Returns:
            A prompt for critiquing the text

        Raises:
            ValueError: If text is empty
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        # Use custom template if available
        template = (
            self._state_manager.get("critique_prompt_template") if self._state_manager else None
        )
        if template:
            return template.format(text=text, principles=self._format_principles() if self else "")

        # Default template
        return f"""You are an AI assistant tasked with ensuring alignment to the following principles:

{self._format_principles() if self else ""}

Please critique the following text based on these principles:

{text}

Critique:"""

    def create_improvement_prompt(
        self, text: str, feedback: str, reflections: Optional[List[str]] = None
    ) -> str:
        """
        Create a prompt for improving text based on feedback and principles.

        Args:
            text: The text to improve
            feedback: The feedback to use for improvement
            reflections: Optional list of reflections to include

        Returns:
            A prompt for improving the text

        Raises:
            ValueError: If text or feedback is empty
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        if not feedback or not isinstance(feedback, str):
            raise ValueError("Feedback must be a non-empty string")

        # Use custom template if available
        template = (
            self._state_manager.get("improvement_prompt_template") if self._state_manager else None
        )
        if template:
            return template.format(
                text=text, feedback=feedback, principles=self._format_principles() if self else ""
            )

        # Default template
        prompt = f"""You are an AI assistant tasked with ensuring alignment to the following principles:

{self._format_principles() if self else ""}

Please improve the following text based on this feedback:

Text:
{text}

Feedback:
{feedback}

Improved text:"""

        # Add reflections if available
        if reflections and len(reflections) > 0:
            reflections_text = "\n".join(reflections)
            prompt += f"\n\nPrevious reflections:\n{reflections_text}"

        return prompt
