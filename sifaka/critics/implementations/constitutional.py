"""
Constitutional critic module for Sifaka.

This module implements a Constitutional AI approach for critics, which evaluates
responses against a set of human-written principles (a "constitution") and provides
natural language feedback when violations are detected.

Based on Constitutional AI: https://arxiv.org/abs/2212.08073

Example:
    ```python
    from sifaka.critics.implementations.constitutional import create_constitutional_critic
    from sifaka.models.providers import OpenAIProvider

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Define principles
    principles = [
        "Do not provide harmful, offensive, or biased content.",
        "Explain reasoning in a clear and truthful manner.",
        "Respect user autonomy and avoid manipulative language.",
    ]

    # Create a constitutional critic
    critic = create_constitutional_critic(
        llm_provider=provider,
        principles=principles
    )

    # Validate a response
    task = "Explain why some people believe climate change isn't real."
    response = "Climate change is a hoax created by scientists to get funding."
    is_valid = critic.validate(response, metadata={"task": task})
    print(f"Response is valid: {is_valid}")

    # Get critique for a response
    critique = critic.critique(response, metadata={"task": task})
    print(f"Critique: {critique}")

    # Improve a response
    improved_response = critic.improve(response, metadata={"task": task})
    print(f"Improved response: {improved_response}")
    ```
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import PrivateAttr, ConfigDict

from ..base import BaseCritic
from ..config import ConstitutionalCriticConfig
from ..interfaces.critic import TextCritic, TextImprover, TextValidator

# Configure logging
logger = logging.getLogger(__name__)


class ConstitutionalCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """
    A critic that evaluates responses against a list of principles (a "constitution")
    and provides natural language feedback for revision.

    Based on Constitutional AI: https://arxiv.org/abs/2212.08073

    This critic analyzes responses for alignment with specified principles and
    generates critiques when violations are detected.

    ## Architecture

    The ConstitutionalCritic follows a component-based architecture with principles-based evaluation:

    1. **Core Components**
       - **ConstitutionalCritic**: Main class that implements the critic interfaces
       - **PrinciplesManager**: Manages the list of principles (the "constitution")
       - **CritiqueGenerator**: Evaluates responses against principles
       - **ResponseImprover**: Improves responses based on critiques
       - **PromptManager**: Creates specialized prompts for critique and improvement
    """

    # Class constants
    DEFAULT_NAME = "constitutional_critic"
    DEFAULT_DESCRIPTION = "Evaluates responses against principles"

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using direct state
    _state = PrivateAttr(default_factory=lambda: None)

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        description: str = DEFAULT_DESCRIPTION,
        llm_provider: Any = None,
        principles: Optional[List[str]] = None,
        config: Optional[ConstitutionalCriticConfig] = None,
    ) -> None:
        """
        Initialize the constitutional critic.

        Args:
            name: Name of the critic
            description: Description of the critic
            llm_provider: Language model provider to use
            principles: List of principles to evaluate responses against
            config: Optional critic configuration (overrides other parameters)

        Raises:
            ValueError: If llm_provider is None or principles is empty
            TypeError: If llm_provider is not a valid provider
        """
        # Validate required parameters
        if llm_provider is None:
            from pydantic import ValidationError

            raise ValidationError.from_exception_data(
                "Field required",
                [{"loc": ("llm_provider",), "msg": "Field required", "type": "missing"}],
            )

        # Create default config if not provided
        if config is None:
            from ..config import DEFAULT_CONSTITUTIONAL_CONFIG

            config = DEFAULT_CONSTITUTIONAL_CONFIG.model_copy(
                update={"name": name, "description": description}
            )

            # Override principles if provided
            if principles is not None:
                config.principles = principles

        # Initialize base class
        super().__init__(config)

        # Initialize state
        from ...utils.state import CriticState

        self._state = CriticState()

        # Store components in state
        self._state.model = llm_provider
        self._state.cache = {
            "principles": config.principles,
            "critique_prompt_template": config.critique_prompt_template,
            "improvement_prompt_template": config.improvement_prompt_template,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        self._state.initialized = True

    def _format_principles(self) -> str:
        """
        Format principles as a bulleted list.

        Returns:
            Formatted principles as a string
        """
        principles = self._state.cache.get("principles", [])
        return "\n".join(f"- {p}" for p in principles)

    def _get_task_from_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """
        Extract task from metadata.

        Args:
            metadata: Optional metadata dictionary

        Returns:
            Task string

        Raises:
            ValueError: If metadata is None or missing task key
        """
        if metadata is None or "task" not in metadata:
            raise ValueError("metadata must contain a 'task' key")
        return metadata["task"]

    @property
    def config(self) -> ConstitutionalCriticConfig:
        """Get the constitutional critic configuration."""
        return cast(ConstitutionalCriticConfig, self._config)

    def validate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate a response against the principles.

        Args:
            text: The response to validate
            metadata: Optional metadata containing the task

        Returns:
            True if the response is valid, False otherwise

        Raises:
            ValueError: If text is empty or metadata is missing required keys
            RuntimeError: If critic is not properly initialized
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("ConstitutionalCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get critique
        critique = self.critique(text, metadata)

        # Response is valid if there are no issues
        return len(critique.get("issues", [])) == 0

    def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a response against the principles and provide detailed feedback.

        Args:
            text: The response to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty or metadata is missing required keys
            RuntimeError: If critic is not properly initialized
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("ConstitutionalCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Format principles
        principles_text = self._format_principles()

        # Create critique prompt
        prompt = self._state.cache.get("critique_prompt_template", "").format(
            principles=principles_text,
            task=task,
            response=text,
        )

        # Generate critique
        critique_text = self._state.model.generate(
            prompt,
            system_prompt=self._state.cache.get("system_prompt", ""),
            temperature=self._state.cache.get("temperature", 0.7),
            max_tokens=self._state.cache.get("max_tokens", 1000),
        ).strip()

        # Parse critique
        issues = []
        suggestions = []

        # Check if critique indicates no issues
        if any(
            phrase in critique_text.lower()
            for phrase in [
                "no issues",
                "no violations",
                "does not violate",
                "aligns with all principles",
                "adheres to all principles",
            ]
        ):
            score = 1.0
            feedback = "Response aligns with all principles."
        else:
            # There are issues
            score = 0.5  # Default score for responses with issues
            feedback = critique_text

            # Extract issues and suggestions
            lines = critique_text.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    if (
                        "should" in line.lower()
                        or "could" in line.lower()
                        or "recommend" in line.lower()
                    ):
                        suggestions.append(line[2:].strip())
                    else:
                        issues.append(line[2:].strip())

        return {
            "score": score,
            "feedback": feedback,
            "issues": issues,
            "suggestions": suggestions,
        }

    def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Improve a response based on principles.

        Args:
            text: The response to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved response

        Raises:
            ValueError: If text is empty or metadata is missing required keys
            RuntimeError: If critic is not properly initialized
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("ConstitutionalCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Get critique
        critique_result = self.critique(text, metadata)

        # If no issues, return original text
        if not critique_result.get("issues", []):
            return text

        # Format principles
        principles_text = self._format_principles()

        # Create improvement prompt
        prompt = self._state.cache.get("improvement_prompt_template", "").format(
            principles=principles_text,
            task=task,
            response=text,
            critique=critique_result["feedback"],
        )

        # Generate improved response
        improved_text = self._state.model.generate(
            prompt,
            system_prompt=self._state.cache.get("system_prompt", ""),
            temperature=self._state.cache.get("temperature", 0.7),
            max_tokens=self._state.cache.get("max_tokens", 1000),
        ).strip()

        return improved_text

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on specific feedback.

        This method improves the text based on the provided feedback,
        which can be more specific than the general improvements based on principles.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text or feedback is empty
            RuntimeError: If critic is not properly initialized
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("ConstitutionalCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("feedback must be a non-empty string")

        # Format principles
        principles_text = self._format_principles()

        # Create improvement prompt
        prompt = (
            f"You are an AI assistant tasked with ensuring alignment to the following principles:\n\n"
            f"{principles_text}\n\n"
            f"Please improve the following response based on this feedback:\n\n"
            f"Response:\n{text}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Improved response:"
        )

        # Generate improved response
        improved_text = self._state.model.generate(
            prompt,
            system_prompt=self._state.cache.get("system_prompt", ""),
            temperature=self._state.cache.get("temperature", 0.7),
            max_tokens=self._state.cache.get("max_tokens", 1000),
        ).strip()

        return improved_text

    async def avalidate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Asynchronously validate a response against the principles.

        Args:
            text: The response to validate
            metadata: Optional metadata containing the task

        Returns:
            True if the response is valid, False otherwise

        Raises:
            ValueError: If text is empty or metadata is missing required keys
            RuntimeError: If critic is not properly initialized
        """
        # For now, use the synchronous implementation
        return self.validate(text, metadata)

    async def acritique(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously analyze a response against the principles and provide detailed feedback.

        Args:
            text: The response to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty or metadata is missing required keys
            RuntimeError: If critic is not properly initialized
        """
        # For now, use the synchronous implementation
        return self.critique(text, metadata)

    async def aimprove(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Asynchronously improve a response based on principles.

        Args:
            text: The response to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved response

        Raises:
            ValueError: If text is empty or metadata is missing required keys
            RuntimeError: If critic is not properly initialized
        """
        # For now, use the synchronous implementation
        return self.improve(text, metadata)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """
        Asynchronously improve text based on specific feedback.

        This method asynchronously improves the text based on the provided feedback,
        which can be more specific than the general improvements based on principles.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text or feedback is empty
            RuntimeError: If critic is not properly initialized
        """
        # For now, use the synchronous implementation
        return self.improve_with_feedback(text, feedback)


def create_constitutional_critic(
    llm_provider: Any,
    principles: List[str] = None,
    name: str = "constitutional_critic",
    description: str = "Evaluates responses against principles",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    critique_prompt_template: Optional[str] = None,
    improvement_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], ConstitutionalCriticConfig]] = None,
    **kwargs: Any,
) -> ConstitutionalCritic:
    """
    Create a constitutional critic with the given parameters.

    Args:
        llm_provider: Language model provider to use
        principles: List of principles to evaluate responses against
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache
        priority: Priority of the critic
        cost: Cost of using the critic
        system_prompt: System prompt for the model
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        critique_prompt_template: Optional custom template for critique prompts
        improvement_prompt_template: Optional custom template for improvement prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments

    Returns:
        ConstitutionalCritic: Configured constitutional critic
    """
    # Create config if not provided
    if config is None:
        from ..config import DEFAULT_CONSTITUTIONAL_CONFIG

        config = DEFAULT_CONSTITUTIONAL_CONFIG.model_copy()

        # Update config with provided values
        updates = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if principles is not None:
            updates["principles"] = principles
        if system_prompt is not None:
            updates["system_prompt"] = system_prompt
        if temperature is not None:
            updates["temperature"] = temperature
        if max_tokens is not None:
            updates["max_tokens"] = max_tokens
        if min_confidence is not None:
            updates["min_confidence"] = min_confidence
        if max_attempts is not None:
            updates["max_attempts"] = max_attempts
        if cache_size is not None:
            updates["cache_size"] = cache_size
        if priority is not None:
            updates["priority"] = priority
        if cost is not None:
            updates["cost"] = cost
        if critique_prompt_template is not None:
            updates["critique_prompt_template"] = critique_prompt_template
        if improvement_prompt_template is not None:
            updates["improvement_prompt_template"] = improvement_prompt_template

        config = config.model_copy(update=updates)
    elif isinstance(config, dict):
        from ..config import ConstitutionalCriticConfig

        config = ConstitutionalCriticConfig(**config)

    # Create and return the critic
    return ConstitutionalCritic(
        name=name,
        description=description,
        llm_provider=llm_provider,
        principles=principles,
        config=config,
    )
