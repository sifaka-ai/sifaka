"""
Self-Refine critic module for Sifaka.

This module implements the Self-Refine approach for critics, which enables language models
to iteratively critique and revise their own outputs without requiring external feedback.
The critic uses the same language model to generate critiques and revisions in multiple rounds.

Based on Self-Refine: https://arxiv.org/abs/2303.17651

Example:
    ```python
    from sifaka.critics.implementations.self_refine import create_self_refine_critic
    from sifaka.models.providers import OpenAIProvider

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Create a self-refine critic
    critic = create_self_refine_critic(
        llm_provider=provider,
        max_iterations=3
    )

    # Use the critic to improve text
    task = "Write a concise explanation of quantum computing."
    initial_output = "Quantum computing uses quantum bits."
    improved_output = critic.improve(initial_output, {"task": task})
    ```
"""

from typing import Any, Dict, List, Optional, Union, cast

from pydantic import Field, ConfigDict, PrivateAttr

from ..base import BaseCritic
from ..config import SelfRefineCriticConfig
from ..interfaces.critic import TextCritic, TextImprover, TextValidator


class SelfRefineCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """
    A critic that implements the Self-Refine approach for iterative self-improvement.

    This critic uses the same language model to critique and revise its own outputs
    in multiple iterations, leading to progressively improved results.

    Based on Self-Refine: https://arxiv.org/abs/2303.17651

    ## Lifecycle Management

    The SelfRefineCritic manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up language model provider
       - Initializes state
       - Allocates resources

    2. **Operation**
       - Validates text
       - Critiques text
       - Improves text through multiple iterations
       - Tracks improvements

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status
    """

    # Class constants
    DEFAULT_NAME = "self_refine_critic"
    DEFAULT_DESCRIPTION = "Improves text through iterative self-critique and revision"

    # State management using direct state
    _state = PrivateAttr(default_factory=lambda: None)

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        description: str = DEFAULT_DESCRIPTION,
        llm_provider: Any = None,
        config: Optional[SelfRefineCriticConfig] = None,
    ) -> None:
        """
        Initialize the self-refine critic.

        Args:
            name: Name of the critic
            description: Description of the critic
            llm_provider: Language model provider to use
            config: Optional critic configuration

        Raises:
            ValueError: If llm_provider is None
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
            from ..config import DEFAULT_SELF_REFINE_CONFIG

            config = DEFAULT_SELF_REFINE_CONFIG.model_copy(
                update={"name": name, "description": description}
            )

        # Initialize base class
        super().__init__(config)

        # Initialize state
        from ...utils.state import CriticState

        self._state = CriticState()

        # Store components in state
        self._state.model = llm_provider
        self._state.cache = {
            "max_iterations": config.max_iterations,
            "critique_prompt_template": config.critique_prompt_template or (
                "Please critique the following response to the task. "
                "Focus on accuracy, clarity, and completeness.\n\n"
                "Task:\n{task}\n\n"
                "Response:\n{response}\n\n"
                "Critique:"
            ),
            "revision_prompt_template": config.revision_prompt_template or (
                "Please revise the following response based on the critique.\n\n"
                "Task:\n{task}\n\n"
                "Response:\n{response}\n\n"
                "Critique:\n{critique}\n\n"
                "Revised response:"
            ),
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        self._state.initialized = True

    def _check_input(self, text: str) -> None:
        """
        Validate input text and initialization state.

        Args:
            text: The text to validate

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if not self._state.initialized:
            raise RuntimeError("SelfRefineCritic not properly initialized")

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
    def config(self) -> SelfRefineCriticConfig:
        """Get the self-refine critic configuration."""
        return cast(SelfRefineCriticConfig, self._config)

    def validate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate
            metadata: Optional metadata containing the task

        Returns:
            True if the text passes validation, False otherwise

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("SelfRefineCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Create critique prompt
        prompt = self._state.cache.get("critique_prompt_template", "").format(
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

        # Check if critique indicates no issues
        no_issues_phrases = [
            "no issues",
            "looks good",
            "well written",
            "excellent",
            "great job",
            "perfect",
        ]
        return any(phrase in critique_text.lower() for phrase in no_issues_phrases)

    def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze text and provide detailed feedback.

        Args:
            text: The text to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("SelfRefineCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Create critique prompt
        prompt = self._state.cache.get("critique_prompt_template", "").format(
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

        # Extract issues and suggestions from critique
        for line in critique_text.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                if (
                    "should" in line.lower()
                    or "could" in line.lower()
                    or "recommend" in line.lower()
                ):
                    suggestions.append(line[2:])
                else:
                    issues.append(line[2:])

        # Calculate score based on issues
        score = 1.0 if not issues else max(0.0, 1.0 - (len(issues) * 0.1))

        return {
            "score": score,
            "feedback": critique_text,
            "issues": issues,
            "suggestions": suggestions,
        }

    def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Improve text through iterative self-critique and revision.

        Args:
            text: The text to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("SelfRefineCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Get max iterations from config
        max_iterations = self._state.cache.get("max_iterations", 3)

        # Start with the original text
        current_output = text

        # Perform iterative refinement
        for _ in range(max_iterations):
            # Step 1: Critique the current output
            critique_prompt = self._state.cache.get("critique_prompt_template", "").format(
                task=task,
                response=current_output,
            )

            critique = self._state.model.generate(
                critique_prompt,
                system_prompt=self._state.cache.get("system_prompt", ""),
                temperature=self._state.cache.get("temperature", 0.7),
                max_tokens=self._state.cache.get("max_tokens", 1000),
            ).strip()

            # Heuristic stopping condition
            no_issues_phrases = [
                "no issues",
                "looks good",
                "well written",
                "excellent",
                "great job",
                "perfect",
            ]
            if any(phrase in critique.lower() for phrase in no_issues_phrases):
                return current_output

            # Step 2: Revise using the critique
            revision_prompt = self._state.cache.get("revision_prompt_template", "").format(
                task=task,
                response=current_output,
                critique=critique,
            )

            revised_output = self._state.model.generate(
                revision_prompt,
                system_prompt=self._state.cache.get("system_prompt", ""),
                temperature=self._state.cache.get("temperature", 0.7),
                max_tokens=self._state.cache.get("max_tokens", 1000),
            ).strip()

            # Check if there's no improvement
            if revised_output == current_output:
                return current_output

            # Update current output
            current_output = revised_output

        return current_output

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on specific feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            Improved text

        Raises:
            ValueError: If text or feedback is empty
            RuntimeError: If critic is not properly initialized
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("SelfRefineCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("feedback must be a non-empty string")

        # Create revision prompt with the provided feedback
        revision_prompt = self._state.cache.get("revision_prompt_template", "").format(
            task="Improve the following text",
            response=text,
            critique=feedback,
        )

        # Generate improved response
        improved_text = self._state.model.generate(
            revision_prompt,
            system_prompt=self._state.cache.get("system_prompt", ""),
            temperature=self._state.cache.get("temperature", 0.7),
            max_tokens=self._state.cache.get("max_tokens", 1000),
        ).strip()

        return improved_text

    # Async methods
    async def avalidate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Asynchronously validate text against quality standards."""
        # For simplicity, we'll use the synchronous implementation for now
        return self.validate(text, metadata)

    async def acritique(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Asynchronously analyze text and provide detailed feedback."""
        # For simplicity, we'll use the synchronous implementation for now
        return self.critique(text, metadata)

    async def aimprove(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Asynchronously improve text through iterative self-critique and revision."""
        # For simplicity, we'll use the synchronous implementation for now
        return self.improve(text, metadata)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """Asynchronously improve text based on specific feedback."""
        # For simplicity, we'll use the synchronous implementation for now
        return self.improve_with_feedback(text, feedback)


def create_self_refine_critic(
    llm_provider: Any,
    name: str = "self_refine_critic",
    description: str = "Improves text through iterative self-critique and revision",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    max_iterations: int = None,
    critique_prompt_template: Optional[str] = None,
    revision_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], SelfRefineCriticConfig]] = None,
    **kwargs: Any,
) -> SelfRefineCritic:
    """
    Create a self-refine critic with the given parameters.

    Args:
        llm_provider: Language model provider to use
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
        max_iterations: Maximum number of refinement iterations
        critique_prompt_template: Optional custom template for critique prompts
        revision_prompt_template: Optional custom template for revision prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        SelfRefineCritic: Configured self-refine critic
    """
    # Create config if not provided
    if config is None:
        from ..config import DEFAULT_SELF_REFINE_CONFIG

        config = DEFAULT_SELF_REFINE_CONFIG.model_copy()

        # Update config with provided values
        updates = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
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
        if max_iterations is not None:
            updates["max_iterations"] = max_iterations
        if critique_prompt_template is not None:
            updates["critique_prompt_template"] = critique_prompt_template
        if revision_prompt_template is not None:
            updates["revision_prompt_template"] = revision_prompt_template

        config = config.model_copy(update=updates)
    elif isinstance(config, dict):
        from ..config import SelfRefineCriticConfig

        config = SelfRefineCriticConfig(**config)

    # Create and return the critic
    return SelfRefineCritic(
        name=name,
        description=description,
        llm_provider=llm_provider,
        config=config,
    )
