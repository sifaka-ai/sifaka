"""
Self-Refine critic module for Sifaka.

This module implements the Self-Refine approach for critics, which enables language models
to iteratively critique and revise their own outputs without requiring external feedback.
The critic uses the same language model to generate critiques and revisions in multiple rounds.

## Overview
The SelfRefineCritic is a specialized implementation of the critic interface
that enables language models to iteratively improve their own outputs through
self-critique and revision. It implements a multi-round refinement process
where the model critiques its own output and then revises it based on that
critique, leading to progressively improved results without external feedback.

## Components
- **SelfRefineCritic**: Main class implementing TextValidator, TextImprover, and TextCritic
- **create_self_refine_critic**: Factory function for creating SelfRefineCritic instances
- **PromptManager**: Creates prompts for critique and revision phases
- **ResponseParser**: Parses and validates model responses

## Architecture
The SelfRefineCritic follows an iterative refinement architecture:
- Uses standardized state management with _state_manager
- Implements a multi-round refinement process
- Provides automatic stopping criteria based on critique quality
- Implements both sync and async interfaces
- Provides comprehensive error handling and recovery
- Tracks performance and iteration statistics

## Usage Examples
```python
from sifaka.critics.implementations.self_refine import create_self_refine_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a self-refine critic
critic = create_self_refine_critic(
    llm_provider=provider,
    max_iterations=3,
    temperature=0.7,
    system_prompt="You are an expert editor focused on clarity and accuracy."
)

# Use the critic to improve text
task = "Write a concise explanation of quantum computing."
initial_output = "Quantum computing uses quantum bits."
improved_output = critic.improve(initial_output, {"task": task})
print(f"Improved output: {improved_output}")

# Get critique for text
critique = critic.critique(initial_output, {"task": task})
print(f"Score: {critique['score']}")
print(f"Feedback: {critique['feedback']}")

# Validate text
is_valid = critic.validate(initial_output, {"task": task})
print(f"Is valid: {is_valid}")

# Improve with specific feedback
feedback = "The explanation should include more details about qubits."
improved_output = critic.improve_with_feedback(initial_output, feedback)
```

## Error Handling
The module implements comprehensive error handling for:
- Input validation (empty text, invalid types)
- Initialization errors (missing provider, invalid config)
- Processing errors (model failures, timeout issues)
- Resource management (cleanup, state preservation)

## References
Based on Self-Refine: https://arxiv.org/abs/2303.17651
"""

import json
import time
from typing import Any, Dict, Optional, Union

from pydantic import Field, ConfigDict, PrivateAttr

from ...core.base import BaseComponent
from ...utils.state import create_critic_state
from ...utils.logging import get_logger
from ...core.base import BaseResult as CriticResult
from sifaka.utils.config.critics import SelfRefineCriticConfig
from ...interfaces.critic import TextCritic, TextImprover, TextValidator

# Configure logging
logger = get_logger(__name__)


class SelfRefineCritic(BaseComponent[str, CriticResult], TextValidator, TextImprover, TextCritic):
    """
    A critic that implements the Self-Refine approach for iterative self-improvement.

    This critic uses the same language model to critique and revise its own outputs
    in multiple iterations, leading to progressively improved results. It implements
    the TextValidator, TextImprover, and TextCritic interfaces to provide a comprehensive
    set of text analysis capabilities with iterative self-refinement.

    Based on Self-Refine: https://arxiv.org/abs/2303.17651

    ## Architecture
    The SelfRefineCritic follows an iterative refinement architecture:
    - Uses standardized state management with _state_manager
    - Implements a multi-round refinement process
    - Provides automatic stopping criteria based on critique quality
    - Implements both sync and async interfaces
    - Provides comprehensive error handling and recovery
    - Tracks performance and iteration statistics

    ## Lifecycle
    1. **Initialization**: Set up with configuration and dependencies
       - Create/validate config
       - Initialize language model provider
       - Set up prompt manager
       - Initialize memory manager
       - Set up state tracking

    2. **Operation**: Process text through various methods
       - validate(): Check if text meets quality standards
       - critique(): Analyze text and provide detailed feedback
       - improve(): Enhance text through multiple iterations of self-critique and revision
       - improve_with_feedback(): Enhance text based on specific feedback

    3. **Cleanup**: Manage resources and finalize state
       - Release resources
       - Reset state
       - Log final status
       - Track performance metrics

    ## Examples
    ```python
    from sifaka.critics.implementations.self_refine import create_self_refine_critic
    from sifaka.models.providers import OpenAIProvider

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Create a self-refine critic
    critic = create_self_refine_critic(
        llm_provider=provider,
        max_iterations=3,
        temperature=0.7
    )

    # Use the critic to improve text
    task = "Write a concise explanation of quantum computing."
    initial_output = "Quantum computing uses quantum bits."
    improved_output = critic.improve(initial_output, {"task": task})
    ```
    """

    # Class constants
    DEFAULT_NAME = "self_refine_critic"
    DEFAULT_DESCRIPTION = "Improves text through iterative self-critique and revision"

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    config: SelfRefineCriticConfig = Field(description="Critic configuration")

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        config: Optional[SelfRefineCriticConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the self-refine critic.

        Args:
            name: Name of the critic
            description: Description of the critic
            llm_provider: Language model provider to use
            config: Optional critic configuration
            **kwargs: Additional configuration parameters

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        # Create default config if not provided
        if config is None:
            from sifaka.utils.config.critics import DEFAULT_SELF_REFINE_CRITIC_CONFIG

            config = DEFAULT_SELF_REFINE_CRITIC_CONFIG.model_copy(
                update={"name": name, "description": description, **kwargs}
            )

        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        try:
            # Import required components
            from sifaka.core.managers.prompt_factories import SelfRefineCriticPromptManager
            from ..managers.response import ResponseParser
            from sifaka.core.managers.memory import BufferMemoryManager as MemoryManager
            from ..services.critique import CritiqueService

            # Store components in state
            self._state_manager.update("model", llm_provider)
            self._state_manager.update("prompt_manager", SelfRefineCriticPromptManager(config))
            self._state_manager.update("response_parser", ResponseParser())
            self._state_manager.update(
                "memory_manager", MemoryManager(buffer_size=10)  # Default buffer size
            )

            # Create service and store in state cache
            cache = self._state_manager.get("cache", {})
            cache["critique_service"] = CritiqueService(
                llm_provider=llm_provider,
                prompt_manager=self._state_manager.get("prompt_manager"),
                response_parser=self._state_manager.get("response_parser"),
                memory_manager=self._state_manager.get("memory_manager"),
            )
            cache["max_iterations"] = config.max_iterations
            cache["critique_prompt_template"] = config.critique_prompt_template or (
                "Please critique the following response to the task. "
                "Focus on accuracy, clarity, and completeness.\n\n"
                "Task:\n{task}\n\n"
                "Response:\n{response}\n\n"
                "Critique:"
            )
            cache["revision_prompt_template"] = config.revision_prompt_template or (
                "Please revise the following response based on the critique.\n\n"
                "Task:\n{task}\n\n"
                "Response:\n{response}\n\n"
                "Critique:\n{critique}\n\n"
                "Revised response:"
            )
            cache["system_prompt"] = config.system_prompt
            cache["temperature"] = config.temperature
            cache["max_tokens"] = config.max_tokens
            self._state_manager.update("cache", cache)

            # Mark as initialized
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("component_type", self.__class__.__name__)
            self._state_manager.set_metadata("initialization_time", time.time())
        except Exception as e:
            self.record_error(e)
            raise ValueError(f"Failed to initialize SelfRefineCritic: {str(e)}") from e

    def process(self, input: str) -> CriticResult:
        """
        Process the input text and return a result.

        This is the main method required by BaseComponent.

        Args:
            input: The text to process

        Returns:
            CriticResult: The result of processing the text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            # Validate input
            if not isinstance(input, str) or not input.strip():
                raise ValueError("Input must be a non-empty string")

            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("SelfRefineCritic not properly initialized")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Delegate to critique service
            critique_result = critique_service.critique(input)

            # Create result
            result = CriticResult(
                passed=critique_result.get("score", 0) >= self.config.min_confidence,
                message=critique_result.get("feedback", ""),
                metadata={"operation": "process"},
                score=critique_result.get("score", 0),
                issues=critique_result.get("issues", []),
                suggestions=critique_result.get("suggestions", []),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Update statistics
            self.update_statistics(result)

            return result

        except Exception as e:
            self.record_error(e)
            processing_time = (time.time() - start_time) * 1000
            return CriticResult(
                passed=False,
                message=f"Error: {str(e)}",
                metadata={"error_type": type(e).__name__},
                score=0.0,
                issues=[f"Processing error: {str(e)}"],
                suggestions=["Retry with different input"],
                processing_time_ms=processing_time,
            )

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

        if not self._state_manager.get("initialized", False):
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
        start_time = time.time()

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("SelfRefineCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Track validation count
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            # Get task from metadata
            task = self._get_task_from_metadata(metadata)

            # Create critique prompt
            prompt = (
                self._state_manager.get("cache", {})
                .get("critique_prompt_template", "")
                .format(
                    task=task,
                    response=text,
                )
            )

            # Generate critique
            model = self._state_manager.get("model")
            critique_text = model.generate(
                prompt,
                system_prompt=self._state_manager.get("cache", {}).get("system_prompt", ""),
                temperature=self._state_manager.get("cache", {}).get("temperature", 0.7),
                max_tokens=self._state_manager.get("cache", {}).get("max_tokens", 1000),
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
            is_valid = any(phrase in critique_text.lower() for phrase in no_issues_phrases)

            # Record result in metadata
            if is_valid:
                valid_count = self._state_manager.get_metadata("valid_count", 0)
                self._state_manager.set_metadata("valid_count", valid_count + 1)
            else:
                invalid_count = self._state_manager.get_metadata("invalid_count", 0)
                self._state_manager.set_metadata("invalid_count", invalid_count + 1)

            # Track performance
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_validation_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_validation_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return is_valid

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to validate text: {str(e)}") from e

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
        start_time = time.time()

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("SelfRefineCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Track critique count
            critique_count = self._state_manager.get_metadata("critique_count", 0)
            self._state_manager.set_metadata("critique_count", critique_count + 1)

            # Get task from metadata
            task = self._get_task_from_metadata(metadata)

            # Create critique prompt
            prompt = (
                self._state_manager.get("cache", {})
                .get("critique_prompt_template", "")
                .format(
                    task=task,
                    response=text,
                )
            )

            # Generate critique
            model = self._state_manager.get("model")
            critique_text = model.generate(
                prompt,
                system_prompt=self._state_manager.get("cache", {}).get("system_prompt", ""),
                temperature=self._state_manager.get("cache", {}).get("temperature", 0.7),
                max_tokens=self._state_manager.get("cache", {}).get("max_tokens", 1000),
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

            # Create result
            critique_result = {
                "score": score,
                "feedback": critique_text,
                "issues": issues,
                "suggestions": suggestions,
            }

            # Track score distribution
            score_distribution = self._state_manager.get_metadata("score_distribution", {})
            score_bucket = round(score * 10) / 10  # Round to nearest 0.1
            score_distribution[str(score_bucket)] = score_distribution.get(str(score_bucket), 0) + 1
            self._state_manager.set_metadata("score_distribution", score_distribution)

            # Track performance
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_critique_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_critique_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return critique_result

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to critique text: {str(e)}") from e

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
        start_time = time.time()

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("SelfRefineCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")

            # Track improvement count
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)

            # Get task from metadata
            task = self._get_task_from_metadata(metadata)

            # Get max iterations from config
            max_iterations = self._state_manager.get("cache", {}).get("max_iterations", 3)

            # Start with the original text
            current_output = text

            # Perform iterative refinement
            for iteration in range(max_iterations):
                # Step 1: Critique the current output
                critique_prompt = (
                    self._state_manager.get("cache", {})
                    .get("critique_prompt_template", "")
                    .format(
                        task=task,
                        response=current_output,
                    )
                )

                model = self._state_manager.get("model")
                critique = model.generate(
                    critique_prompt,
                    system_prompt=self._state_manager.get("cache", {}).get("system_prompt", ""),
                    temperature=self._state_manager.get("cache", {}).get("temperature", 0.7),
                    max_tokens=self._state_manager.get("cache", {}).get("max_tokens", 1000),
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
                    # Track iterations
                    self._state_manager.set_metadata("last_improvement_iterations", iteration + 1)
                    break

                # Step 2: Revise using the critique
                revision_prompt = (
                    self._state_manager.get("cache", {})
                    .get("revision_prompt_template", "")
                    .format(
                        task=task,
                        response=current_output,
                        critique=critique,
                    )
                )

                revised_output = model.generate(
                    revision_prompt,
                    system_prompt=self._state_manager.get("cache", {}).get("system_prompt", ""),
                    temperature=self._state_manager.get("cache", {}).get("temperature", 0.7),
                    max_tokens=self._state_manager.get("cache", {}).get("max_tokens", 1000),
                ).strip()

                # Check if there's no improvement
                if revised_output == current_output:
                    # Track iterations
                    self._state_manager.set_metadata("last_improvement_iterations", iteration + 1)
                    break

                # Update current output
                current_output = revised_output

            # Track memory usage
            memory_manager = self._state_manager.get("memory_manager")
            if memory_manager:
                memory_item = json.dumps(
                    {
                        "original_text": text,
                        "task": task,
                        "improved_text": current_output,
                        "timestamp": time.time(),
                    }
                )
                memory_manager.add_to_memory(memory_item)

            # Track performance
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_improvement_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_improvement_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return current_output

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to improve text: {str(e)}") from e

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
        start_time = time.time()

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("SelfRefineCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")

            if not isinstance(feedback, str) or not feedback.strip():
                raise ValueError("feedback must be a non-empty string")

            # Track feedback improvement count
            feedback_count = self._state_manager.get_metadata("feedback_improvement_count", 0)
            self._state_manager.set_metadata("feedback_improvement_count", feedback_count + 1)

            # Create revision prompt with the provided feedback
            revision_prompt = (
                self._state_manager.get("cache", {})
                .get("revision_prompt_template", "")
                .format(
                    task="Improve the following text",
                    response=text,
                    critique=feedback,
                )
            )

            # Generate improved response
            model = self._state_manager.get("model")
            improved_text = model.generate(
                revision_prompt,
                system_prompt=self._state_manager.get("cache", {}).get("system_prompt", ""),
                temperature=self._state_manager.get("cache", {}).get("temperature", 0.7),
                max_tokens=self._state_manager.get("cache", {}).get("max_tokens", 1000),
            ).strip()

            # Track memory usage
            memory_manager = self._state_manager.get("memory_manager")
            if memory_manager:
                memory_item = json.dumps(
                    {
                        "original_text": text,
                        "feedback": feedback,
                        "improved_text": improved_text,
                        "timestamp": time.time(),
                    }
                )
                memory_manager.add_to_memory(memory_item)

            # Track performance
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata(
                    "total_feedback_improvement_time_ms", 0.0
                )
                self._state_manager.set_metadata(
                    "total_feedback_improvement_time_ms",
                    total_time + (time.time() - start_time) * 1000,
                )

            return improved_text

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to improve text with feedback: {str(e)}") from e

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about critic usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "validation_count": self._state_manager.get_metadata("validation_count", 0),
            "valid_count": self._state_manager.get_metadata("valid_count", 0),
            "invalid_count": self._state_manager.get_metadata("invalid_count", 0),
            "critique_count": self._state_manager.get_metadata("critique_count", 0),
            "improvement_count": self._state_manager.get_metadata("improvement_count", 0),
            "feedback_improvement_count": self._state_manager.get_metadata(
                "feedback_improvement_count", 0
            ),
            "last_improvement_iterations": self._state_manager.get_metadata(
                "last_improvement_iterations", 0
            ),
            "score_distribution": self._state_manager.get_metadata("score_distribution", {}),
            "total_validation_time_ms": self._state_manager.get_metadata(
                "total_validation_time_ms", 0
            ),
            "total_critique_time_ms": self._state_manager.get_metadata("total_critique_time_ms", 0),
            "total_improvement_time_ms": self._state_manager.get_metadata(
                "total_improvement_time_ms", 0
            ),
            "total_feedback_improvement_time_ms": self._state_manager.get_metadata(
                "total_feedback_improvement_time_ms", 0
            ),
        }

    # Async methods
    async def avalidate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Asynchronously validate text."""
        # For now, use the synchronous implementation
        return self.validate(text, metadata)

    async def acritique(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Asynchronously critique text."""
        # For now, use the synchronous implementation
        return self.critique(text, metadata)

    async def aimprove(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Asynchronously improve text."""
        # For now, use the synchronous implementation
        return self.improve(text, metadata)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """Asynchronously improve text based on specific feedback."""
        # For now, use the synchronous implementation
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

    This factory function creates and configures a SelfRefineCritic instance with
    the specified parameters and components. It provides a convenient way to create
    a self-refine critic with customized settings for iterative text improvement.

    ## Architecture
    The factory function follows the Factory Method pattern to:
    - Create standardized configuration objects
    - Instantiate critic classes with consistent parameters
    - Support optional parameter overrides
    - Provide type safety through return types
    - Handle error cases gracefully

    ## Lifecycle
    1. **Configuration**: Create and validate configuration
       - Use default configuration as base
       - Apply provided parameter overrides
       - Validate configuration values
       - Handle configuration errors

    2. **Instantiation**: Create and initialize critic
       - Create SelfRefineCritic instance
       - Initialize with resolved dependencies
       - Apply configuration
       - Handle initialization errors

    ## Examples
    ```python
    from sifaka.critics.implementations.self_refine import create_self_refine_critic
    from sifaka.models.providers import OpenAIProvider

    # Create with basic parameters
    provider = OpenAIProvider(api_key="your-api-key")
    critic = create_self_refine_critic(
        llm_provider=provider,
        max_iterations=3,
        temperature=0.7,
        system_prompt="You are an expert editor focused on clarity and accuracy."
    )

    # Create with custom configuration
    from sifaka.utils.config.critics import SelfRefineCriticConfig
    config = SelfRefineCriticConfig(
        name="custom_self_refine_critic",
        description="A custom self-refine critic",
        system_prompt="You are an expert editor focused on clarity and accuracy.",
        temperature=0.5,
        max_tokens=2000,
        max_iterations=5
    )
    critic = create_self_refine_critic(
        llm_provider=provider,
        config=config
    )
    ```

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

    Raises:
        ValueError: If required parameters are missing or invalid
        TypeError: If llm_provider is not a valid provider
    """
    try:
        # Create config if not provided
        if config is None:
            from sifaka.utils.config.critics import DEFAULT_SELF_REFINE_CRITIC_CONFIG

            # Start with default config
            config = DEFAULT_SELF_REFINE_CRITIC_CONFIG.model_copy()

            # Update with provided values
            updates = {
                "name": name,
                "description": description,
            }

            # Add optional parameters if provided
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

            # Add any additional kwargs
            updates.update(kwargs)

            # Create updated config
            config = config.model_copy(update=updates)
        elif isinstance(config, dict):
            from sifaka.utils.config.critics import SelfRefineCriticConfig

            config = SelfRefineCriticConfig(**config)

        # Create and return the critic
        return SelfRefineCritic(
            name=name,
            description=description,
            llm_provider=llm_provider,
            config=config,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Failed to create self-refine critic: {str(e)}")
        raise ValueError(f"Failed to create self-refine critic: {str(e)}") from e
