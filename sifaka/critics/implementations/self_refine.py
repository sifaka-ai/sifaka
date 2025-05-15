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
improved_output = critic.improve(initial_output, {"task": task}) if critic else ""
print(f"Improved output: {improved_output}")

# Get critique for text
critique = critic.critique(initial_output, {"task": task}) if critic else ""
print(f"Score: {critique['score']}")
print(f"Feedback: {critique['feedback']}")

# Validate text
is_valid = critic.validate(initial_output, {"task": task}) if critic else ""
print(f"Is valid: {is_valid}")

# Improve with specific feedback
feedback = "The explanation should include more details about qubits."
improved_output = critic.improve_with_feedback(initial_output, feedback) if critic else ""
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
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, ConfigDict, PrivateAttr

from ...core.base import BaseComponent, BaseConfig
from ...utils.state import create_critic_state
from ...utils.logging import get_logger
from ...core.base import BaseResult as CriticResult
from sifaka.utils.config.critics import SelfRefineCriticConfig
from ...interfaces.critic import TextCritic, TextImprover, TextValidator, CritiqueResult

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
    improved_output = critic.improve(initial_output, {"task": task}) if critic else ""
    ```
    """

    # Class constants
    DEFAULT_NAME = "self_refine_critic"
    DEFAULT_DESCRIPTION = "Improves text through iterative self-critique and revision"

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    config: BaseConfig = Field(description="Critic configuration")

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        config: Optional[Optional[SelfRefineCriticConfig]] = None,
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

            # Convert dict to SelfRefineCriticConfig if it's a dict
            if isinstance(DEFAULT_SELF_REFINE_CRITIC_CONFIG, dict):
                # Extract known parameters
                config_dict = DEFAULT_SELF_REFINE_CRITIC_CONFIG.copy()
                config_dict.update(kwargs)
                config_dict["name"] = name
                config_dict["description"] = description

                # Create config with explicit parameters with proper type handling
                # Use default values if not provided or if conversion fails
                min_confidence: float = 0.7
                try:
                    min_conf_val = config_dict.get("min_confidence")
                    if min_conf_val is not None:
                        if isinstance(min_conf_val, (int, float)):
                            min_confidence = float(min_conf_val)
                        elif isinstance(min_conf_val, str):
                            min_confidence = float(min_conf_val)
                except (ValueError, TypeError):
                    pass

                max_attempts: int = 3
                try:
                    max_att_val = config_dict.get("max_attempts")
                    if max_att_val is not None:
                        if isinstance(max_att_val, (int, float)):
                            max_attempts = int(max_att_val)
                        elif isinstance(max_att_val, str):
                            max_attempts = int(max_att_val)
                except (ValueError, TypeError):
                    pass

                cache_size: int = 100
                try:
                    cache_val = config_dict.get("cache_size")
                    if cache_val is not None:
                        if isinstance(cache_val, (int, float)):
                            cache_size = int(cache_val)
                        elif isinstance(cache_val, str):
                            cache_size = int(cache_val)
                except (ValueError, TypeError):
                    pass

                priority: int = 1
                try:
                    priority_val = config_dict.get("priority")
                    if priority_val is not None:
                        if isinstance(priority_val, (int, float)):
                            priority = int(priority_val)
                        elif isinstance(priority_val, str):
                            priority = int(priority_val)
                except (ValueError, TypeError):
                    pass

                cost: Optional[float] = None
                try:
                    cost_val = config_dict.get("cost")
                    if cost_val is not None:
                        if isinstance(cost_val, (int, float)):
                            cost = float(cost_val)
                        elif isinstance(cost_val, str):
                            cost = float(cost_val)
                except (ValueError, TypeError):
                    pass

                track_performance: bool = True
                try:
                    track_perf_val = config_dict.get("track_performance")
                    if track_perf_val is not None:
                        track_performance = bool(track_perf_val)
                except (ValueError, TypeError):
                    pass

                system_prompt: str = ""
                try:
                    sys_prompt_val = config_dict.get("system_prompt")
                    if sys_prompt_val is not None:
                        system_prompt = str(sys_prompt_val)
                except (ValueError, TypeError):
                    pass

                temperature: float = 0.7
                try:
                    temp_val = config_dict.get("temperature")
                    if temp_val is not None:
                        if isinstance(temp_val, (int, float)):
                            temperature = float(temp_val)
                        elif isinstance(temp_val, str):
                            temperature = float(temp_val)
                except (ValueError, TypeError):
                    pass

                max_tokens: int = 1000
                try:
                    max_tok_val = config_dict.get("max_tokens")
                    if max_tok_val is not None:
                        if isinstance(max_tok_val, (int, float)):
                            max_tokens = int(max_tok_val)
                        elif isinstance(max_tok_val, str):
                            max_tokens = int(max_tok_val)
                except (ValueError, TypeError):
                    pass

                max_iterations: int = 3
                try:
                    max_iter_val = config_dict.get("max_iterations")
                    if max_iter_val is not None:
                        if isinstance(max_iter_val, (int, float)):
                            max_iterations = int(max_iter_val)
                        elif isinstance(max_iter_val, str):
                            max_iterations = int(max_iter_val)
                except (ValueError, TypeError):
                    pass

                refine_prompt: str = ""
                try:
                    refine_val = config_dict.get("refine_prompt")
                    if refine_val is not None:
                        refine_prompt = str(refine_val)
                except (ValueError, TypeError):
                    pass

                memory_buffer_size: int = 10
                try:
                    mem_buf_val = config_dict.get("memory_buffer_size")
                    if mem_buf_val is not None:
                        if isinstance(mem_buf_val, (int, float)):
                            memory_buffer_size = int(mem_buf_val)
                        elif isinstance(mem_buf_val, str):
                            memory_buffer_size = int(mem_buf_val)
                except (ValueError, TypeError):
                    pass

                eager_initialization: bool = False
                try:
                    eager_val = config_dict.get("eager_initialization")
                    if eager_val is not None:
                        eager_initialization = bool(eager_val)
                except (ValueError, TypeError):
                    pass

                track_errors: bool = True
                try:
                    track_err_val = config_dict.get("track_errors")
                    if track_err_val is not None:
                        track_errors = bool(track_err_val)
                except (ValueError, TypeError):
                    pass

                params: Dict[str, Any] = {}
                params_val = config_dict.get("params")
                if isinstance(params_val, dict):
                    params = params_val

                # Create the config with proper typing
                from sifaka.utils.config.critics import standardize_critic_config

                config = standardize_critic_config(
                    config_class=SelfRefineCriticConfig,
                    name=name,
                    description=description,
                    min_confidence=min_confidence,
                    max_attempts=max_attempts,
                    cache_size=cache_size,
                    priority=priority,
                    cost=cost,
                    track_performance=track_performance,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_iterations=max_iterations,
                    refine_prompt=refine_prompt,
                    memory_buffer_size=memory_buffer_size,
                    eager_initialization=eager_initialization,
                    track_errors=track_errors,
                    params=params,
                )
            else:
                config = DEFAULT_SELF_REFINE_CRITIC_CONFIG.model_copy(
                    update={"name": name, "description": description, **kwargs}
                )

        # Initialize base component with config cast to BaseConfig
        base_config = BaseConfig(name=name, description=description)
        super().__init__(name=name, description=description, config=base_config)

        # Store the SelfRefineCriticConfig separately
        self._critic_config = config

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
            cache["max_iterations"] = (
                self._critic_config.max_iterations if hasattr(self, "_critic_config") else 3
            )
            # Set critique prompt template with safe access
            default_critique_template = (
                "Please critique the following response to the task. "
                "Focus on accuracy, clarity, and completeness.\n\n"
                "Task:\n{task}\n\n"
                "Response:\n{response}\n\n"
                "Critique:"
            )

            if hasattr(self, "_critic_config") and hasattr(
                self._critic_config, "critique_prompt_template"
            ):
                cache["critique_prompt_template"] = self._critic_config.critique_prompt_template
            else:
                cache["critique_prompt_template"] = default_critique_template
            # Set revision prompt template with safe access
            default_revision_template = (
                "Please revise the following response based on the critique.\n\n"
                "Task:\n{task}\n\n"
                "Response:\n{response}\n\n"
                "Critique:\n{critique}\n\n"
                "Revised response:"
            )

            if hasattr(self, "_critic_config") and hasattr(
                self._critic_config, "revision_prompt_template"
            ):
                cache["revision_prompt_template"] = self._critic_config.revision_prompt_template
            else:
                cache["revision_prompt_template"] = default_revision_template
            if hasattr(self, "_critic_config"):
                cache["system_prompt"] = self._critic_config.system_prompt
                cache["temperature"] = self._critic_config.temperature
                cache["max_tokens"] = self._critic_config.max_tokens
            else:
                cache["system_prompt"] = ""
                cache["temperature"] = 0.7
                cache["max_tokens"] = 1000
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

            # Get min_confidence from critic_config or use default
            min_confidence = 0.7
            if hasattr(self, "_critic_config") and hasattr(self._critic_config, "min_confidence"):
                min_confidence = self._critic_config.min_confidence

            # Create result
            result: CriticResult = CriticResult(
                passed=critique_result.get("score", 0) >= min_confidence,
                message=critique_result.get("feedback", ""),
                metadata={"operation": "process"},
                score=critique_result.get("score", 0),
                issues=critique_result.get("issues", []),
                suggestions=critique_result.get("suggestions", []),
                processing_time_ms=float(time.time() - start_time) * 1000.0,
            )

            # Update statistics
            self.update_statistics(result)

            return result

        except Exception as e:
            self.record_error(e)
            elapsed_time = time.time() - start_time
            processing_time = float(elapsed_time) * 1000.0
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

    def validate(self, text: str) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

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

            # Use a default task since metadata is not available in this interface
            task: str = "Validate the following text"

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
            if hasattr(self, "config") and self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_validation_time_ms", 0.0)
                elapsed_time = time.time() - start_time
                self._state_manager.set_metadata(
                    "total_validation_time_ms", total_time + (float(elapsed_time) * 1000)
                )

            return bool(is_valid)

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to validate text: {str(e)}") from e

    def critique(self, text: str) -> CritiqueResult:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CritiqueResult: A dictionary containing critique information

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
            if not isinstance(text, str) or not text.strip() if text else "":
                raise ValueError("text must be a non-empty string")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service") if cache else ""
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Track critique count
            critique_count = self._state_manager.get_metadata("critique_count", 0)
            self._state_manager.set_metadata("critique_count", critique_count + 1)

            # Use empty task since metadata is not available in this interface
            task = "Analyze the following text"

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
                system_prompt=(
                    self._state_manager.get("cache", {}).get("system_prompt", "") if model else ""
                ),
                temperature=self._state_manager.get("cache", {}).get("temperature", 0.7),
                max_tokens=self._state_manager.get("cache", {}).get("max_tokens", 1000),
            ).strip()

            # Parse critique
            issues: List[str] = []
            suggestions: List[str] = []

            # Extract issues and suggestions from critique
            for line in critique_text.split("\n") if critique_text else "":
                line = line.strip() if line else ""
                if line.startswith("- ") if line else "" or line.startswith("* ") if line else "":
                    if (
                        "should" in line.lower()
                        if line
                        else (
                            "" or "could" in line.lower()
                            if line
                            else "" or "recommend" in line.lower() if line else ""
                        )
                    ):
                        suggestions.append(line[2:]) if suggestions else ""
                    else:
                        issues.append(line[2:]) if issues else ""

            # Calculate score based on issues
            score = 1.0 if not issues else max(0.0, 1.0 - (len(issues) * 0.1))

            # Create result
            critique_result: CritiqueResult = {
                "score": float(score),
                "feedback": str(critique_text),
                "issues": issues,
                "suggestions": suggestions,
            }

            # Track score distribution
            score_distribution = self._state_manager.get_metadata("score_distribution", {})
            score_bucket = round(score * 10) / 10  # Round to nearest 0.1
            score_distribution[str(score_bucket)] = score_distribution.get(str(score_bucket), 0) + 1
            self._state_manager.set_metadata("score_distribution", score_distribution)

            # Track performance
            if hasattr(self, "config") and self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_critique_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_critique_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return critique_result

        except Exception as e:
            self.record_error(e) if self else ""
            raise RuntimeError(f"Failed to critique text: {str(e)}") from e

    def improve(self, text: str, feedback: str) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time() if time else ""

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("SelfRefineCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip() if text else "":
                raise ValueError("text must be a non-empty string")

            # Track improvement count
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)

            # Use feedback as task
            task = f"Improve the text based on this feedback: {feedback}"

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
                    system_prompt=(
                        self._state_manager.get("cache", {}).get("system_prompt", "")
                        if model
                        else ""
                    ),
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
                if any(
                    phrase in critique.lower() if critique else "" for phrase in no_issues_phrases
                ):
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
                    system_prompt=(
                        self._state_manager.get("cache", {}).get("system_prompt", "")
                        if model
                        else ""
                    ),
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
            if hasattr(self, "config") and self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_improvement_time_ms", 0.0)
                elapsed_time = float(time.time() - start_time)
                self._state_manager.set_metadata(
                    "total_improvement_time_ms",
                    float(total_time) + (elapsed_time * 1000),
                )

            return str(current_output)

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
            if hasattr(self, "config") and self.config.track_performance:
                total_time = self._state_manager.get_metadata(
                    "total_feedback_improvement_time_ms", 0.0
                )

                elapsed_time = time.time() - start_time
                self._state_manager.set_metadata(
                    "total_feedback_improvement_time_ms",
                    total_time + (float(elapsed_time) * 1000),
                )

            return str(improved_text)

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


def create_self_refine_critic(
    llm_provider: Any,
    name: str = "self_refine_critic",
    description: str = "Improves text through iterative self-critique and revision",
    min_confidence: Optional[float] = None,
    max_attempts: Optional[int] = None,
    cache_size: Optional[int] = None,
    priority: Optional[int] = None,
    cost: Optional[float] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_iterations: Optional[int] = None,
    critique_prompt_template: Optional[Optional[str]] = None,
    revision_prompt_template: Optional[Optional[str]] = None,
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
            if isinstance(DEFAULT_SELF_REFINE_CRITIC_CONFIG, dict):
                from sifaka.utils.config.critics import standardize_critic_config

                config = standardize_critic_config(
                    config_class=SelfRefineCriticConfig, config=DEFAULT_SELF_REFINE_CRITIC_CONFIG
                )
            else:
                config = DEFAULT_SELF_REFINE_CRITIC_CONFIG.model_copy()

            # Update with provided values
            updates = {
                "name": name,
                "description": description,
            }

            # Add optional parameters if provided
            if system_prompt is not None:
                updates["system_prompt"] = str(system_prompt)
            if temperature is not None:
                if isinstance(temperature, (int, float)):
                    updates["temperature"] = float(temperature)
                elif isinstance(temperature, str):
                    try:
                        updates["temperature"] = float(temperature)
                    except (ValueError, TypeError):
                        pass
            if max_tokens is not None:
                if isinstance(max_tokens, (int, float)):
                    updates["max_tokens"] = int(max_tokens)
                elif isinstance(max_tokens, str):
                    try:
                        updates["max_tokens"] = int(max_tokens)
                    except (ValueError, TypeError):
                        pass
            if min_confidence is not None:
                if isinstance(min_confidence, (int, float)):
                    updates["min_confidence"] = float(min_confidence)
                elif isinstance(min_confidence, str):
                    try:
                        updates["min_confidence"] = float(min_confidence)
                    except (ValueError, TypeError):
                        pass
            if max_attempts is not None:
                if isinstance(max_attempts, (int, float)):
                    updates["max_attempts"] = int(max_attempts)
                elif isinstance(max_attempts, str):
                    try:
                        updates["max_attempts"] = int(max_attempts)
                    except (ValueError, TypeError):
                        pass
            if cache_size is not None:
                if isinstance(cache_size, (int, float)):
                    updates["cache_size"] = int(cache_size)
                elif isinstance(cache_size, str):
                    try:
                        updates["cache_size"] = int(cache_size)
                    except (ValueError, TypeError):
                        pass
            if priority is not None:
                if isinstance(priority, (int, float)):
                    updates["priority"] = int(priority)
                elif isinstance(priority, str):
                    try:
                        updates["priority"] = int(priority)
                    except (ValueError, TypeError):
                        pass
            if cost is not None:
                if isinstance(cost, (int, float)):
                    updates["cost"] = float(cost)
                elif isinstance(cost, str):
                    try:
                        updates["cost"] = float(cost)
                    except (ValueError, TypeError):
                        pass
            if max_iterations is not None:
                if isinstance(max_iterations, (int, float)):
                    updates["max_iterations"] = int(max_iterations)
                elif isinstance(max_iterations, str):
                    try:
                        updates["max_iterations"] = int(max_iterations)
                    except (ValueError, TypeError):
                        pass
            if critique_prompt_template is not None:
                updates["critique_prompt_template"] = critique_prompt_template
            if revision_prompt_template is not None:
                updates["revision_prompt_template"] = revision_prompt_template

            # Add any additional kwargs
            updates.update(kwargs)

            # Create updated config
            if hasattr(config, "model_copy"):
                config = config.model_copy(update=updates)
            elif isinstance(config, dict):
                config_dict = dict(config)  # Create a new dict instead of using copy()
                config_dict.update(updates)
                from sifaka.utils.config.critics import standardize_critic_config

                config = standardize_critic_config(
                    config_class=SelfRefineCriticConfig, config=config_dict
                )
            else:
                # Create a new config with updates
                from sifaka.utils.config.critics import standardize_critic_config

                config = standardize_critic_config(config_class=SelfRefineCriticConfig, **updates)
        elif isinstance(config, dict):
            from sifaka.utils.config.critics import SelfRefineCriticConfig

            from sifaka.utils.config.critics import standardize_critic_config

            config = standardize_critic_config(config_class=SelfRefineCriticConfig, config=config)

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
