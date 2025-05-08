"""
Implementation of a self-refine critic using composition over inheritance.

This module provides a critic implementation that uses language models to iteratively
critique and revise their own outputs. It follows the composition over inheritance pattern.

## Component Lifecycle

### Self-Refine Critic Implementation Lifecycle

1. **Initialization Phase**
   - Configuration validation
   - Provider setup
   - Factory initialization
   - Resource allocation

2. **Operation Phase**
   - Text validation
   - Critique generation
   - Iterative text improvement
   - Feedback processing

3. **Cleanup Phase**
   - Resource cleanup
   - State reset
   - Error recovery

## Examples

```python
from sifaka.critics.implementations.self_refine_implementation import SelfRefineCriticImplementation
from sifaka.critics.base import CompositionCritic, create_composition_critic
from sifaka.critics.models import SelfRefineCriticConfig
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a self-refine critic configuration
config = SelfRefineCriticConfig(
    name="my_self_refine_critic",
    description="A critic that iteratively improves text",
    system_prompt="You are an expert at critiquing and revising content.",
    max_iterations=3
)

# Create a self-refine critic implementation
implementation = SelfRefineCriticImplementation(config, provider)

# Create a critic with the implementation
critic = create_composition_critic(
    name="my_self_refine_critic",
    description="A critic that iteratively improves text",
    implementation=implementation
)

# Use the critic
text = "This is a sample technical document."
is_valid = critic.validate(text)
improved = critic.improve(text, {"task": "Improve this technical document"})
feedback = critic.critique(text)
```
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import PrivateAttr

from ..models import SelfRefineCriticConfig
from ...utils.state import CriticState, create_critic_state

# Configure logging
logger = logging.getLogger(__name__)


class SelfRefineCriticImplementation:
    """
    Implementation of a self-refine critic using language models.

    This class implements the CriticImplementation protocol for a self-refine critic
    that uses language models to iteratively critique and revise text outputs.

    ## Lifecycle Management

    The SelfRefineCriticImplementation manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up language model provider
       - Initializes prompt factory
       - Allocates resources

    2. **Operation**
       - Validates text input
       - Generates critiques
       - Iteratively improves text quality
       - Processes feedback

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status

    ## Examples

    ```python
    from sifaka.critics.implementations.self_refine_implementation import SelfRefineCriticImplementation
    from sifaka.critics.models import SelfRefineCriticConfig
    from sifaka.models.providers import OpenAIProvider

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Create a configuration
    config = SelfRefineCriticConfig(
        name="my_self_refine_critic",
        description="A critic that iteratively improves text",
        system_prompt="You are an expert at critiquing and revising content.",
        max_iterations=3
    )

    # Create an implementation
    implementation = SelfRefineCriticImplementation(config, provider)
    ```
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        config: SelfRefineCriticConfig,
        llm_provider: Any,
        prompt_factory: Optional[Any] = None,
    ) -> None:
        """
        Initialize the self-refine critic implementation.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use
            prompt_factory: Optional custom prompt factory

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If provider setup fails
        """
        self.config = config

        # Initialize state
        state = self._state_manager.get_state()

        # Create components
        from ..managers.prompt_factories import PromptCriticPromptManager
        from ..managers.response import ResponseParser
        from ..managers.memory import MemoryManager
        from ..services.critique import CritiqueService

        # Store components in state
        state.model = llm_provider
        state.prompt_manager = prompt_factory or PromptCriticPromptManager(config)
        state.response_parser = ResponseParser()
        state.memory_manager = MemoryManager(buffer_size=10)

        # Create critique service
        critique_service = CritiqueService(
            model=llm_provider,
            prompt_manager=state.prompt_manager,
            response_parser=state.response_parser,
        )

        # Store critique service in state cache
        state.cache["critique_service"] = critique_service

        # Store configuration values in state cache
        state.cache["system_prompt"] = config.system_prompt
        state.cache["temperature"] = config.temperature
        state.cache["max_tokens"] = config.max_tokens
        state.cache["max_iterations"] = config.max_iterations
        state.cache["critique_prompt_template"] = config.critique_prompt_template
        state.cache["revision_prompt_template"] = config.revision_prompt_template

        # Mark as initialized
        state.initialized = True

        logger.info(f"Initialized SelfRefineCriticImplementation with config: {config.name}")

    def _check_input(self, text: str) -> None:
        """
        Check if input text is valid.

        Args:
            text: Text to check

        Raises:
            ValueError: If text is empty or invalid
        """
        # Get state
        state = self._state_manager.get_state()

        if not state.initialized:
            raise RuntimeError("SelfRefineCriticImplementation not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

    def _format_feedback(self, feedback: Optional[Any]) -> str:
        """
        Format feedback to string.

        Args:
            feedback: Feedback to format

        Returns:
            str: Formatted feedback
        """
        if feedback is None:
            return "Please improve this text for clarity and effectiveness."

        if isinstance(feedback, dict):
            return json.dumps(feedback)

        return str(feedback)

    def validate_impl(self, text: str) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: Text to validate

        Returns:
            bool: True if text is valid, False otherwise

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If validation fails
        """
        self._check_input(text)

        # Get state
        state = self._state_manager.get_state()

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.validate(text)

    def improve_impl(self, text: str, feedback: Optional[Any] = None) -> str:
        """
        Improve text through iterative self-critique and revision.

        Args:
            text: Text to improve
            feedback: Optional feedback to guide improvement, can be a dict with 'task' key

        Returns:
            str: Improved text

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If improvement fails
        """
        self._check_input(text)

        # Get state
        state = self._state_manager.get_state()

        # Extract task from feedback if it's a dictionary
        task = None
        if isinstance(feedback, dict) and "task" in feedback:
            task = feedback["task"]
        else:
            task = self._format_feedback(feedback)

        # Initialize current output
        current_output = text

        # Get max iterations from config
        max_iterations = state.cache.get("max_iterations", 3)

        # Perform iterative refinement
        for _ in range(max_iterations):
            # Step 1: Critique the current output
            critique_prompt = state.cache.get("critique_prompt_template", "").format(
                task=task,
                response=current_output,
            )

            critique = state.model.generate(
                critique_prompt,
                system_prompt=state.cache.get("system_prompt", ""),
                temperature=state.cache.get("temperature", 0.7),
                max_tokens=state.cache.get("max_tokens", 1000),
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
            revision_prompt = state.cache.get("revision_prompt_template", "").format(
                task=task,
                response=current_output,
                critique=critique,
            )

            revised_output = state.model.generate(
                revision_prompt,
                system_prompt=state.cache.get("system_prompt", ""),
                temperature=state.cache.get("temperature", 0.7),
                max_tokens=state.cache.get("max_tokens", 1000),
            ).strip()

            # Check if there's no improvement
            if revised_output == current_output:
                return current_output

            # Update current output
            current_output = revised_output

        return current_output

    def critique_impl(self, text: str) -> Dict[str, Any]:
        """
        Critique text and provide feedback.

        Args:
            text: Text to critique

        Returns:
            Dict[str, Any]: A dictionary containing critique information

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If critique fails
        """
        self._check_input(text)

        # Get state
        state = self._state_manager.get_state()

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        critique = critique_service.critique(text)

        # Convert to dictionary format
        return {
            "score": critique.score,
            "feedback": critique.feedback,
            "issues": critique.issues,
            "suggestions": critique.suggestions,
        }

    def warm_up_impl(self) -> None:
        """
        Warm up the critic implementation.

        This method initializes any resources needed by the critic implementation.
        """
        # Get state to ensure it's initialized
        state = self._state_manager.get_state()

        # Check if already initialized
        if not state.initialized:
            # Create components
            from ..managers.prompt_factories import PromptCriticPromptManager
            from ..managers.response import ResponseParser
            from ..managers.memory import MemoryManager
            from ..services.critique import CritiqueService

            # Initialize components if not already done
            if not hasattr(state, "prompt_manager") or state.prompt_manager is None:
                state.prompt_manager = PromptCriticPromptManager(self.config)

            if not hasattr(state, "response_parser") or state.response_parser is None:
                state.response_parser = ResponseParser()

            if not hasattr(state, "memory_manager") or state.memory_manager is None:
                state.memory_manager = MemoryManager(buffer_size=10)

            # Initialize critique service if not already done
            if "critique_service" not in state.cache:
                critique_service = CritiqueService(
                    model=state.model,
                    prompt_manager=state.prompt_manager,
                    response_parser=state.response_parser,
                )
                state.cache["critique_service"] = critique_service

            # Store configuration values in state cache if not already done
            if "system_prompt" not in state.cache:
                state.cache["system_prompt"] = self.config.system_prompt
                state.cache["temperature"] = self.config.temperature
                state.cache["max_tokens"] = self.config.max_tokens
                state.cache["max_iterations"] = self.config.max_iterations
                state.cache["critique_prompt_template"] = self.config.critique_prompt_template
                state.cache["revision_prompt_template"] = self.config.revision_prompt_template

            # Mark as initialized
            state.initialized = True
