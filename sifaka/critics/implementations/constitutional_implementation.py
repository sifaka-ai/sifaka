"""
Implementation of a constitutional critic using composition over inheritance.

This module provides a critic implementation that evaluates responses against a set of
human-written principles (a "constitution") and provides natural language feedback when
violations are detected. It follows the composition over inheritance pattern.

Based on Constitutional AI: https://arxiv.org/abs/2212.08073

## Component Lifecycle

### Constitutional Critic Implementation Lifecycle

1. **Initialization Phase**
   - Configuration validation
   - Provider setup
   - Principles initialization
   - Resource allocation

2. **Operation Phase**
   - Text validation against principles
   - Critique generation
   - Text improvement based on critiques
   - Feedback processing

3. **Cleanup Phase**
   - Resource cleanup
   - State reset
   - Error recovery

## Examples

```python
from sifaka.critics.implementations.constitutional_implementation import ConstitutionalCriticImplementation
from sifaka.critics.base import CompositionCritic, create_composition_critic
from sifaka.critics.models import ConstitutionalCriticConfig
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Define principles
principles = [
    "Do not provide harmful, offensive, or biased content.",
    "Explain reasoning in a clear and truthful manner.",
    "Respect user autonomy and avoid manipulative language.",
]

# Create a constitutional critic configuration
config = ConstitutionalCriticConfig(
    name="constitutional_critic",
    description="A critic that evaluates content against principles",
    principles=principles,
    system_prompt="You are an expert at evaluating content against principles.",
    temperature=0.7,
    max_tokens=1000
)

# Create a constitutional critic implementation
implementation = ConstitutionalCriticImplementation(config, provider)

# Create a critic with the implementation
critic = create_composition_critic(
    name="constitutional_critic",
    description="A critic that evaluates content against principles",
    implementation=implementation
)

# Use the critic
text = "Climate change is just a hoax created by scientists who want more funding."
metadata = {"task": "Explain why some people believe climate change isn't real."}
is_valid = critic.validate(text, metadata=metadata)
critique = critic.critique(text, metadata=metadata)
improved_text = critic.improve(text, metadata=metadata)
```
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import PrivateAttr

from ..models import ConstitutionalCriticConfig, CriticConfig
from ..protocols import CriticImplementation
from ...utils.state import CriticState, create_critic_state

# Configure logging
logger = logging.getLogger(__name__)

# Default prompt templates
DEFAULT_CRITIQUE_PROMPT_TEMPLATE = """
You are evaluating a response against these principles:

{principles}

Task: {task}

Response to evaluate:
{response}

Provide a detailed critique of this response, identifying any violations of the principles.
Be specific about issues and suggest improvements.
"""

DEFAULT_IMPROVEMENT_PROMPT_TEMPLATE = """
You are improving a response based on these principles:

{principles}

Task: {task}

Original response:
{response}

Critique:
{critique}

Provide an improved version that addresses all the issues identified in the critique
while maintaining accuracy and adhering to all principles.
"""


class ConstitutionalCriticImplementation:
    """
    Implementation of a constitutional critic using language models.

    This class implements the CriticImplementation protocol for a constitutional critic
    that evaluates responses against a set of principles.

    ## Lifecycle Management

    The ConstitutionalCriticImplementation manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up language model provider
       - Initializes principles
       - Allocates resources

    2. **Operation**
       - Validates text against principles
       - Generates critiques
       - Improves text based on critiques
       - Processes feedback

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status

    ## Error Handling

    The implementation handles various error conditions:
    - Empty or invalid input text
    - Missing task information
    - Model generation failures
    - Response parsing errors
    - State initialization issues
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        config: Union[CriticConfig, ConstitutionalCriticConfig],
        llm_provider: Any,
    ) -> None:
        """
        Initialize the constitutional critic implementation.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider

        Raises:
            ValueError: If configuration is invalid
            TypeError: If config is not a ConstitutionalCriticConfig
        """
        # Validate config type
        if not isinstance(config, ConstitutionalCriticConfig):
            raise TypeError("config must be a ConstitutionalCriticConfig")

        # Validate principles
        if not config.principles or not all(isinstance(p, str) for p in config.principles):
            raise ValueError("principles must be a non-empty list of strings")

        self.config = config

        # Initialize state
        state = self._state_manager.get_state()

        # Store components in state
        state.model = llm_provider
        state.cache = {
            "principles": config.principles,
            "critique_prompt_template": config.params.get(
                "critique_prompt_template", DEFAULT_CRITIQUE_PROMPT_TEMPLATE
            ),
            "improvement_prompt_template": config.params.get(
                "improvement_prompt_template", DEFAULT_IMPROVEMENT_PROMPT_TEMPLATE
            ),
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

        # Mark as initialized
        state.initialized = True

    def _format_principles(self) -> str:
        """
        Format principles as a bulleted list.

        Returns:
            Formatted principles as a string
        """
        # Get state
        state = self._state_manager.get_state()

        principles = state.cache.get("principles", [])
        return "\n".join(f"- {p}" for p in principles)

    def _get_task_from_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Extract task from metadata.

        Args:
            metadata: Optional metadata dictionary

        Returns:
            Task string

        Raises:
            ValueError: If task is not provided in metadata
        """
        if not metadata or "task" not in metadata:
            raise ValueError("metadata must contain 'task' key")
        return metadata["task"]

    def _check_input(self, text: str) -> None:
        """
        Check if input text is valid.

        Args:
            text: Text to check

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If implementation is not initialized
        """
        # Get state
        state = self._state_manager.get_state()

        if not state.initialized:
            raise RuntimeError("ConstitutionalCriticImplementation not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

    def validate_impl(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate text against principles.

        Args:
            text: Text to validate
            metadata: Optional metadata containing the task

        Returns:
            True if text is valid, False otherwise

        Raises:
            ValueError: If text is empty or metadata is missing required keys
            RuntimeError: If validation fails
        """
        self._check_input(text)

        # Get critique
        critique_result = self.critique_impl(text, metadata)

        # Check if critique indicates violations
        return not critique_result.get("issues", [])

    def critique_impl(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Critique text against principles.

        Args:
            text: Text to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary with critique information

        Raises:
            ValueError: If text is empty or metadata is missing required keys
            RuntimeError: If critique fails
        """
        self._check_input(text)

        # Get state
        state = self._state_manager.get_state()

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Format principles
        principles_text = self._format_principles()

        # Create critique prompt
        prompt = state.cache.get("critique_prompt_template", "").format(
            principles=principles_text,
            task=task,
            response=text,
        )

        # Generate critique
        critique_text = state.model.generate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
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

            # If no structured issues were found, add the whole critique as an issue
            if not issues:
                issues = [critique_text]

        return {"score": score, "feedback": feedback, "issues": issues, "suggestions": suggestions}

    def improve_impl(self, text: str, feedback: Optional[Any] = None) -> str:
        """
        Improve text based on principles and feedback.

        Args:
            text: Text to improve
            feedback: Optional feedback to guide improvement

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If improvement fails
        """
        self._check_input(text)

        # Get state
        state = self._state_manager.get_state()

        # Handle metadata if provided in feedback
        metadata = None
        if isinstance(feedback, dict) and "metadata" in feedback:
            metadata = feedback["metadata"]

        # Get task from metadata
        try:
            task = self._get_task_from_metadata(metadata)
        except ValueError:
            # If no task in metadata, use a generic task
            task = "Improve the following text"

        # Get critique if not provided
        critique_result = None
        if isinstance(feedback, dict) and "critique" in feedback:
            critique_result = feedback["critique"]
        else:
            critique_result = self.critique_impl(text, metadata)

        # If no issues, return original text
        if not critique_result.get("issues", []):
            return text

        # Format principles
        principles_text = self._format_principles()

        # Create improvement prompt
        prompt = state.cache.get("improvement_prompt_template", "").format(
            principles=principles_text,
            task=task,
            response=text,
            critique=critique_result["feedback"],
        )

        # Generate improved response
        improved_text = state.model.generate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
        ).strip()

        return improved_text

    def warm_up_impl(self) -> None:
        """
        Warm up the critic implementation.

        This method initializes any resources needed by the critic implementation.
        """
        # Get state to ensure it's initialized
        state = self._state_manager.get_state()

        # Check if already initialized
        if not state.initialized:
            # Validate config type
            if not isinstance(self.config, ConstitutionalCriticConfig):
                raise TypeError("config must be a ConstitutionalCriticConfig")

            # Initialize state
            state.cache = {
                "principles": self.config.principles,
                "critique_prompt_template": self.config.params.get(
                    "critique_prompt_template", DEFAULT_CRITIQUE_PROMPT_TEMPLATE
                ),
                "improvement_prompt_template": self.config.params.get(
                    "improvement_prompt_template", DEFAULT_IMPROVEMENT_PROMPT_TEMPLATE
                ),
                "system_prompt": self.config.system_prompt,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }

            # Mark as initialized
            state.initialized = True
