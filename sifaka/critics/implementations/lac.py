"""
LAC (LLM-Based Actor-Critic) critic module for Sifaka.

This module implements the LAC approach for critics, which combines language feedback
and value scoring to improve language model-based decision making.

Based on: Language Feedback Improves Language Model-based Decision Making
https://arxiv.org/abs/2403.03692

Example:
    ```python
    from sifaka.critics.implementations.lac import create_lac_critic
    from sifaka.models.providers import OpenAIProvider

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Create a LAC critic
    critic = create_lac_critic(llm_provider=provider)

    # Use the critic to improve text
    task = "Summarize the causes of World War I in 3 bullet points."
    response = provider.generate(f"Task:\n{task}")
    results = critic.critique(response, {"task": task})

    print("Feedback:", results["feedback"])
    print("Value Score:", results["value"])
    ```
"""

from typing import Any, Dict, Optional, Union

from pydantic import Field, PrivateAttr, ConfigDict

from ..base import BaseCritic
from ..config import FeedbackCriticConfig, ValueCriticConfig, LACCriticConfig
from ..interfaces.critic import TextCritic, TextImprover, TextValidator

# Default prompt templates
DEFAULT_FEEDBACK_PROMPT_TEMPLATE = (
    "Task:\n{task}\n\nResponse:\n{response}\n\n"
    "Provide natural language feedback: what could be improved or was done well?"
)

DEFAULT_VALUE_PROMPT_TEMPLATE = (
    "Task:\n{task}\n\nResponse:\n{response}\n\n"
    "On a scale of 0 to 1, how likely is this response to be correct and complete? "
    "Respond with a float only."
)

DEFAULT_SYSTEM_PROMPT = "You are an expert at evaluating and improving text."


class FeedbackCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """
    A critic that produces natural language feedback for a model's response to a task.

    This critic analyzes text and provides detailed feedback on what could be
    improved or what was done well.
    """

    # Class constants
    DEFAULT_NAME = "feedback_critic"
    DEFAULT_DESCRIPTION = "Provides natural language feedback for text"

    # State management using direct state
    _state = PrivateAttr(default_factory=lambda: None)

    def __init__(
        self,
        config: FeedbackCriticConfig,
        llm_provider: Any,
    ) -> None:
        """
        Initialize the feedback critic.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use

        Raises:
            ValueError: If llm_provider is None
            TypeError: If llm_provider is not a valid provider
        """
        # Validate required parameters
        if llm_provider is None:
            raise ValueError("llm_provider cannot be None")

        # Initialize base class
        super().__init__(config)

        # Initialize state
        from ...utils.state import CriticState

        self._state = CriticState()

        # Store components in state
        self._state.model = llm_provider
        self._state.cache = {
            "feedback_prompt_template": config.feedback_prompt_template,
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
            raise RuntimeError("FeedbackCritic not properly initialized")

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

    def run(self, task: str, response: str) -> str:
        """
        Generate natural language feedback for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to provide feedback on

        Returns:
            Natural language feedback

        Raises:
            ValueError: If response is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(response)

        # Create feedback prompt
        prompt = self._state.cache.get("feedback_prompt_template", "").format(
            task=task,
            response=response,
        )

        # Generate feedback
        feedback = self._state.model.generate(
            prompt,
            system_prompt=self._state.cache.get("system_prompt", ""),
            temperature=self._state.cache.get("temperature", 0.7),
            max_tokens=self._state.cache.get("max_tokens", 1000),
        ).strip()

        return feedback

    async def arun(self, task: str, response: str) -> str:
        """
        Asynchronously generate natural language feedback for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to provide feedback on

        Returns:
            Natural language feedback

        Raises:
            ValueError: If response is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(response)

        # Create feedback prompt
        prompt = self._state.cache.get("feedback_prompt_template", "").format(
            task=task,
            response=response,
        )

        # Generate feedback
        feedback = await self._state.model.agenerate(
            prompt,
            system_prompt=self._state.cache.get("system_prompt", ""),
            temperature=self._state.cache.get("temperature", 0.7),
            max_tokens=self._state.cache.get("max_tokens", 1000),
        )
        feedback = feedback.strip()

        return feedback

    def validate(self, text: str) -> bool:
        """
        Check if text meets quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards

        Raises:
            ValueError: If text is empty
        """
        self._check_input(text)
        # Feedback critics always return True for validation
        # as they focus on providing feedback rather than validation
        return True

    def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Generate feedback
        feedback = self.run(task, text)

        # Create improvement prompt
        prompt = (
            f"Task:\n{task}\n\n"
            f"Original response:\n{text}\n\n"
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
        self._check_input(text)
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("feedback must be a non-empty string")

        # Create improvement prompt
        prompt = f"Original text:\n{text}\n\n" f"Feedback:\n{feedback}\n\n" f"Improved text:"

        # Generate improved response
        improved_text = self._state.model.generate(
            prompt,
            system_prompt=self._state.cache.get("system_prompt", ""),
            temperature=self._state.cache.get("temperature", 0.7),
            max_tokens=self._state.cache.get("max_tokens", 1000),
        ).strip()

        return improved_text

    def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze text and provide detailed feedback.

        Args:
            text: The text to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing feedback

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Generate feedback
        feedback = self.run(task, text)

        # Create critique result
        return {
            "score": 0.5,  # Default score since feedback critics don't provide scores
            "feedback": feedback,
            "issues": [],
            "suggestions": [],
        }


class ValueCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """
    A critic that estimates a numeric value for a model's response to a task.

    This critic analyzes text and provides a numeric score (e.g., probability of success)
    for the response.
    """

    # Class constants
    DEFAULT_NAME = "value_critic"
    DEFAULT_DESCRIPTION = "Provides numeric value scoring for text"

    # State management using direct state
    _state = PrivateAttr(default_factory=lambda: None)

    def __init__(
        self,
        config: ValueCriticConfig,
        llm_provider: Any,
    ) -> None:
        """
        Initialize the value critic.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use

        Raises:
            ValueError: If llm_provider is None
            TypeError: If llm_provider is not a valid provider
        """
        # Validate required parameters
        if llm_provider is None:
            raise ValueError("llm_provider cannot be None")

        # Initialize base class
        super().__init__(config)

        # Initialize state
        from ...utils.state import CriticState

        self._state = CriticState()

        # Store components in state
        self._state.model = llm_provider
        self._state.cache = {
            "value_prompt_template": config.value_prompt_template,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "min_score": config.min_score,
            "max_score": config.max_score,
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
            raise RuntimeError("ValueCritic not properly initialized")

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

    def run(self, task: str, response: str) -> float:
        """
        Generate a numeric value for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to provide a value for

        Returns:
            Numeric value between min_score and max_score

        Raises:
            ValueError: If response is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(response)

        # Create value prompt
        prompt = self._state.cache.get("value_prompt_template", "").format(
            task=task,
            response=response,
        )

        # Generate value
        value_text = self._state.model.generate(
            prompt,
            system_prompt=self._state.cache.get("system_prompt", ""),
            temperature=self._state.cache.get("temperature", 0.3),
            max_tokens=self._state.cache.get("max_tokens", 100),
        ).strip()

        # Parse value
        try:
            value = float(value_text)
            # Clamp value to range
            min_score = self._state.cache.get("min_score", 0.0)
            max_score = self._state.cache.get("max_score", 1.0)
            value = max(min_score, min(max_score, value))
        except ValueError:
            # Default value if parsing fails
            value = 0.5

        return value

    def validate(self, text: str) -> bool:
        """
        Check if text meets quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards

        Raises:
            ValueError: If text is empty
        """
        self._check_input(text)
        # Value critics always return True for validation
        # as they focus on providing values rather than validation
        return True

    def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Improve text based on value.

        Args:
            text: The text to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Generate value
        value = self.run(task, text)

        # Create improvement prompt
        prompt = (
            f"Task:\n{task}\n\n"
            f"Original response:\n{text}\n\n"
            f"Quality score: {value:.2f} (on a scale of 0 to 1)\n\n"
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

    def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze text and provide a value score.

        Args:
            text: The text to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing score

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Generate value
        value = self.run(task, text)

        # Create critique result
        return {
            "score": value,
            "feedback": f"Quality score: {value:.2f}",
            "issues": [],
            "suggestions": [],
        }


class LACCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """
    A critic that implements the LLM-Based Actor-Critic (LAC) approach.

    This critic combines language feedback and value scoring to improve
    language model-based decision making.

    Based on: Language Feedback Improves Language Model-based Decision Making
    https://arxiv.org/abs/2403.03692
    """

    # Class constants
    DEFAULT_NAME = "lac_critic"
    DEFAULT_DESCRIPTION = "Combines language feedback and value scoring"

    # State management using direct state
    _state = PrivateAttr(default_factory=lambda: None)

    def __init__(
        self,
        config: LACCriticConfig,
        llm_provider: Any,
    ) -> None:
        """
        Initialize the LAC critic.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use

        Raises:
            ValueError: If llm_provider is None
            TypeError: If llm_provider is not a valid provider
        """
        # Validate required parameters
        if llm_provider is None:
            raise ValueError("llm_provider cannot be None")

        # Initialize base class
        super().__init__(config)

        # Initialize state
        from ...utils.state import CriticState

        self._state = CriticState()

        # Create feedback critic config
        feedback_config = FeedbackCriticConfig(
            name=f"{config.name}_feedback",
            description=f"Feedback component for {config.name}",
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            min_confidence=config.min_confidence,
            max_attempts=config.max_attempts,
            cache_size=config.cache_size,
            priority=config.priority,
            cost=config.cost,
            feedback_prompt_template=config.feedback_prompt_template,
            feedback_dimensions=config.feedback_dimensions,
        )

        # Create value critic config
        value_config = ValueCriticConfig(
            name=f"{config.name}_value",
            description=f"Value component for {config.name}",
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            min_confidence=config.min_confidence,
            max_attempts=config.max_attempts,
            cache_size=config.cache_size,
            priority=config.priority,
            cost=config.cost,
            value_prompt_template=config.value_prompt_template,
            value_dimensions=config.value_dimensions,
            min_score=config.min_score,
            max_score=config.max_score,
        )

        # Create feedback and value critics
        feedback_critic = FeedbackCritic(
            config=feedback_config,
            llm_provider=llm_provider,
        )

        value_critic = ValueCritic(
            config=value_config,
            llm_provider=llm_provider,
        )

        # Store components in state
        self._state.model = llm_provider
        self._state.cache = {
            "feedback_critic": feedback_critic,
            "value_critic": value_critic,
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
            raise RuntimeError("LACCritic not properly initialized")

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

    def run(self, task: str, response: str) -> Dict[str, Any]:
        """
        Generate feedback and value for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to provide feedback and value for

        Returns:
            Dictionary containing feedback and value

        Raises:
            ValueError: If response is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(response)

        # Get feedback and value critics
        feedback_critic = self._state.cache.get("feedback_critic")
        value_critic = self._state.cache.get("value_critic")

        # Generate feedback and value
        feedback = feedback_critic.run(task, response)
        value = value_critic.run(task, response)

        # Return results
        return {
            "feedback": feedback,
            "value": value,
        }

    def validate(self, text: str) -> bool:
        """
        Check if text meets quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards

        Raises:
            ValueError: If text is empty
        """
        self._check_input(text)
        # LAC critics always return True for validation
        # as they focus on providing feedback and values rather than validation
        return True

    def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Improve text based on feedback and value.

        Args:
            text: The text to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Generate feedback and value
        result = self.run(task, text)
        feedback = result["feedback"]
        value = result["value"]

        # Create improvement prompt
        prompt = (
            f"Task:\n{task}\n\n"
            f"Original response:\n{text}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Quality score: {value:.2f} (on a scale of 0 to 1)\n\n"
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

    def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze text and provide feedback and value.

        Args:
            text: The text to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing feedback and value

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Generate feedback and value
        result = self.run(task, text)
        feedback = result["feedback"]
        value = result["value"]

        # Create critique result
        return {
            "score": value,
            "feedback": feedback,
            "value": value,
            "issues": [],
            "suggestions": [],
        }


def create_feedback_critic(
    llm_provider: Any,
    name: str = "feedback_critic",
    description: str = "Provides natural language feedback for text",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    feedback_prompt_template: str = None,
    config: Optional[Union[Dict[str, Any], FeedbackCriticConfig]] = None,
    **kwargs: Any,
) -> FeedbackCritic:
    """
    Create a feedback critic with the given parameters.

    This factory function creates a configured feedback critic instance
    that provides natural language feedback for text.
    """
    # Create config if not provided
    if config is None:
        from ..config import DEFAULT_FEEDBACK_CONFIG

        config = DEFAULT_FEEDBACK_CONFIG.model_copy()

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
        if feedback_prompt_template is not None:
            updates["feedback_prompt_template"] = feedback_prompt_template

        config = config.model_copy(update=updates)
    elif isinstance(config, dict):
        from ..config import FeedbackCriticConfig

        config = FeedbackCriticConfig(**config)

    # Create and return the critic
    return FeedbackCritic(
        config=config,
        llm_provider=llm_provider,
    )


def create_value_critic(
    llm_provider: Any,
    name: str = "value_critic",
    description: str = "Provides numeric value scoring for text",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    value_prompt_template: str = None,
    min_score: float = None,
    max_score: float = None,
    config: Optional[Union[Dict[str, Any], ValueCriticConfig]] = None,
    **kwargs: Any,
) -> ValueCritic:
    """
    Create a value critic with the given parameters.

    This factory function creates a configured value critic instance
    that provides numeric value scoring for text.
    """
    # Create config if not provided
    if config is None:
        from ..config import DEFAULT_VALUE_CONFIG

        config = DEFAULT_VALUE_CONFIG.model_copy()

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
        if value_prompt_template is not None:
            updates["value_prompt_template"] = value_prompt_template
        if min_score is not None:
            updates["min_score"] = min_score
        if max_score is not None:
            updates["max_score"] = max_score

        config = config.model_copy(update=updates)
    elif isinstance(config, dict):
        from ..config import ValueCriticConfig

        config = ValueCriticConfig(**config)

    # Create and return the critic
    return ValueCritic(
        config=config,
        llm_provider=llm_provider,
    )


def create_lac_critic(
    llm_provider: Any,
    name: str = "lac_critic",
    description: str = "Combines language feedback and value scoring",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    feedback_prompt_template: str = None,
    value_prompt_template: str = None,
    config: Optional[Union[Dict[str, Any], LACCriticConfig]] = None,
    **kwargs: Any,
) -> LACCritic:
    """
    Create a LAC critic with the given parameters.

    This factory function creates a configured LAC critic instance
    that combines language feedback and value scoring.
    """
    # Create config if not provided
    if config is None:
        from ..config import DEFAULT_LAC_CONFIG

        config = DEFAULT_LAC_CONFIG.model_copy()

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
        if feedback_prompt_template is not None:
            updates["feedback_prompt_template"] = feedback_prompt_template
        if value_prompt_template is not None:
            updates["value_prompt_template"] = value_prompt_template

        config = config.model_copy(update=updates)
    elif isinstance(config, dict):
        from ..config import LACCriticConfig

        config = LACCriticConfig(**config)

    # Create and return the critic
    return LACCritic(
        config=config,
        llm_provider=llm_provider,
    )
