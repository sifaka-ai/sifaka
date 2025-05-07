"""
LAC (LLM-Based Actor-Critic) critic module for Sifaka.

This module implements the LAC approach for critics, which combines language feedback
and value scoring to improve language model-based decision making.

Based on: Language Feedback Improves Language Model-based Decision Making
https://arxiv.org/abs/2403.03692

Example:
    ```python
    from sifaka.critics.lac import create_lac_critic
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

from typing import Any, Dict, List, Optional, Union

from pydantic import Field, PrivateAttr, ConfigDict

from .base import BaseCritic, TextCritic, TextImprover, TextValidator
from .models import CriticConfig, PromptCriticConfig
from ..utils.state import create_critic_state

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


class FeedbackCriticConfig(PromptCriticConfig):
    """
    Configuration for feedback critics.

    This model extends PromptCriticConfig with feedback-specific settings
    for critics that generate natural language feedback.

    Examples:
        ```python
        from sifaka.critics.lac import FeedbackCriticConfig

        # Create a feedback critic config
        config = FeedbackCriticConfig(
            name="feedback_critic",
            description="A critic that provides natural language feedback",
            system_prompt="You are an expert at providing constructive feedback.",
            temperature=0.7,
            max_tokens=1000,
            feedback_prompt_template="Task: {task}\nResponse: {response}\nProvide feedback:"
        )
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    feedback_prompt_template: str = Field(
        default=DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
        description="Template for feedback prompts",
    )


class ValueCriticConfig(PromptCriticConfig):
    """
    Configuration for value critics.

    This model extends PromptCriticConfig with value-specific settings
    for critics that estimate numeric values (e.g., probability of success).

    Examples:
        ```python
        from sifaka.critics.lac import ValueCriticConfig

        # Create a value critic config
        config = ValueCriticConfig(
            name="value_critic",
            description="A critic that estimates numeric values",
            system_prompt="You are an expert at estimating the quality of responses.",
            temperature=0.3,
            max_tokens=100,
            value_prompt_template="Task: {task}\nResponse: {response}\nEstimate quality (0-1):"
        )
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value_prompt_template: str = Field(
        default=DEFAULT_VALUE_PROMPT_TEMPLATE,
        description="Template for value prompts",
    )


class LACCriticConfig(CriticConfig):
    """
    Configuration for LAC critics.

    This model extends CriticConfig with LAC-specific settings
    for critics that combine language feedback and value scoring.

    Examples:
        ```python
        from sifaka.critics.lac import LACCriticConfig

        # Create a LAC critic config
        config = LACCriticConfig(
            name="lac_critic",
            description="A critic that combines feedback and value scoring",
            system_prompt="You are an expert at evaluating and improving text.",
            temperature=0.7,
            max_tokens=1000,
            feedback_prompt_template="Task: {task}\nResponse: {response}\nProvide feedback:",
            value_prompt_template="Task: {task}\nResponse: {response}\nEstimate quality (0-1):"
        )
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt for the model",
        min_length=1,
    )
    temperature: float = Field(
        default=0.7, description="Temperature for model generation", ge=0.0, le=1.0
    )
    max_tokens: int = Field(default=1000, description="Maximum tokens for model generation", gt=0)
    feedback_prompt_template: str = Field(
        default=DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
        description="Template for feedback prompts",
    )
    value_prompt_template: str = Field(
        default=DEFAULT_VALUE_PROMPT_TEMPLATE,
        description="Template for value prompts",
    )


class FeedbackCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """
    A critic that produces natural language feedback for a model's response to a task.

    This critic analyzes text and provides detailed feedback on what could be
    improved or what was done well.

    Examples:
        ```python
        from sifaka.critics.lac import FeedbackCritic, FeedbackCriticConfig
        from sifaka.models.providers import OpenAIProvider

        # Create a language model provider
        provider = OpenAIProvider(api_key="your-api-key")

        # Create a feedback critic configuration
        config = FeedbackCriticConfig(
            name="feedback_critic",
            description="A critic that provides natural language feedback",
            system_prompt="You are an expert at providing constructive feedback.",
            temperature=0.7,
            max_tokens=1000
        )

        # Create a feedback critic
        critic = FeedbackCritic(
            config=config,
            llm_provider=provider
        )

        # Use the critic
        task = "Summarize the causes of World War I in 3 bullet points."
        response = "World War I was caused by nationalism, militarism, and alliances."
        feedback = critic.run(task, response)
        print(f"Feedback: {feedback}")
        ```
    """

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        config: FeedbackCriticConfig,
        llm_provider: Any,
    ) -> None:
        """
        Initialize the feedback critic.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use for generating feedback

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        # Initialize base class
        super().__init__(config)

        # Initialize state
        state = self._state_manager.get_state()

        # Store components in state
        state.model = llm_provider
        state.cache = {
            "feedback_prompt_template": config.feedback_prompt_template,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        state.initialized = True

    def _check_input(self, text: str) -> None:
        """
        Check if input text is valid.

        Args:
            text: The text to check

        Raises:
            ValueError: If text is empty
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

    def _get_task_from_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Get task from metadata.

        Args:
            metadata: Optional metadata containing the task

        Returns:
            The task string
        """
        if metadata and "task" in metadata:
            return str(metadata["task"])
        return "Evaluate the following text"

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

        state = self._state_manager.get_state()

        # Create feedback prompt
        prompt = state.cache.get("feedback_prompt_template", "").format(
            task=task,
            response=response,
        )

        # Generate feedback
        feedback = state.model.generate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
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

        state = self._state_manager.get_state()

        # Create feedback prompt
        prompt = state.cache.get("feedback_prompt_template", "").format(
            task=task,
            response=response,
        )

        # Generate feedback
        feedback = await state.model.agenerate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
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

        state = self._state_manager.get_state()

        # Generate improved response
        improved_text = state.model.generate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
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

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text or feedback is invalid
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        if not feedback or not isinstance(feedback, str):
            raise ValueError("feedback must be a non-empty string")

        # Create improvement prompt
        prompt = f"Original text:\n{text}\n\n" f"Feedback:\n{feedback}\n\n" f"Improved text:"

        state = self._state_manager.get_state()

        # Generate improved response
        improved_text = state.model.generate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
        ).strip()

        return improved_text


class ValueCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """
    A critic that estimates a numeric value for a model's response to a task.

    This critic analyzes text and provides a numeric score (e.g., probability of success)
    for the response.

    Examples:
        ```python
        from sifaka.critics.lac import ValueCritic, ValueCriticConfig
        from sifaka.models.providers import OpenAIProvider

        # Create a language model provider
        provider = OpenAIProvider(api_key="your-api-key")

        # Create a value critic configuration
        config = ValueCriticConfig(
            name="value_critic",
            description="A critic that estimates numeric values",
            system_prompt="You are an expert at estimating the quality of responses.",
            temperature=0.3,
            max_tokens=100
        )

        # Create a value critic
        critic = ValueCritic(
            config=config,
            llm_provider=provider
        )

        # Use the critic
        task = "Summarize the causes of World War I in 3 bullet points."
        response = "World War I was caused by nationalism, militarism, and alliances."
        value = critic.run(task, response)
        print(f"Value: {value}")
        ```
    """

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        config: ValueCriticConfig,
        llm_provider: Any,
    ) -> None:
        """
        Initialize the value critic.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use for generating values

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        # Initialize base class
        super().__init__(config)

        # Initialize state
        state = self._state_manager.get_state()

        # Store components in state
        state.model = llm_provider
        state.cache = {
            "value_prompt_template": config.value_prompt_template,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        state.initialized = True

    def _check_input(self, text: str) -> None:
        """
        Check if input text is valid.

        Args:
            text: The text to check

        Raises:
            ValueError: If text is empty
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

    def _get_task_from_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Get task from metadata.

        Args:
            metadata: Optional metadata containing the task

        Returns:
            The task string
        """
        if metadata and "task" in metadata:
            return str(metadata["task"])
        return "Evaluate the following text"

    def run(self, task: str, response: str) -> float:
        """
        Estimate a numeric value for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to estimate a value for

        Returns:
            Numeric value between 0 and 1

        Raises:
            ValueError: If response is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(response)

        state = self._state_manager.get_state()

        # Create value prompt
        prompt = state.cache.get("value_prompt_template", "").format(
            task=task,
            response=response,
        )

        # Generate value
        value_str = state.model.generate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.3),
            max_tokens=state.cache.get("max_tokens", 100),
        ).strip()

        # Parse value
        try:
            value = float(value_str)
            # Clamp value to [0, 1]
            value = max(0.0, min(1.0, value))
            return value
        except ValueError:
            # If value cannot be parsed, return a default value
            return 0.5

    async def arun(self, task: str, response: str) -> float:
        """
        Asynchronously estimate a numeric value for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to estimate a value for

        Returns:
            Numeric value between 0 and 1

        Raises:
            ValueError: If response is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(response)

        state = self._state_manager.get_state()

        # Create value prompt
        prompt = state.cache.get("value_prompt_template", "").format(
            task=task,
            response=response,
        )

        # Generate value
        value_str = await state.model.agenerate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.3),
            max_tokens=state.cache.get("max_tokens", 100),
        )
        value_str = value_str.strip()

        # Parse value
        try:
            value = float(value_str)
            # Clamp value to [0, 1]
            value = max(0.0, min(1.0, value))
            return value
        except ValueError:
            # If value cannot be parsed, return a default value
            return 0.5

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

        state = self._state_manager.get_state()

        # Generate improved response
        improved_text = state.model.generate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
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

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text or feedback is invalid
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        if not feedback or not isinstance(feedback, str):
            raise ValueError("feedback must be a non-empty string")

        # Create improvement prompt
        prompt = f"Original text:\n{text}\n\n" f"Feedback:\n{feedback}\n\n" f"Improved text:"

        state = self._state_manager.get_state()

        # Generate improved response
        improved_text = state.model.generate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
        ).strip()

        return improved_text


class LACCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """
    A critic that implements the LLM-Based Actor-Critic (LAC) approach.

    This critic combines language feedback and value scoring to improve
    language model-based decision making. It uses two specialized components:
    a FeedbackCritic that provides qualitative natural language feedback, and
    a ValueCritic that estimates quantitative scores for text quality.

    Based on: Language Feedback Improves Language Model-based Decision Making
    https://arxiv.org/abs/2403.03692

    ## Lifecycle Management

    The LACCritic manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up language model provider
       - Creates FeedbackCritic and ValueCritic components
       - Initializes state
       - Allocates resources

    2. **Operation**
       - Validates text quality
       - Generates qualitative feedback
       - Estimates quantitative value scores
       - Improves text based on feedback
       - Combines feedback and value information

    3. **Cleanup**
       - Releases resources
       - Clears state
       - Logs final status

    ## Architecture

    The LACCritic uses a composite architecture with three main components:

    1. **FeedbackCritic**: Provides natural language feedback for text
       - Analyzes text for issues and improvement opportunities
       - Generates detailed, actionable feedback
       - Focuses on qualitative assessment

    2. **ValueCritic**: Estimates numeric values for text
       - Evaluates text quality on a numeric scale (0.0-1.0)
       - Provides quantitative assessment
       - Focuses on objective metrics

    3. **LACCritic**: Combines both feedback and value critics
       - Coordinates between the two specialized critics
       - Integrates qualitative and quantitative assessments
       - Provides comprehensive evaluation and improvement

    ## Error Handling

    The LACCritic handles errors through:

    1. **Input Validation**
       - Empty text checks
       - Missing task information
       - Type validation
       - Format verification

    2. **Component Errors**
       - FeedbackCritic failures
       - ValueCritic failures
       - Model generation errors
       - Prompt formatting issues

    3. **Recovery Strategies**
       - Fallback to partial results
       - Default values for missing components
       - Graceful degradation
       - Comprehensive error reporting

    ## Examples

    Basic usage with factory function:

    ```python
    from sifaka.critics import create_lac_critic
    from sifaka.models.openai import create_openai_provider

    # Create a language model provider
    provider = create_openai_provider(
        model_name="gpt-3.5-turbo",
        api_key="your-api-key",
        temperature=0.7,
        max_tokens=1000
    )

    # Create a LAC critic
    critic = create_lac_critic(
        llm_provider=provider,
        name="demo_lac_critic",
        description="A demo LAC critic",
        system_prompt="You are an expert at evaluating and improving text."
    )

    # Define a task and response
    task = "Summarize the causes of World War I in 3 bullet points."
    response = "World War I was caused by nationalism, militarism, and alliances."

    # Get critique with both feedback and value
    result = critic.critique(response, {"task": task})
    print(f"Feedback: {result['feedback']}")
    print(f"Value score: {result['value']}")

    # Improve the response
    improved_response = critic.improve(response, {"task": task})
    print(f"Improved response: {improved_response}")
    ```

    Advanced usage with custom configuration:

    ```python
    from sifaka.critics.lac import LACCritic, LACCriticConfig
    from sifaka.models.anthropic import create_anthropic_provider

    # Create a language model provider
    provider = create_anthropic_provider(
        model_name="claude-3-haiku-20240307",
        api_key="your-api-key"
    )

    # Create a custom configuration with specialized prompt templates
    config = LACCriticConfig(
        name="custom_lac_critic",
        description="A critic for evaluating and improving technical documentation",
        system_prompt="You are an expert technical writer and evaluator.",
        temperature=0.5,
        max_tokens=2000,
        feedback_prompt_template=(
            "You are evaluating technical documentation.\n\n"
            "Task: {task}\n\n"
            "Response: {response}\n\n"
            "Provide detailed feedback on clarity, accuracy, completeness, and structure:"
        ),
        value_prompt_template=(
            "You are scoring technical documentation quality.\n\n"
            "Task: {task}\n\n"
            "Response: {response}\n\n"
            "Rate the quality from 0.0 (poor) to 1.0 (excellent) based on clarity, "
            "accuracy, completeness, and structure. Return only the numeric score:"
        )
    )

    # Create a LAC critic with custom configuration
    critic = LACCritic(
        config=config,
        llm_provider=provider
    )

    # Use individual components separately
    task = "Explain how to install Python on Windows."
    response = "Download Python from python.org and run the installer."

    # Get feedback only
    state = critic._state_manager.get_state()
    feedback_critic = state.cache["feedback_critic"]
    feedback = feedback_critic.run(task, response)
    print(f"Feedback: {feedback}")

    # Get value score only
    value_critic = state.cache["value_critic"]
    value = value_critic.run(task, response)
    print(f"Value score: {value}")

    # Use the combined critic
    result = critic.run(task, response)
    print(f"Combined result: {result}")

    # Improve the response
    improved_response = critic.improve(response, {"task": task})
    print(f"Improved response: {improved_response}")
    ```

    Using with asynchronous methods:

    ```python
    import asyncio
    from sifaka.critics import create_lac_critic
    from sifaka.models.openai import create_openai_provider

    async def evaluate_text():
        provider = create_openai_provider(
            model_name="gpt-4",
            api_key="your-api-key"
        )

        critic = create_lac_critic(llm_provider=provider)

        task = "Write a concise definition of machine learning."
        response = "Machine learning is when computers learn from data."

        # Get critique asynchronously
        critique = await critic.acritique(response, {"task": task})
        print(f"Feedback: {critique['feedback']}")
        print(f"Value: {critique['value']}")

        # Improve the response asynchronously
        improved = await critic.aimprove(response, {"task": task})
        return improved

    # Run the async function
    improved_response = asyncio.run(evaluate_text())
    ```
    """

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        config: LACCriticConfig,
        llm_provider: Any,
    ) -> None:
        """
        Initialize the LAC critic.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use for generating feedback and values

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        # Initialize base class
        super().__init__(config)

        # Initialize state
        state = self._state_manager.get_state()

        # Create feedback critic configuration
        feedback_config = FeedbackCriticConfig(
            name=f"{config.name}_feedback",
            description=f"Feedback component of {config.name}",
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            feedback_prompt_template=config.feedback_prompt_template,
        )

        # Create value critic configuration
        value_config = ValueCriticConfig(
            name=f"{config.name}_value",
            description=f"Value component of {config.name}",
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            value_prompt_template=config.value_prompt_template,
        )

        # Create feedback and value critics
        state.cache = {
            "feedback_critic": FeedbackCritic(config=feedback_config, llm_provider=llm_provider),
            "value_critic": ValueCritic(config=value_config, llm_provider=llm_provider),
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        state.model = llm_provider
        state.initialized = True

    def _check_input(self, text: str) -> None:
        """
        Check if input text is valid.

        Args:
            text: The text to check

        Raises:
            ValueError: If text is empty
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

    def _get_task_from_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Get task from metadata.

        Args:
            metadata: Optional metadata containing the task

        Returns:
            The task string
        """
        if metadata and "task" in metadata:
            return str(metadata["task"])
        return "Evaluate the following text"

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

        state = self._state_manager.get_state()

        # Get feedback and value critics
        feedback_critic = state.cache.get("feedback_critic")
        value_critic = state.cache.get("value_critic")

        # Generate feedback and value
        feedback = feedback_critic.run(task, response)
        value = value_critic.run(task, response)

        # Return results
        return {
            "feedback": feedback,
            "value": value,
        }

    async def arun(self, task: str, response: str) -> Dict[str, Any]:
        """
        Asynchronously generate feedback and value for a response to a task.

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

        state = self._state_manager.get_state()

        # Get feedback and value critics
        feedback_critic = state.cache.get("feedback_critic")
        value_critic = state.cache.get("value_critic")

        # Generate feedback and value
        feedback = await feedback_critic.arun(task, response)
        value = await value_critic.arun(task, response)

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

        state = self._state_manager.get_state()

        # Generate improved response
        improved_text = state.model.generate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
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

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text or feedback is invalid
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        if not feedback or not isinstance(feedback, str):
            raise ValueError("feedback must be a non-empty string")

        # Create improvement prompt
        prompt = f"Original text:\n{text}\n\n" f"Feedback:\n{feedback}\n\n" f"Improved text:"

        state = self._state_manager.get_state()

        # Generate improved response
        improved_text = state.model.generate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
        ).strip()

        return improved_text


def create_feedback_critic(
    llm_provider: Any,
    name: str = "feedback_critic",
    description: str = "Provides natural language feedback for text",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    feedback_prompt_template: str = DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
    config: Optional[Union[Dict[str, Any], FeedbackCriticConfig]] = None,
    **kwargs: Any,
) -> FeedbackCritic:
    """
    Create a feedback critic with the given parameters.

    This factory function creates a configured feedback critic instance
    that provides natural language feedback for text.

    Args:
        llm_provider: Language model provider to use for generating feedback
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
        feedback_prompt_template: Template for feedback prompts
        config: Optional configuration object
        **kwargs: Additional keyword arguments for the config

    Returns:
        A configured feedback critic

    Examples:
        ```python
        from sifaka.critics.lac import create_feedback_critic
        from sifaka.models.providers import OpenAIProvider

        # Create a language model provider
        provider = OpenAIProvider(api_key="your-api-key")

        # Create a feedback critic
        critic = create_feedback_critic(
            llm_provider=provider,
            name="my_feedback_critic",
            description="A custom feedback critic",
            system_prompt="You are an expert at providing constructive feedback.",
            temperature=0.7,
            max_tokens=1000
        )

        # Use the critic
        task = "Summarize the causes of World War I in 3 bullet points."
        response = "World War I was caused by nationalism, militarism, and alliances."
        feedback = critic.run(task, response)
        print(f"Feedback: {feedback}")
        ```
    """
    # Try to use standardize_critic_config if available
    try:
        from ..utils.config import standardize_critic_config

        # If standardize_critic_config is available, use it
        critic_config = standardize_critic_config(
            config=config,
            config_class=FeedbackCriticConfig,
            name=name,
            description=description,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            feedback_prompt_template=feedback_prompt_template,
            **kwargs,
        )
    except (ImportError, AttributeError):
        # If standardize_critic_config is not available, create config manually
        if config is None:
            critic_config = FeedbackCriticConfig(
                name=name,
                description=description,
                min_confidence=min_confidence,
                max_attempts=max_attempts,
                cache_size=cache_size,
                priority=priority,
                cost=cost,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                feedback_prompt_template=feedback_prompt_template,
                params=kwargs,
            )
        elif isinstance(config, dict):
            critic_config = FeedbackCriticConfig(**config)
        else:
            critic_config = config

    # Create and return the critic
    return FeedbackCritic(config=critic_config, llm_provider=llm_provider)


def create_value_critic(
    llm_provider: Any,
    name: str = "value_critic",
    description: str = "Estimates numeric values for text",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.3,
    max_tokens: int = 100,
    value_prompt_template: str = DEFAULT_VALUE_PROMPT_TEMPLATE,
    config: Optional[Union[Dict[str, Any], ValueCriticConfig]] = None,
    **kwargs: Any,
) -> ValueCritic:
    """
    Create a value critic with the given parameters.

    This factory function creates a configured value critic instance
    that estimates numeric values for text.

    Args:
        llm_provider: Language model provider to use for generating values
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
        value_prompt_template: Template for value prompts
        config: Optional configuration object
        **kwargs: Additional keyword arguments for the config

    Returns:
        A configured value critic

    Examples:
        ```python
        from sifaka.critics.lac import create_value_critic
        from sifaka.models.providers import OpenAIProvider

        # Create a language model provider
        provider = OpenAIProvider(api_key="your-api-key")

        # Create a value critic
        critic = create_value_critic(
            llm_provider=provider,
            name="my_value_critic",
            description="A custom value critic",
            system_prompt="You are an expert at estimating the quality of responses.",
            temperature=0.3,
            max_tokens=100
        )

        # Use the critic
        task = "Summarize the causes of World War I in 3 bullet points."
        response = "World War I was caused by nationalism, militarism, and alliances."
        value = critic.run(task, response)
        print(f"Value: {value}")
        ```
    """
    # Try to use standardize_critic_config if available
    try:
        from ..utils.config import standardize_critic_config

        # If standardize_critic_config is available, use it
        critic_config = standardize_critic_config(
            config=config,
            config_class=ValueCriticConfig,
            name=name,
            description=description,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            value_prompt_template=value_prompt_template,
            **kwargs,
        )
    except (ImportError, AttributeError):
        # If standardize_critic_config is not available, create config manually
        if config is None:
            critic_config = ValueCriticConfig(
                name=name,
                description=description,
                min_confidence=min_confidence,
                max_attempts=max_attempts,
                cache_size=cache_size,
                priority=priority,
                cost=cost,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                value_prompt_template=value_prompt_template,
                params=kwargs,
            )
        elif isinstance(config, dict):
            critic_config = ValueCriticConfig(**config)
        else:
            critic_config = config

    # Create and return the critic
    return ValueCritic(config=critic_config, llm_provider=llm_provider)


def create_lac_critic(
    llm_provider: Any,
    name: str = "lac_critic",
    description: str = "Combines language feedback and value scoring",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    feedback_prompt_template: str = DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
    value_prompt_template: str = DEFAULT_VALUE_PROMPT_TEMPLATE,
    config: Optional[Union[Dict[str, Any], LACCriticConfig]] = None,
    **kwargs: Any,
) -> LACCritic:
    """
    Create a LAC critic with the given parameters.

    This factory function creates a configured LAC critic instance
    that combines language feedback and value scoring.

    Args:
        llm_provider: Language model provider to use for generating feedback and values
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
        feedback_prompt_template: Template for feedback prompts
        value_prompt_template: Template for value prompts
        config: Optional configuration object
        **kwargs: Additional keyword arguments for the config

    Returns:
        A configured LAC critic

    Examples:
        ```python
        from sifaka.critics.lac import create_lac_critic
        from sifaka.models.providers import OpenAIProvider

        # Create a language model provider
        provider = OpenAIProvider(api_key="your-api-key")

        # Create a LAC critic
        critic = create_lac_critic(
            llm_provider=provider,
            name="my_lac_critic",
            description="A custom LAC critic",
            system_prompt="You are an expert at evaluating and improving text.",
            temperature=0.7,
            max_tokens=1000
        )

        # Use the critic
        task = "Summarize the causes of World War I in 3 bullet points."
        response = "World War I was caused by nationalism, militarism, and alliances."
        result = critic.critique(response, {"task": task})
        print(f"Feedback: {result['feedback']}")
        print(f"Value: {result['value']}")
        ```
    """
    # Try to use standardize_critic_config if available
    try:
        from ..utils.config import standardize_critic_config

        # If standardize_critic_config is available, use it
        critic_config = standardize_critic_config(
            config=config,
            config_class=LACCriticConfig,
            name=name,
            description=description,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            feedback_prompt_template=feedback_prompt_template,
            value_prompt_template=value_prompt_template,
            **kwargs,
        )
    except (ImportError, AttributeError):
        # If standardize_critic_config is not available, create config manually
        if config is None:
            critic_config = LACCriticConfig(
                name=name,
                description=description,
                min_confidence=min_confidence,
                max_attempts=max_attempts,
                cache_size=cache_size,
                priority=priority,
                cost=cost,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                feedback_prompt_template=feedback_prompt_template,
                value_prompt_template=value_prompt_template,
                params=kwargs,
            )
        elif isinstance(config, dict):
            critic_config = LACCriticConfig(**config)
        else:
            critic_config = config

    # Create and return the critic
    return LACCritic(config=critic_config, llm_provider=llm_provider)
