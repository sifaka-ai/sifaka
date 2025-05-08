"""
LAC (LLM-Based Actor-Critic) critic implementation for Sifaka.

This module implements the LAC approach for critics, which combines language feedback
and value scoring to improve language model-based decision making.

Based on: Language Feedback Improves Language Model-based Decision Making
https://arxiv.org/abs/2403.03692

Example:
    ```python
    from sifaka.critics.implementations import LACCriticImplementation
    from sifaka.critics.lac import LACCriticConfig
    from sifaka.critics.base import CompositionCritic, create_composition_critic
    from sifaka.models.providers import OpenAIProvider

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Create a LAC critic implementation
    implementation = LACCriticImplementation(
        config=LACCriticConfig(
            name="lac_critic",
            description="A critic that combines feedback and value scoring",
            system_prompt="You are an expert at evaluating and improving text.",
            temperature=0.7,
            max_tokens=1000
        ),
        llm_provider=provider
    )

    # Create a critic with the implementation
    critic = create_composition_critic(
        name="lac_critic",
        description="A critic that combines feedback and value scoring",
        implementation=implementation
    )

    # Use the critic
    task = "Summarize the causes of World War I in 3 bullet points."
    response = "World War I was caused by nationalism, militarism, and alliances."
    result = critic.critique(response, {"task": task})
    print(f"Feedback: {result['feedback']}")
    print(f"Value: {result['value']}")
    ```
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union, cast

from ..protocols import CriticImplementation
from ..lac import (
    LACCriticConfig,
    FeedbackCriticConfig,
    ValueCriticConfig,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
    DEFAULT_VALUE_PROMPT_TEMPLATE,
)
from ..utils.state import CriticState

# Configure logging
logger = logging.getLogger(__name__)


class FeedbackCriticImplementation(CriticImplementation):
    """
    Implementation of a critic that produces natural language feedback.

    This implementation analyzes text and provides detailed feedback on what could be
    improved or what was done well.
    """

    def __init__(
        self,
        config: FeedbackCriticConfig,
        llm_provider: Any,
    ) -> None:
        """
        Initialize the feedback critic implementation.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use for generating feedback

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        self.config = config
        self._state = CriticState()
        self._state.model = llm_provider
        self._state.cache = {
            "feedback_prompt_template": config.feedback_prompt_template,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        self._state.initialized = True

    def validate_impl(self, text: str, **kwargs: Any) -> bool:
        """
        Validate text quality.

        Args:
            text: Text to validate
            **kwargs: Additional arguments (e.g., task)

        Returns:
            True if text is valid, False otherwise
        """
        if not text or not text.strip():
            return False

        task = kwargs.get("task", "")
        if not task:
            logger.warning("No task provided for feedback validation")
            return True

        # Generate feedback
        feedback = self.run(task, text)

        # Check if feedback indicates major issues
        negative_patterns = [
            r"incorrect",
            r"incomplete",
            r"missing",
            r"wrong",
            r"error",
            r"fail",
            r"poor",
            r"bad",
            r"inadequate",
        ]

        for pattern in negative_patterns:
            if re.search(pattern, feedback, re.IGNORECASE):
                return False

        return True

    def improve_impl(self, text: str, **kwargs: Any) -> str:
        """
        Improve text based on feedback.

        Args:
            text: Text to improve
            **kwargs: Additional arguments (e.g., task, feedback)

        Returns:
            Improved text
        """
        if not text or not text.strip():
            return text

        task = kwargs.get("task", "")
        feedback = kwargs.get("feedback", "")

        if not feedback and task:
            # Generate feedback if not provided
            feedback = self.run(task, text)

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

    def critique_impl(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Critique text and provide feedback.

        Args:
            text: Text to critique
            **kwargs: Additional arguments (e.g., task)

        Returns:
            Dictionary with feedback and metadata
        """
        if not text or not text.strip():
            return {"feedback": "Empty text provided", "score": 0.0}

        task = kwargs.get("task", "")
        if not task:
            logger.warning("No task provided for feedback critique")
            task = "Evaluate the following text"

        # Generate feedback
        feedback = self.run(task, text)

        # Estimate score based on feedback sentiment
        score = self._estimate_score_from_feedback(feedback)

        return {
            "feedback": feedback,
            "score": score,
        }

    def warm_up_impl(self) -> None:
        """Warm up the critic by initializing resources."""
        self._state.initialized = True

    def run(self, task: str, response: str) -> str:
        """
        Generate natural language feedback for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to provide feedback on

        Returns:
            Natural language feedback
        """
        if not response or not response.strip():
            return "Empty text provided"

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

    def _estimate_score_from_feedback(self, feedback: str) -> float:
        """
        Estimate a score from feedback text.

        Args:
            feedback: Feedback text

        Returns:
            Estimated score between 0 and 1
        """
        # Count positive and negative words
        positive_words = [
            "good", "great", "excellent", "well", "clear", "concise",
            "comprehensive", "accurate", "detailed", "thorough"
        ]
        negative_words = [
            "bad", "poor", "unclear", "confusing", "incomplete", "inaccurate",
            "missing", "wrong", "error", "vague"
        ]

        positive_count = sum(1 for word in positive_words if re.search(r"\b" + word + r"\b", feedback, re.IGNORECASE))
        negative_count = sum(1 for word in negative_words if re.search(r"\b" + word + r"\b", feedback, re.IGNORECASE))

        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.5  # Neutral

        score = positive_count / total_count
        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1


class ValueCriticImplementation(CriticImplementation):
    """
    Implementation of a critic that estimates numeric values for text.

    This implementation analyzes text and provides a numeric score (e.g., probability of success)
    for the response.
    """

    def __init__(
        self,
        config: ValueCriticConfig,
        llm_provider: Any,
    ) -> None:
        """
        Initialize the value critic implementation.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use for generating values

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        self.config = config
        self._state = CriticState()
        self._state.model = llm_provider
        self._state.cache = {
            "value_prompt_template": config.value_prompt_template,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        self._state.initialized = True

    def validate_impl(self, text: str, **kwargs: Any) -> bool:
        """
        Validate text quality based on value.

        Args:
            text: Text to validate
            **kwargs: Additional arguments (e.g., task, threshold)

        Returns:
            True if text is valid, False otherwise
        """
        if not text or not text.strip():
            return False

        task = kwargs.get("task", "")
        threshold = kwargs.get("threshold", 0.7)

        if not task:
            logger.warning("No task provided for value validation")
            return True

        # Generate value
        value = self.run(task, text)

        # Check if value meets threshold
        return value >= threshold

    def improve_impl(self, text: str, **kwargs: Any) -> str:
        """
        Improve text based on value.

        Args:
            text: Text to improve
            **kwargs: Additional arguments (e.g., task, feedback)

        Returns:
            Improved text
        """
        if not text or not text.strip():
            return text

        task = kwargs.get("task", "")
        feedback = kwargs.get("feedback", "")

        if not feedback and task:
            # Generate generic feedback based on value
            value = self.run(task, text)
            if value < 0.3:
                feedback = "The response is poor quality. It needs significant improvement in accuracy, completeness, and clarity."
            elif value < 0.7:
                feedback = "The response is average quality. It could be improved with more detail and better organization."
            else:
                feedback = "The response is good quality. Minor improvements could make it excellent."

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

    def critique_impl(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Critique text and provide a value.

        Args:
            text: Text to critique
            **kwargs: Additional arguments (e.g., task)

        Returns:
            Dictionary with value and metadata
        """
        if not text or not text.strip():
            return {"value": 0.0, "score": 0.0}

        task = kwargs.get("task", "")
        if not task:
            logger.warning("No task provided for value critique")
            task = "Evaluate the following text"

        # Generate value
        value = self.run(task, text)

        return {
            "value": value,
            "score": value,
        }

    def warm_up_impl(self) -> None:
        """Warm up the critic by initializing resources."""
        self._state.initialized = True

    def run(self, task: str, response: str) -> float:
        """
        Estimate a numeric value for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to estimate a value for

        Returns:
            Numeric value between 0 and 1
        """
        if not response or not response.strip():
            return 0.0

        # Create value prompt
        prompt = self._state.cache.get("value_prompt_template", "").format(
            task=task,
            response=response,
        )

        # Generate value
        value_str = self._state.model.generate(
            prompt,
            system_prompt=self._state.cache.get("system_prompt", ""),
            temperature=self._state.cache.get("temperature", 0.3),
            max_tokens=self._state.cache.get("max_tokens", 100),
        ).strip()

        # Parse value
        try:
            # Extract the first float from the response
            match = re.search(r"(\d+(\.\d+)?)", value_str)
            if match:
                value = float(match.group(1))
                # Normalize to [0, 1]
                if value > 1.0:
                    value = value / 10.0 if value <= 10.0 else 1.0
                return min(max(value, 0.0), 1.0)  # Clamp between 0 and 1
            else:
                logger.warning(f"Could not parse value from: {value_str}")
                return 0.5  # Default to neutral
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing value: {e}")
            return 0.5  # Default to neutral


class LACCriticImplementation(CriticImplementation):
    """
    Implementation of a critic that combines language feedback and value scoring.

    This implementation uses the LLM-Based Actor-Critic (LAC) approach, which combines
    language feedback and value scoring to improve language model-based decision making.
    """

    def __init__(
        self,
        config: LACCriticConfig,
        llm_provider: Any,
    ) -> None:
        """
        Initialize the LAC critic implementation.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use for generating feedback and values

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        self.config = config
        self._state = CriticState()
        self._state.model = llm_provider

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
        self._state.cache = {
            "feedback_critic": FeedbackCriticImplementation(config=feedback_config, llm_provider=llm_provider),
            "value_critic": ValueCriticImplementation(config=value_config, llm_provider=llm_provider),
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        self._state.initialized = True

    def validate_impl(self, text: str, **kwargs: Any) -> bool:
        """
        Validate text quality using both feedback and value.

        Args:
            text: Text to validate
            **kwargs: Additional arguments (e.g., task, threshold)

        Returns:
            True if text is valid, False otherwise
        """
        if not text or not text.strip():
            return False

        task = kwargs.get("task", "")
        threshold = kwargs.get("threshold", 0.7)

        if not task:
            logger.warning("No task provided for LAC validation")
            return True

        # Run both critics
        result = self.run(task, text)

        # Check if value meets threshold
        return result["value"] >= threshold

    def improve_impl(self, text: str, **kwargs: Any) -> str:
        """
        Improve text based on feedback and value.

        Args:
            text: Text to improve
            **kwargs: Additional arguments (e.g., task, feedback)

        Returns:
            Improved text
        """
        if not text or not text.strip():
            return text

        task = kwargs.get("task", "")
        feedback = kwargs.get("feedback", "")

        if not feedback and task:
            # Generate feedback
            result = self.run(task, text)
            feedback = result["feedback"]

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

    def critique_impl(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Critique text using both feedback and value.

        Args:
            text: Text to critique
            **kwargs: Additional arguments (e.g., task)

        Returns:
            Dictionary with feedback, value, and metadata
        """
        if not text or not text.strip():
            return {"feedback": "Empty text provided", "value": 0.0, "score": 0.0}

        task = kwargs.get("task", "")
        if not task:
            logger.warning("No task provided for LAC critique")
            task = "Evaluate the following text"

        # Run both critics
        result = self.run(task, text)

        return {
            "feedback": result["feedback"],
            "value": result["value"],
            "score": result["value"],  # Use value as score
        }

    def warm_up_impl(self) -> None:
        """Warm up the critic by initializing resources."""
        self._state.initialized = True

    def run(self, task: str, response: str) -> Dict[str, Any]:
        """
        Run the LAC critic on a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to critique

        Returns:
            Dictionary with feedback and value
        """
        if not response or not response.strip():
            return {"feedback": "Empty text provided", "value": 0.0}

        # Get feedback and value critics
        feedback_critic = self._state.cache.get("feedback_critic")
        value_critic = self._state.cache.get("value_critic")

        # Generate feedback and value
        feedback = feedback_critic.run(task, response)
        value = value_critic.run(task, response)

        return {
            "feedback": feedback,
            "value": value,
        }
