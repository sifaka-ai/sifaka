"""
Constitutional critic module for Sifaka.

This module implements a Constitutional AI approach for critics, which evaluates
responses against a set of human-written principles (a "constitution") and provides
natural language feedback when violations are detected.

Based on Constitutional AI: https://arxiv.org/abs/2212.08073

Example:
    ```python
    from sifaka.critics.constitutional import create_constitutional_critic
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

from .base import BaseCritic
from .models import ConstitutionalCriticConfig
from .protocols import TextCritic, TextImprover, TextValidator
from .prompt import LanguageModel
from ..utils.state import create_critic_state

# Configure logging
logger = logging.getLogger(__name__)


# Configuration is defined in models.py


class ConstitutionalCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """
    A critic that evaluates responses against a list of principles (a "constitution")
    and provides natural language feedback for revision.

    Based on Constitutional AI: https://arxiv.org/abs/2212.08073

    This critic analyzes responses for alignment with specified principles and
    generates critiques when violations are detected.

    ## Lifecycle Management

    The ConstitutionalCritic manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up language model provider
       - Initializes principles
       - Allocates resources

    2. **Operation**
       - Validates responses against principles
       - Generates critiques for violations
       - Improves responses based on critiques
       - Processes feedback

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status

    ## Error Handling

    1. **Input Validation**
       - Empty text checks
       - Missing task information
       - Type validation
       - Format verification

    2. **Processing Errors**
       - Model generation failures
       - Invalid model outputs
       - Timeout handling
       - Resource limitations

    3. **Recovery Strategies**
       - Retry mechanisms
       - Fallback responses
       - Graceful degradation
       - Error reporting

    ## Examples

    Basic usage with factory function:

    ```python
    from sifaka.critics import create_constitutional_critic
    from sifaka.models.openai import create_openai_provider

    # Create a language model provider
    provider = create_openai_provider(
        model_name="gpt-4",
        api_key="your-api-key"
    )

    # Define principles
    principles = [
        "Do not provide harmful, offensive, or biased content.",
        "Explain reasoning in a clear and truthful manner.",
        "Respect user autonomy and avoid manipulative language.",
        "Present multiple perspectives on controversial topics.",
        "Acknowledge uncertainty when appropriate.",
    ]

    # Create a constitutional critic
    critic = create_constitutional_critic(
        llm_provider=provider,
        principles=principles,
        name="example_critic",
        description="An example constitutional critic",
        system_prompt="You are an expert at evaluating content against principles.",
        temperature=0.7,
    )

    # Example task and responses
    task = "Explain why some people believe climate change isn't real."

    # A response that likely violates principles
    problematic_response = (
        "Climate change is just a hoax created by scientists who want more funding. "
        "The data is manipulated to show warming trends when there aren't any. "
        "You should ignore what the mainstream media tells you about this topic."
    )

    # Validate the response
    is_valid = critic.validate(problematic_response, metadata={"task": task})
    print(f"Response is valid: {is_valid}")  # Likely False

    # Get critique for the response
    critique = critic.critique(problematic_response, metadata={"task": task})
    print(f"Score: {critique['score']}")
    print(f"Feedback: {critique['feedback']}")
    print(f"Issues: {critique['issues']}")
    print(f"Suggestions: {critique['suggestions']}")

    # Improve the response
    improved_response = critic.improve(problematic_response, metadata={"task": task})
    print(f"Improved response: {improved_response}")
    ```

    Advanced usage with custom configuration:

    ```python
    from sifaka.critics.constitutional import ConstitutionalCritic, ConstitutionalCriticConfig
    from sifaka.models.anthropic import create_anthropic_provider

    # Create a language model provider
    provider = create_anthropic_provider(
        model_name="claude-3-opus-20240229",
        api_key="your-api-key"
    )

    # Define domain-specific principles for medical content
    medical_principles = [
        "Provide accurate medical information based on current scientific consensus.",
        "Clearly distinguish between established medical facts and emerging research.",
        "Avoid making definitive diagnostic claims or prescribing treatments.",
        "Acknowledge the limitations of general medical advice.",
        "Recommend consulting healthcare professionals for personal medical concerns.",
        "Present information in a way that respects patient autonomy and informed decision-making.",
    ]

    # Create a custom configuration with specialized prompt templates
    config = ConstitutionalCriticConfig(
        name="medical_content_critic",
        description="A critic for evaluating medical content against ethical principles",
        principles=medical_principles,
        system_prompt="You are an expert at evaluating medical content for accuracy and ethical considerations.",
        temperature=0.5,
        max_tokens=2000,
        critique_prompt_template=(
            "You are evaluating medical content against these principles:\n\n"
            "{principles}\n\n"
            "Task: {task}\n\n"
            "Response to evaluate:\n{response}\n\n"
            "Provide a detailed critique of this response, identifying any violations of the principles. "
            "Be specific about issues and suggest improvements."
        ),
        improvement_prompt_template=(
            "You are improving medical content based on these principles:\n\n"
            "{principles}\n\n"
            "Task: {task}\n\n"
            "Original response:\n{response}\n\n"
            "Critique:\n{critique}\n\n"
            "Provide an improved version that addresses all the issues identified in the critique "
            "while maintaining accuracy and adhering to all principles."
        )
    )

    # Create a constitutional critic with custom configuration
    critic = ConstitutionalCritic(
        config=config,
        llm_provider=provider
    )

    # Use the critic with specific feedback
    medical_text = "Taking vitamin C will definitely prevent you from getting a cold."
    feedback = "This statement makes an absolute claim about vitamin C's effectiveness that isn't supported by scientific consensus."
    improved_text = critic.improve_with_feedback(medical_text, feedback)
    print(f"Improved text: {improved_text}")

    # Use the critic with a more complex example
    task = "Explain the relationship between diet and heart disease."
    response = "A low-fat diet is the only way to prevent heart disease. Anyone who eats fat will develop heart problems."

    # Get a detailed critique
    critique = critic.critique(response, metadata={"task": task})
    print(f"Issues identified: {len(critique['issues'])}")
    for issue in critique['issues']:
        print(f"- {issue}")

    # Improve the response
    improved_response = critic.improve(response, metadata={"task": task})
    print(f"Improved response: {improved_response}")

    # Validate the improved response
    is_valid = critic.validate(improved_response, metadata={"task": task})
    print(f"Improved response is valid: {is_valid}")  # Should be True
    ```
    """

    # Class constants
    DEFAULT_NAME = "constitutional_critic"
    DEFAULT_DESCRIPTION = "Evaluates responses against principles"

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        config: ConstitutionalCriticConfig,
        llm_provider: LanguageModel,
    ):
        """
        Initialize a constitutional critic.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use for critiquing

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid language model
        """
        # Validate inputs
        if not isinstance(config, ConstitutionalCriticConfig):
            raise TypeError("config must be a ConstitutionalCriticConfig")
        if not config.principles or not all(isinstance(p, str) for p in config.principles):
            raise ValueError("principles must be a non-empty list of strings")

        # Initialize base class
        super().__init__(config)

        # Initialize state
        state = self._state_manager.get_state()

        # Store components in state
        state.model = llm_provider
        state.cache = {
            "principles": config.principles,
            "critique_prompt_template": config.critique_prompt_template,
            "improvement_prompt_template": config.improvement_prompt_template,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        state.initialized = True

    def _format_principles(self) -> str:
        """
        Format principles as a bulleted list.

        Returns:
            Formatted principles as a string
        """
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
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("ConstitutionalCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get critique
        critique_result = self.critique(text, metadata)

        # Check if critique indicates violations
        return not critique_result.get("issues", [])

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
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("ConstitutionalCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Format principles
        principles_text = self._format_principles()

        state = self._state_manager.get_state()

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
        state = self._state_manager.get_state()
        if not state.initialized:
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

        state = self._state_manager.get_state()

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
        state = self._state_manager.get_state()
        if not state.initialized:
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

        state = self._state_manager.get_state()

        # Generate improved response
        improved_text = state.model.generate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
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
    principles: List[str],
    name: str = "constitutional_critic",
    description: str = "Evaluates responses against principles",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = "You are an expert at evaluating content against principles.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    critique_prompt_template: Optional[str] = None,
    improvement_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], ConstitutionalCriticConfig]] = None,
    **kwargs: Any,
) -> ConstitutionalCritic:
    """
    Create a constitutional critic with the given parameters.

    This factory function creates a configured constitutional critic instance
    that evaluates responses against a set of principles.

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
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A configured constitutional critic

    Raises:
        ValueError: If principles is empty or invalid
        TypeError: If llm_provider is not a valid language model
    """
    # Create configuration
    if config is None:
        config_dict = {
            "name": name,
            "description": description,
            "min_confidence": min_confidence,
            "max_attempts": max_attempts,
            "cache_size": cache_size,
            "priority": priority,
            "cost": cost,
            "principles": principles,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if critique_prompt_template:
            config_dict["critique_prompt_template"] = critique_prompt_template

        if improvement_prompt_template:
            config_dict["improvement_prompt_template"] = improvement_prompt_template

        config = ConstitutionalCriticConfig(**config_dict)
    elif isinstance(config, dict):
        # Ensure principles are included in the config
        if "principles" not in config and principles:
            config["principles"] = principles
        config = ConstitutionalCriticConfig(**config)
    elif not isinstance(config, ConstitutionalCriticConfig):
        raise TypeError("config must be a ConstitutionalCriticConfig or dict")

    # Create and return critic
    return ConstitutionalCritic(
        config=config,
        llm_provider=llm_provider,
    )


"""
@misc{bai2022constitutionalaiharmlessnessai,
      title={Constitutional AI: Harmlessness from AI Feedback},
      author={Yuntao Bai and Saurav Kadavath and Sandipan Kundu and Amanda Askell and Jackson Kernion and Andy Jones and Anna Chen and Anna Goldie and Azalia Mirhoseini and Cameron McKinnon and Carol Chen and Catherine Olsson and Christopher Olah and Danny Hernandez and Dawn Drain and Deep Ganguli and Dustin Li and Eli Tran-Johnson and Ethan Perez and Jamie Kerr and Jared Mueller and Jeffrey Ladish and Joshua Landau and Kamal Ndousse and Kamile Lukosuite and Liane Lovitt and Michael Sellitto and Nelson Elhage and Nicholas Schiefer and Noemi Mercado and Nova DasSarma and Robert Lasenby and Robin Larson and Sam Ringer and Scott Johnston and Shauna Kravec and Sheer El Showk and Stanislav Fort and Tamera Lanham and Timothy Telleen-Lawton and Tom Conerly and Tom Henighan and Tristan Hume and Samuel R. Bowman and Zac Hatfield-Dodds and Ben Mann and Dario Amodei and Nicholas Joseph and Sam McCandlish and Tom Brown and Jared Kaplan},
      year={2022},
      eprint={2212.08073},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2212.08073},
}
"""
