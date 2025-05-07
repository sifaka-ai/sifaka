"""
Self-Refine critic module for Sifaka.

This module implements the Self-Refine approach for critics, which enables language models
to iteratively critique and revise their own outputs without requiring external feedback.
The critic uses the same language model to generate critiques and revisions in multiple rounds.

Based on Self-Refine: https://arxiv.org/abs/2303.17651

Example:
    ```python
    from sifaka.critics.self_refine import create_self_refine_critic
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

from .base import BaseCritic, TextCritic, TextImprover, TextValidator, create_critic_state
from .models import PromptCriticConfig


class SelfRefineCriticConfig(PromptCriticConfig):
    """
    Configuration for self-refine critics.

    This model extends PromptCriticConfig with self-refine-specific settings
    for critics that iteratively critique and revise their own outputs.

    ## Lifecycle Management

    1. **Initialization**
       - Set base configuration
       - Configure self-refine settings
       - Validate field values
       - Create immutable instance

    2. **Validation**
       - Check field types
       - Verify value ranges
       - Ensure required fields
       - Validate custom rules

    3. **Usage**
       - Access configuration values
       - Create modified instances
       - Serialize to/from JSON
       - Validate against schema

    Examples:
        ```python
        from sifaka.critics.self_refine import SelfRefineCriticConfig

        # Create a self-refine critic config
        config = SelfRefineCriticConfig(
            name="self_refine_critic",
            description="A self-refine critic",
            max_iterations=3,
            system_prompt="You are an expert at critiquing and revising content.",
            temperature=0.7,
            max_tokens=1000
        )

        # Access configuration values
        print(f"Max iterations: {config.max_iterations}")
        print(f"System prompt: {config.system_prompt}")

        # Create modified config
        new_config = config.model_copy(
            update={"max_iterations": 5}
        )
        ```
    """

    max_iterations: int = Field(
        default=3, description="Maximum number of refinement iterations", gt=0
    )
    critique_prompt_template: str = Field(
        default=(
            "Critique the following response and suggest improvements:\n\n"
            "Task:\n{task}\n\n"
            "Response:\n{response}\n\n"
            "Critique:"
        ),
        description="Template for critique prompts",
    )
    revision_prompt_template: str = Field(
        default=(
            "Revise the original response using the critique:\n\n"
            "Task:\n{task}\n\n"
            "Original Response:\n{response}\n\n"
            "Critique:\n{critique}\n\n"
            "Revised Response:"
        ),
        description="Template for revision prompts",
    )


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
       - Clears state
       - Logs results
       - Handles errors

    ## Error Handling

    The critic handles various error conditions:
    - Empty or invalid input text
    - Missing task information
    - Model generation failures
    - Iteration limits
    - No improvement detection

    ## Examples

    Basic usage with factory function:

    ```python
    from sifaka.critics import create_self_refine_critic
    from sifaka.models.openai import create_openai_provider

    # Create a language model provider
    provider = create_openai_provider(
        model_name="gpt-4",
        api_key="your-api-key",
        temperature=0.7,
        max_tokens=1000
    )

    # Create a self-refine critic
    critic = create_self_refine_critic(
        llm_provider=provider,
        name="example_critic",
        description="A critic for improving explanations",
        max_iterations=3,
        system_prompt="You are an expert at explaining complex concepts clearly and concisely."
    )

    # Define a task and initial output
    task = "Explain quantum computing to a high school student."
    initial_output = "Quantum computing uses qubits instead of regular bits. Qubits can be in multiple states at once due to superposition."

    # Improve the output using the critic
    improved_output = critic.improve(initial_output, {"task": task})
    print(f"Improved output: {improved_output}")

    # Get critique of the improved output
    critique = critic.critique(improved_output, {"task": task})
    print(f"Score: {critique['score']}")
    print(f"Feedback: {critique['feedback']}")
    print(f"Issues: {critique['issues']}")
    print(f"Suggestions: {critique['suggestions']}")
    ```

    Advanced usage with custom configuration:

    ```python
    from sifaka.critics.self_refine import SelfRefineCritic, SelfRefineCriticConfig
    from sifaka.models.anthropic import create_anthropic_provider

    # Create a language model provider
    provider = create_anthropic_provider(
        model_name="claude-3-opus-20240229",
        api_key="your-api-key"
    )

    # Create a custom configuration with specialized prompts
    config = SelfRefineCriticConfig(
        name="technical_editor",
        description="A critic for improving technical documentation",
        max_iterations=5,
        system_prompt="You are an expert technical editor who specializes in clear, concise explanations.",
        temperature=0.5,
        max_tokens=2000,
        critique_prompt_template=(
            "You are reviewing technical documentation. Critique the following response:\n\n"
            "Task:\n{task}\n\n"
            "Response:\n{response}\n\n"
            "Provide specific issues and suggestions for improvement:"
        ),
        revision_prompt_template=(
            "Revise the technical documentation based on the critique:\n\n"
            "Task:\n{task}\n\n"
            "Original Response:\n{response}\n\n"
            "Critique:\n{critique}\n\n"
            "Provide an improved version that addresses all issues:"
        )
    )

    # Create a self-refine critic with custom configuration
    critic = SelfRefineCritic(
        config=config,
        llm_provider=provider
    )

    # Use the critic with specific feedback
    text = "The API uses REST principles and returns JSON responses. Endpoints require authentication."
    feedback = "The documentation lacks specific examples, authentication details, and error handling information."
    improved_text = critic.improve_with_feedback(text, feedback)
    print(f"Improved text: {improved_text}")

    # Use the critic asynchronously
    import asyncio

    async def improve_documentation():
        task = "Document the authentication process for the API."
        initial_text = "Users need to authenticate with the API using a token."
        improved_text = await critic.aimprove(initial_text, {"task": task})
        return improved_text

    improved_documentation = asyncio.run(improve_documentation())
    ```
    """

    # Class constants
    DEFAULT_NAME = "self_refine_critic"
    DEFAULT_DESCRIPTION = "Improves text through iterative self-critique and revision"

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using state manager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        config: SelfRefineCriticConfig,
        llm_provider: Any,
    ) -> None:
        """
        Initialize the self-refine critic.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider to use for critiquing and revising

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        # Initialize base class
        super().__init__(config)

        # Get state from state manager
        state = self._state_manager.get_state()

        # Store components in state
        state.model = llm_provider
        state.cache = {
            "max_iterations": config.max_iterations,
            "critique_prompt_template": config.critique_prompt_template,
            "revision_prompt_template": config.revision_prompt_template,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        state.initialized = True

    @property
    def config(self) -> SelfRefineCriticConfig:
        """Get the self-refine critic configuration."""
        return cast(SelfRefineCriticConfig, self._config)

    def _get_task_from_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Extract task from metadata.

        Args:
            metadata: Optional metadata containing the task

        Returns:
            The task as a string

        Raises:
            ValueError: If metadata is missing or does not contain a task
        """
        if not metadata:
            return "Improve the following text."

        task = metadata.get("task", "")
        if not task:
            task = "Improve the following text."

        return task

    def validate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate text by checking if it needs improvement.

        Args:
            text: The text to validate
            metadata: Optional metadata containing the task

        Returns:
            True if the text is valid, False otherwise

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Get state from state manager
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("SelfRefineCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Create critique prompt
        prompt = state.cache.get("critique_prompt_template", "").format(
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
        # Get state from state manager
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("SelfRefineCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Create critique prompt
        prompt = state.cache.get("critique_prompt_template", "").format(
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
        # Get state from state manager
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("SelfRefineCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Get max iterations from state
        max_iterations = state.cache.get("max_iterations", 3)

        # Start with the initial text
        current_output = text

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

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on specific feedback.

        Args:
            text: The text to improve
            feedback: The feedback to use for improvement

        Returns:
            Improved text

        Raises:
            ValueError: If text or feedback is empty
            RuntimeError: If critic is not properly initialized
        """
        # Get state from state manager
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("SelfRefineCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("feedback must be a non-empty string")

        # Create revision prompt with the provided feedback
        revision_prompt = state.cache.get("revision_prompt_template", "").format(
            task="Improve the following text",
            response=text,
            critique=feedback,
        )

        # Generate improved response
        improved_text = state.model.generate(
            revision_prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
        ).strip()

        return improved_text

    async def avalidate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Asynchronously validate text by checking if it needs improvement.

        Args:
            text: The text to validate
            metadata: Optional metadata containing the task

        Returns:
            True if the text is valid, False otherwise

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Get state from state manager
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("SelfRefineCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Create critique prompt
        prompt = state.cache.get("critique_prompt_template", "").format(
            task=task,
            response=text,
        )

        # Generate critique
        critique_text = await state.model.agenerate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
        )
        critique_text = critique_text.strip()

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

    async def acritique(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously analyze text and provide detailed feedback.

        Args:
            text: The text to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Get state from state manager
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("SelfRefineCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Create critique prompt
        prompt = state.cache.get("critique_prompt_template", "").format(
            task=task,
            response=text,
        )

        # Generate critique
        critique_text = await state.model.agenerate(
            prompt,
            system_prompt=state.cache.get("system_prompt", ""),
            temperature=state.cache.get("temperature", 0.7),
            max_tokens=state.cache.get("max_tokens", 1000),
        )
        critique_text = critique_text.strip()

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

    async def aimprove(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Asynchronously improve text through iterative self-critique and revision.

        Args:
            text: The text to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Get state from state manager
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("SelfRefineCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Get max iterations from state
        max_iterations = state.cache.get("max_iterations", 3)

        # Start with the initial text
        current_output = text

        # Perform iterative refinement
        for _ in range(max_iterations):
            # Step 1: Critique the current output
            critique_prompt = state.cache.get("critique_prompt_template", "").format(
                task=task,
                response=current_output,
            )

            critique = await state.model.agenerate(
                critique_prompt,
                system_prompt=state.cache.get("system_prompt", ""),
                temperature=state.cache.get("temperature", 0.7),
                max_tokens=state.cache.get("max_tokens", 1000),
            )
            critique = critique.strip()

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

            revised_output = await state.model.agenerate(
                revision_prompt,
                system_prompt=state.cache.get("system_prompt", ""),
                temperature=state.cache.get("temperature", 0.7),
                max_tokens=state.cache.get("max_tokens", 1000),
            )
            revised_output = revised_output.strip()

            # Check if there's no improvement
            if revised_output == current_output:
                return current_output

            # Update current output
            current_output = revised_output

        return current_output


def create_self_refine_critic(
    llm_provider: Any,
    name: str = "self_refine_critic",
    description: str = "Improves text through iterative self-critique and revision",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = "You are an expert at critiquing and revising content.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    max_iterations: int = 3,
    critique_prompt_template: Optional[str] = None,
    revision_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], SelfRefineCriticConfig]] = None,
    **kwargs: Any,
) -> SelfRefineCritic:
    """
    Create a self-refine critic with the given parameters.

    This function creates a self-refine critic that iteratively critiques and
    revises text using the same language model.

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
        **kwargs: Additional keyword arguments for the critic configuration

    Returns:
        A configured self-refine critic

    Raises:
        ValueError: If max_iterations is less than 1
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
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_iterations": max_iterations,
            **kwargs,  # Include additional kwargs in the config
        }

        if critique_prompt_template:
            config_dict["critique_prompt_template"] = critique_prompt_template

        if revision_prompt_template:
            config_dict["revision_prompt_template"] = revision_prompt_template

        config = SelfRefineCriticConfig(**config_dict)
    elif isinstance(config, dict):
        # Ensure max_iterations is included in the config
        if "max_iterations" not in config and max_iterations:
            config["max_iterations"] = max_iterations

        # Include additional kwargs
        if kwargs:
            config.update(kwargs)

        config = SelfRefineCriticConfig(**config)
    elif isinstance(config, SelfRefineCriticConfig) and kwargs:
        # If config is provided but there are also kwargs, create a new config with both
        config_dict = config.model_dump()
        config_dict.update(kwargs)
        config = SelfRefineCriticConfig(**config_dict)
    elif not isinstance(config, SelfRefineCriticConfig):
        raise TypeError("config must be a SelfRefineCriticConfig or dict")

    # Create and return critic
    return SelfRefineCritic(
        config=config,
        llm_provider=llm_provider,
    )


"""
@misc{selfrefine2023,
      title={Self-Refine: Iterative Refinement with Self-Feedback},
      author={Aman Madaan and Niket Tandon and Prakhar Gupta and Skyler Hallinan and Luyu Gao and Sarah Wiegreffe and Uri Alon and Nouha Dziri and Shrimai Prabhumoye and Yiming Yang and Sean Welleck and Bodhisattwa Prasad Majumder and Shashank Gupta and Amir Yazdanbakhsh and Peter Clark},
      year={2023},
      eprint={2303.17651},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""
