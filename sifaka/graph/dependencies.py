"""SifakaDependencies: Dependency injection system for Sifaka.

This module implements the dependency injection container that provides
all necessary components to the graph nodes during execution.

Key features:
- PydanticAI agent configuration
- Validator and critic management
- Tool creation for retrievers
- Default configuration factory
- Context manager support for cleanup
"""

from typing import Any, Dict, List, Union

from pydantic_ai import Agent

from sifaka.critics import (
    ConstitutionalCritic,
    MetaEvaluationCritic,
    NCriticsCritic,
    PromptCritic,
    ReflexionCritic,
    SelfConsistencyCritic,
    SelfRAGCritic,
    SelfRefineCritic,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class SifakaDependencies:
    """Dependencies injected into graph nodes.

    This container holds all the components needed by the Sifaka workflow:
    - Generator agent for text generation
    - Critics for iterative improvement
    - Validators for content validation
    - Retrievers for context gathering

    The dependencies are designed to be immutable during graph execution
    to ensure consistent behavior across nodes.

    The constructor accepts both model names (strings) and instances for flexible usage:

    Examples:
        # Using model names (strings) - auto-creates instances
        deps = SifakaDependencies(
            generator="openai:gpt-4",
            critics={"reflexion": "openai:gpt-3.5-turbo"},
            validators=[validator]
        )

        # Using instances - uses them directly
        deps = SifakaDependencies(
            generator=my_agent,
            critics={"reflexion": my_critic},
            validators=[validator]
        )

        # Mixed usage is also supported
        deps = SifakaDependencies(
            generator="openai:gpt-4",  # String
            critics={"reflexion": my_critic},  # Instance
            validators=[validator]
        )
    """

    def __init__(
        self,
        generator: Union[str, Agent] = None,
        critics: Dict[str, Union[str, Any]] = None,
        validators: List[Any] = None,
        retrievers: Dict[str, Any] = None,
        always_apply_critics: bool = False,
        never_apply_critics: bool = False,
        always_include_validation_results: bool = True,
        validation_weight: float = 0.6,
        critic_weight: float = 0.4,
        # Memory management options
        auto_optimize_memory: bool = False,
        memory_optimization_interval: int = 5,
        keep_last_n_iterations: int = 3,
        max_messages_per_iteration: int = 10,
        max_tool_result_size_bytes: int = 10240,
    ):
        """Initialize SifakaDependencies with flexible parameter types.

        Args:
            generator: Generator agent (Agent instance) or model name (str)
            critics: Dict mapping critic names to critic instances or model names
            validators: List of validator instances
            retrievers: Dict of retriever instances
            always_apply_critics: If True, critics always run regardless of validation results
            never_apply_critics: If True, critics never run (overrides always_apply_critics)
            always_include_validation_results: If True, validation results always included in context
            validation_weight: Weight for validation feedback in context (0.0-1.0, default 0.6)
            critic_weight: Weight for critic feedback in context (0.0-1.0, default 0.4)
            auto_optimize_memory: If True, automatically optimize memory during processing
            memory_optimization_interval: Optimize memory every N iterations
            keep_last_n_iterations: Number of iterations to keep in history
            max_messages_per_iteration: Maximum conversation messages per iteration
            max_tool_result_size_bytes: Maximum size for tool results
        """

        # Set defaults
        if generator is None:
            generator = "openai:gpt-4"
        if critics is None:
            critics = {}
        if validators is None:
            validators = []
        if retrievers is None:
            retrievers = {}

        # Validate weights
        if validation_weight < 0 or validation_weight > 1:
            raise ValueError(f"validation_weight must be between 0 and 1, got {validation_weight}")
        if critic_weight < 0 or critic_weight > 1:
            raise ValueError(f"critic_weight must be between 0 and 1, got {critic_weight}")

        # Warn if weights don't sum to 1.0 (but allow it for flexibility)
        total_weight = validation_weight + critic_weight
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point differences
            import warnings

            warnings.warn(
                f"Validation weight ({validation_weight}) + Critic weight ({critic_weight}) = {total_weight:.2f}. "
                f"Consider adjusting weights to sum to 1.0 for optimal balance.",
                UserWarning,
            )

        # Convert generator to Agent instance if needed
        if isinstance(generator, str):
            self.generator_agent = Agent(
                generator,
                system_prompt=(
                    "Generate high-quality content using available tools when needed. "
                    "Focus on accuracy, clarity, and helpfulness."
                ),
            )
        else:
            self.generator_agent = generator

        # Convert critics to instances if needed
        self.critics = {}
        for name, critic in critics.items():
            if isinstance(critic, str):
                self.critics[name] = self._create_critic_instance(name, critic)
            else:
                self.critics[name] = critic

        # Store other attributes
        self.validators = validators
        self.retrievers = retrievers
        self.always_apply_critics = always_apply_critics
        self.never_apply_critics = never_apply_critics
        self.always_include_validation_results = always_include_validation_results
        self.validation_weight = validation_weight
        self.critic_weight = critic_weight

        # Memory management configuration
        self.auto_optimize_memory = auto_optimize_memory
        self.memory_optimization_interval = memory_optimization_interval
        self.keep_last_n_iterations = keep_last_n_iterations
        self.max_messages_per_iteration = max_messages_per_iteration
        self.max_tool_result_size_bytes = max_tool_result_size_bytes

        logger.info(
            "SifakaDependencies initialized",
            extra={
                "generator_model": (
                    str(self.generator_agent.model)
                    if hasattr(self.generator_agent, "model")
                    else "unknown"
                ),
                "critic_count": len(self.critics),
                "critic_names": list(self.critics.keys()),
                "validator_count": len(self.validators),
                "always_apply_critics": self.always_apply_critics,
                "validation_weight": self.validation_weight,
                "critic_weight": self.critic_weight,
                "memory_management": {
                    "auto_optimize": self.auto_optimize_memory,
                    "optimization_interval": self.memory_optimization_interval,
                    "keep_iterations": self.keep_last_n_iterations,
                    "max_messages": self.max_messages_per_iteration,
                    "max_tool_size": self.max_tool_result_size_bytes,
                },
            },
        )

    def _create_critic_instance(self, name: str, model_name: str) -> Any:
        """Create a critic instance from a model name.

        Args:
            name: Name of the critic type
            model_name: Model name to use for the critic

        Returns:
            Critic instance
        """
        critic_classes = {
            "reflexion": ReflexionCritic,
            "constitutional": ConstitutionalCritic,
            "self_refine": SelfRefineCritic,
            "n_critics": NCriticsCritic,
            "self_consistency": SelfConsistencyCritic,
            "prompt": PromptCritic,
            "meta_rewarding": MetaEvaluationCritic,
            "self_rag": SelfRAGCritic,
        }

        if name in critic_classes:
            return critic_classes[name](model_name=model_name)
        else:
            # Fallback to PromptCritic for unknown critics
            logger.warning(f"Unknown critic type '{name}', falling back to PromptCritic")
            return PromptCritic(model_name=model_name)

    @classmethod
    def create_default(cls) -> "SifakaDependencies":
        """Create default dependency configuration.

        This factory method creates a basic configuration suitable for
        most use cases. It can be customized by creating dependencies
        manually or by modifying the returned instance.

        Returns:
            SifakaDependencies with default configuration
        """
        logger.info("Creating default SifakaDependencies configuration")

        # Research-based critics using small, fast models from different providers
        critics = {
            # Reflexion critic - OpenAI small model
            "reflexion": "openai:gpt-4o-mini",
            # Constitutional AI critic - Anthropic small model
            "constitutional": "anthropic:claude-3-5-haiku-20241022",
            # Self-Refine critic - Gemini Flash
            "self_refine": "gemini-1.5-flash",
            # N-Critics ensemble - Groq fast model
            "n_critics": "groq:llama-3.1-8b-instant",
            # Self-Consistency critic - OpenAI alternative model
            "self_consistency": "openai:gpt-3.5-turbo",
            # Prompt-based customizable critic - Anthropic
            "prompt": "anthropic:claude-3-haiku-20240307",
            # Meta-Evaluation critic - Gemini
            "meta_rewarding": "gemini-1.5-flash",
            # Self-RAG critic - Groq
            "self_rag": "groq:mixtral-8x7b-32768",
        }

        # Basic validators for common use cases
        from sifaka.validators import (
            min_length_validator,
            max_length_validator,
            sentiment_validator,
        )

        validators = [
            # Length validation - ensure reasonable text length
            min_length_validator(min_length=10, unit="characters", strict=False),
            max_length_validator(max_length=5000, unit="characters", strict=False),
            # Sentiment validation - avoid negative sentiment by default
            sentiment_validator(
                forbidden_sentiments=["negative"],
                min_confidence=0.7,
                cached=True,
            ),
        ]

        logger.info(
            "Default dependencies created",
            extra={
                "generator_model": "openai:gpt-4",
                "critic_count": len(critics),
                "critic_names": list(critics.keys()),
                "validator_count": len(validators),
            },
        )

        return cls(generator="openai:gpt-4", critics=critics, validators=validators, retrievers={})

    @staticmethod
    def _get_critic_system_prompt(critic_name: str) -> str:
        """Get system prompt for a specific critic.

        Args:
            critic_name: Name of the critic

        Returns:
            System prompt string for the critic
        """
        prompts = {
            "reflexion": (
                "You are a Reflexion critic implementing Shinn et al. 2023. "
                "Provide specific feedback for iterative improvement. "
                "Focus on reasoning errors, missing information, and clarity issues."
            ),
            "constitutional": (
                "You are a Constitutional AI critic. "
                "Evaluate content for helpfulness, harmlessness, and honesty. "
                "Provide specific improvement suggestions."
            ),
            "self_refine": (
                "You are a Self-Refine critic implementing Madaan et al. 2023. "
                "Suggest iterative improvements through self-critique."
            ),
            "n_critics": (
                "You are part of an N-Critics ensemble implementing the approach from "
                "https://arxiv.org/abs/2310.18679. Provide specialized critique from "
                "multiple perspectives: clarity, accuracy, completeness, and style."
            ),
            "self_consistency": (
                "You are a Self-Consistency critic. Generate multiple critiques "
                "and use consensus to determine reliable feedback."
            ),
            "prompt": (
                "You are a customizable prompt-based critic. Evaluate content "
                "based on the specific criteria provided in the prompt."
            ),
            "meta_rewarding": (
                "You are a Meta-Evaluation critic that evaluates and improves "
                "critique quality. Focus on the quality of feedback and suggestions."
            ),
            "self_rag": (
                "You are a Self-RAG critic implementing retrieval-augmented "
                "generation critique. Evaluate whether content would benefit "
                "from additional retrieval and provide specific guidance."
            ),
        }

        return prompts.get(critic_name, "You are a helpful critic. Provide constructive feedback.")

    def __enter__(self):
        """Enter context manager - for future resource management."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - for future cleanup."""
        # Parameters are required by context manager protocol but not used
        pass
