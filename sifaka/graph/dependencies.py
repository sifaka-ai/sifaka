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

from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic_ai import Agent

from sifaka.critics import (
    ConstitutionalCritic,
    MetaRewardingCritic,
    NCriticsCritic,
    PromptCritic,
    ReflexionCritic,
    SelfConsistencyCritic,
    SelfRAGCritic,
    SelfRefineCritic,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SifakaDependencies:
    """Dependencies injected into graph nodes.

    This container holds all the components needed by the Sifaka workflow:
    - Generator agent for text generation
    - Critics for iterative improvement
    - Validators for content validation
    - Retrievers for context gathering

    The dependencies are designed to be immutable during graph execution
    to ensure consistent behavior across nodes.
    """

    generator_agent: Agent
    critics: Dict[str, Any]  # BaseCritic instances
    validators: List[Any]
    retrievers: Dict[str, Any]

    # Configuration options
    always_apply_critics: bool = (
        False  # If True, critics always run regardless of validation results
    )
    never_apply_critics: bool = False  # If True, critics never run
    always_include_validation_results: bool = (
        True  # If True, validation results always included in context
    )

    # Feedback weighting (should sum to 1.0)
    validation_weight: float = 0.6  # Weight for validation feedback (60% by default)
    critic_weight: float = 0.4  # Weight for critic feedback (40% by default)

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

        # Main generator agent
        generator = Agent(
            "openai:gpt-4",
            system_prompt=(
                "Generate high-quality content. " "Focus on accuracy, clarity, and helpfulness."
            ),
        )

        # Research-based critics using small, fast models from different providers
        critics = {
            # Reflexion critic - OpenAI small model
            "reflexion": ReflexionCritic(model_name="openai:gpt-4o-mini"),
            # Constitutional AI critic - Anthropic small model
            "constitutional": ConstitutionalCritic(
                model_name="anthropic:claude-3-5-haiku-20241022"
            ),
            # Self-Refine critic - Gemini Flash
            "self_refine": SelfRefineCritic(model_name="gemini-1.5-flash"),
            # N-Critics ensemble - Groq fast model
            "n_critics": NCriticsCritic(model_name="groq:llama-3.1-8b-instant"),
            # Self-Consistency critic - OpenAI alternative model
            "self_consistency": SelfConsistencyCritic(model_name="openai:gpt-3.5-turbo"),
            # Prompt-based customizable critic - Anthropic
            "prompt": PromptCritic(model_name="anthropic:claude-3-haiku-20240307"),
            # Meta-Rewarding critic - Gemini
            "meta_rewarding": MetaRewardingCritic(model_name="gemini-1.5-flash"),
            # Self-RAG critic - Groq
            "self_rag": SelfRAGCritic(model_name="groq:mixtral-8x7b-32768"),
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

        return cls(generator_agent=generator, critics=critics, validators=validators, retrievers={})

    @classmethod
    def create_custom(
        cls,
        generator_model: str = "openai:gpt-4",
        critic_models: Dict[str, str] = None,
        validators: List[Any] = None,
        retrievers: Dict[str, Any] = None,
        always_apply_critics: bool = False,
        never_apply_critics: bool = False,
        always_include_validation_results: bool = True,
        validation_weight: float = 0.6,
        critic_weight: float = 0.4,
    ) -> "SifakaDependencies":
        """Create custom dependency configuration.

        Args:
            generator_model: Model name for the generator agent
            critic_models: Dict mapping critic names to model names
            validators: List of validator instances
            retrievers: Dict of retriever instances
            always_apply_critics: If True, critics always run regardless of validation results
            never_apply_critics: If True, critics never run (overrides always_apply_critics)
            always_include_validation_results: If True, validation results always included in context
            validation_weight: Weight for validation feedback in context (0.0-1.0, default 0.6)
            critic_weight: Weight for critic feedback in context (0.0-1.0, default 0.4)

        Returns:
            SifakaDependencies with custom configuration
        """
        if critic_models is None:
            critic_models = {
                "reflexion": "openai:gpt-4o-mini",
                "constitutional": "anthropic:claude-3-5-haiku-20241022",
                "self_refine": "gemini-1.5-flash",
                "n_critics": "groq:llama-3.1-8b-instant",
                "self_consistency": "openai:gpt-3.5-turbo",
                "prompt": "anthropic:claude-3-haiku-20240307",
                "meta_rewarding": "gemini-1.5-flash",
                "self_rag": "groq:mixtral-8x7b-32768",
            }

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

        # Create generator agent
        generator = Agent(
            generator_model,
            system_prompt=(
                "Generate high-quality content using available tools when needed. "
                "Focus on accuracy, clarity, and helpfulness."
            ),
        )

        # Create critic instances
        critics = {}
        critic_classes = {
            "reflexion": ReflexionCritic,
            "constitutional": ConstitutionalCritic,
            "self_refine": SelfRefineCritic,
            "n_critics": NCriticsCritic,
            "self_consistency": SelfConsistencyCritic,
            "prompt": PromptCritic,
            "meta_rewarding": MetaRewardingCritic,
            "self_rag": SelfRAGCritic,
        }

        for name, model in critic_models.items():
            if name in critic_classes:
                critics[name] = critic_classes[name](model_name=model)
            else:
                # Fallback to PromptCritic for unknown critics
                critics[name] = PromptCritic(model_name=model)

        return cls(
            generator_agent=generator,
            critics=critics,
            validators=validators,
            retrievers=retrievers,
            always_apply_critics=always_apply_critics,
            never_apply_critics=never_apply_critics,
            always_include_validation_results=always_include_validation_results,
            validation_weight=validation_weight,
            critic_weight=critic_weight,
        )

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
                "You are a Meta-Rewarding critic that evaluates and improves "
                "reward models. Focus on the quality of feedback and suggestions."
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
        pass
