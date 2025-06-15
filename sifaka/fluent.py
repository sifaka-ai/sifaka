"""Fluent API for Sifaka - Simple builder pattern for common use cases.

This module provides a fluent, chainable API that simplifies the most common
Sifaka usage patterns. Instead of manually creating dependencies and engines,
users can chain method calls to build their configuration.

Example:
    ```python
    from sifaka import Sifaka

    result = await (Sifaka("Explain AI")
                    .min_length(100)
                    .positive_sentiment()
                    .with_reflexion()
                    .with_constitutional()
                    .improve())
    ```

This is equivalent to the more verbose traditional approach:
    ```python
    from sifaka import SifakaEngine
    from sifaka.graph.dependencies import SifakaDependencies
    from sifaka.critics import ReflexionCritic, ConstitutionalCritic
    from sifaka.validators import min_length_validator, sentiment_validator

    deps = SifakaDependencies(
        critics={"reflexion": ReflexionCritic(), "constitutional": ConstitutionalCritic()},
        validators=[min_length_validator(100), sentiment_validator(required="positive")]
    )
    engine = SifakaEngine(dependencies=deps)
    result = await engine.think("Explain AI")
    ```
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from sifaka.core.engine import SifakaEngine
    from sifaka.core.thought import SifakaThought

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class Sifaka:
    """Fluent API builder for Sifaka workflows.

    This class provides a chainable interface for building Sifaka configurations
    and executing workflows. Each method returns self to enable method chaining.

    The builder accumulates configuration options and creates the appropriate
    SifakaDependencies and SifakaEngine when .improve() is called.
    """

    def __init__(self, prompt: str):
        """Initialize the fluent builder with a prompt.

        Args:
            prompt: The initial prompt for text generation
        """
        self.prompt = prompt
        self.validators: List[Any] = []
        self.critics: Dict[str, Any] = {}
        self.retrievers: Dict[str, Any] = {}
        self.generator_config: Optional[Union[str, Any]] = None  # Any instead of Agent for runtime
        self.max_iterations_config: int = 3
        self.validation_weight_config: float = 0.6
        self.critic_weight_config: float = 0.4
        self.always_apply_critics_config: bool = False
        self.never_apply_critics_config: bool = False

        # Memory management configuration
        self.auto_optimize_memory_config: bool = False
        self.memory_optimization_interval_config: int = 5
        self.keep_last_n_iterations_config: int = 3
        self.max_messages_per_iteration_config: int = 10
        self.max_tool_result_size_bytes_config: int = 10240

        logger.debug(f"Created Sifaka fluent builder with prompt: {prompt[:50]}...")

    # Validator methods
    def min_length(self, length: int, unit: str = "characters", strict: bool = True) -> "Sifaka":
        """Add minimum length validation.

        Args:
            length: Minimum required length
            unit: Unit of measurement ("characters" or "words")
            strict: Whether to fail validation on any violation

        Returns:
            Self for method chaining
        """
        from sifaka.validators import min_length_validator

        validator = min_length_validator(length, unit=unit, strict=strict)
        self.validators.append(validator)
        logger.debug(f"Added min_length validator: {length} {unit}")
        return self

    def max_length(self, length: int, unit: str = "characters", strict: bool = True) -> "Sifaka":
        """Add maximum length validation.

        Args:
            length: Maximum allowed length
            unit: Unit of measurement ("characters" or "words")
            strict: Whether to fail validation on any violation

        Returns:
            Self for method chaining
        """
        from sifaka.validators import max_length_validator

        validator = max_length_validator(length, unit=unit, strict=strict)
        self.validators.append(validator)
        logger.debug(f"Added max_length validator: {length} {unit}")
        return self

    def length(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        unit: str = "characters",
        strict: bool = True,
    ) -> "Sifaka":
        """Add length validation with both min and max constraints.

        Args:
            min_length: Minimum required length (None for no minimum)
            max_length: Maximum allowed length (None for no maximum)
            unit: Unit of measurement ("characters" or "words")
            strict: Whether to fail validation on any violation

        Returns:
            Self for method chaining
        """
        from sifaka.validators import create_length_validator

        validator = create_length_validator(
            min_length=min_length, max_length=max_length, unit=unit, strict=strict
        )
        self.validators.append(validator)
        logger.debug(f"Added length validator: {min_length}-{max_length} {unit}")
        return self

    def positive_sentiment(self, min_confidence: float = 0.6) -> "Sifaka":
        """Add positive sentiment validation.

        Args:
            min_confidence: Minimum confidence threshold for sentiment classification

        Returns:
            Self for method chaining
        """
        from sifaka.validators import sentiment_validator

        validator = sentiment_validator(
            required_sentiment="positive", min_confidence=min_confidence, cached=True
        )
        self.validators.append(validator)
        logger.debug(f"Added positive sentiment validator with confidence {min_confidence}")
        return self

    def negative_sentiment(self, min_confidence: float = 0.6) -> "Sifaka":
        """Add negative sentiment validation.

        Args:
            min_confidence: Minimum confidence threshold for sentiment classification

        Returns:
            Self for method chaining
        """
        from sifaka.validators import sentiment_validator

        validator = sentiment_validator(
            required_sentiment="negative", min_confidence=min_confidence, cached=True
        )
        self.validators.append(validator)
        logger.debug(f"Added negative sentiment validator with confidence {min_confidence}")
        return self

    def sentiment(
        self,
        required: Optional[str] = None,
        forbidden: Optional[List[str]] = None,
        min_confidence: float = 0.6,
    ) -> "Sifaka":
        """Add custom sentiment validation.

        Args:
            required: Required sentiment label
            forbidden: List of forbidden sentiment labels
            min_confidence: Minimum confidence threshold for sentiment classification

        Returns:
            Self for method chaining
        """
        from sifaka.validators import sentiment_validator

        validator = sentiment_validator(
            required_sentiment=required,
            forbidden_sentiments=forbidden,
            min_confidence=min_confidence,
            cached=True,
        )
        self.validators.append(validator)
        logger.debug(
            f"Added custom sentiment validator: required={required}, forbidden={forbidden}"
        )
        return self

    def required_content(self, patterns: List[str], case_sensitive: bool = False) -> "Sifaka":
        """Add required content validation.

        Args:
            patterns: List of required patterns/words
            case_sensitive: Whether matching is case-sensitive

        Returns:
            Self for method chaining
        """
        from sifaka.validators import required_content_validator

        validator = required_content_validator(patterns, case_sensitive=case_sensitive)
        self.validators.append(validator)
        logger.debug(f"Added required content validator: {patterns}")
        return self

    def prohibited_content(self, patterns: List[str], case_sensitive: bool = False) -> "Sifaka":
        """Add prohibited content validation.

        Args:
            patterns: List of prohibited patterns/words
            case_sensitive: Whether matching is case-sensitive

        Returns:
            Self for method chaining
        """
        from sifaka.validators import prohibited_content_validator

        validator = prohibited_content_validator(patterns, case_sensitive=case_sensitive)
        self.validators.append(validator)
        logger.debug(f"Added prohibited content validator: {patterns}")
        return self

    def json_format(self) -> "Sifaka":
        """Add JSON format validation.

        Returns:
            Self for method chaining
        """
        from sifaka.validators import json_validator

        validator = json_validator()
        self.validators.append(validator)
        logger.debug("Added JSON format validator")
        return self

    def markdown_format(self) -> "Sifaka":
        """Add Markdown format validation.

        Returns:
            Self for method chaining
        """
        from sifaka.validators import markdown_validator

        validator = markdown_validator()
        self.validators.append(validator)
        logger.debug("Added Markdown format validator")
        return self

    # Critic methods
    def with_reflexion(self, model_name: Optional[str] = None) -> "Sifaka":
        """Add Reflexion critic for iterative improvement.

        Args:
            model_name: Model to use for the critic (None for default)

        Returns:
            Self for method chaining
        """
        from sifaka.critics import ReflexionCritic

        if model_name:
            self.critics["reflexion"] = ReflexionCritic(model_name=model_name)
        else:
            self.critics["reflexion"] = "gemini-1.5-flash"  # Use string for default

        logger.debug(f"Added Reflexion critic with model: {model_name or 'default'}")
        return self

    def with_constitutional(self, model_name: Optional[str] = None) -> "Sifaka":
        """Add Constitutional AI critic for principle-based improvement.

        Args:
            model_name: Model to use for the critic (None for default)

        Returns:
            Self for method chaining
        """
        from sifaka.critics import ConstitutionalCritic

        if model_name:
            self.critics["constitutional"] = ConstitutionalCritic(model_name=model_name)
        else:
            self.critics["constitutional"] = "openai:gpt-3.5-turbo"  # Use string for default

        logger.debug(f"Added Constitutional critic with model: {model_name or 'default'}")
        return self

    def with_self_refine(self, model_name: Optional[str] = None) -> "Sifaka":
        """Add Self-Refine critic for iterative self-improvement.

        Args:
            model_name: Model to use for the critic (None for default)

        Returns:
            Self for method chaining
        """
        from sifaka.critics import SelfRefineCritic

        if model_name:
            self.critics["self_refine"] = SelfRefineCritic(model_name=model_name)
        else:
            self.critics["self_refine"] = "gemini-1.5-flash"  # Use string for default

        logger.debug(f"Added Self-Refine critic with model: {model_name or 'default'}")
        return self

    def with_n_critics(self, model_name: Optional[str] = None) -> "Sifaka":
        """Add N-Critics ensemble critic for multi-perspective improvement.

        Args:
            model_name: Model to use for the critic (None for default)

        Returns:
            Self for method chaining
        """
        from sifaka.critics import NCriticsCritic

        if model_name:
            self.critics["n_critics"] = NCriticsCritic(model_name=model_name)
        else:
            self.critics["n_critics"] = "gemini-1.5-flash"  # Use string for default

        logger.debug(f"Added N-Critics critic with model: {model_name or 'default'}")
        return self

    def with_self_rag(self, model_name: Optional[str] = None) -> "Sifaka":
        """Add Self-RAG critic for retrieval-augmented improvement.

        Args:
            model_name: Model to use for the critic (None for default)

        Returns:
            Self for method chaining
        """
        from sifaka.critics import SelfRAGCritic

        if model_name:
            self.critics["self_rag"] = SelfRAGCritic(model_name=model_name)
        else:
            self.critics["self_rag"] = "openai:gpt-4"  # Use string for default

        logger.debug(f"Added Self-RAG critic with model: {model_name or 'default'}")
        return self

    def with_meta_evaluation(self, model_name: Optional[str] = None) -> "Sifaka":
        """Add Meta-Evaluation critic for meta-judging approach.

        Args:
            model_name: Model to use for the critic (None for default)

        Returns:
            Self for method chaining
        """
        from sifaka.critics import MetaEvaluationCritic

        if model_name:
            self.critics["meta_evaluation"] = MetaEvaluationCritic(model_name=model_name)
        else:
            self.critics["meta_evaluation"] = "openai:gpt-4"  # Use string for default

        logger.debug(f"Added Meta-Evaluation critic with model: {model_name or 'default'}")
        return self

    def with_self_consistency(self, model_name: Optional[str] = None) -> "Sifaka":
        """Add Self-Consistency critic for consistency-based improvement.

        Args:
            model_name: Model to use for the critic (None for default)

        Returns:
            Self for method chaining
        """
        from sifaka.critics import SelfConsistencyCritic

        if model_name:
            self.critics["self_consistency"] = SelfConsistencyCritic(model_name=model_name)
        else:
            self.critics["self_consistency"] = "openai:gpt-3.5-turbo"  # Use string for default

        logger.debug(f"Added Self-Consistency critic with model: {model_name or 'default'}")
        return self

    def with_prompt_critic(self, prompt: str, model_name: Optional[str] = None) -> "Sifaka":
        """Add custom prompt-based critic.

        Args:
            prompt: Custom evaluation prompt for the critic
            model_name: Model to use for the critic (None for default)

        Returns:
            Self for method chaining
        """
        from sifaka.critics import PromptCritic

        if model_name:
            critic = PromptCritic(evaluation_prompt=prompt, model_name=model_name)
        else:
            critic = PromptCritic(evaluation_prompt=prompt)

        # Use a unique key for custom prompt critics
        key = f"prompt_critic_{len([k for k in self.critics.keys() if k.startswith('prompt_critic')])}"
        self.critics[key] = critic

        logger.debug(f"Added custom prompt critic with model: {model_name or 'default'}")
        return self

    # Configuration methods
    def generator(self, model_name_or_agent: Union[str, Any]) -> "Sifaka":
        """Set the generator model or agent.

        Args:
            model_name_or_agent: Model name string or PydanticAI Agent instance

        Returns:
            Self for method chaining
        """
        self.generator_config = model_name_or_agent
        logger.debug(f"Set generator: {model_name_or_agent}")
        return self

    def max_iterations(self, count: int) -> "Sifaka":
        """Set maximum number of improvement iterations.

        Args:
            count: Maximum number of iterations

        Returns:
            Self for method chaining
        """
        self.max_iterations_config = count
        logger.debug(f"Set max iterations: {count}")
        return self

    def validation_weight(self, weight: float) -> "Sifaka":
        """Set the weight for validation feedback in improvement decisions.

        Args:
            weight: Weight between 0.0 and 1.0 for validation feedback

        Returns:
            Self for method chaining
        """
        self.validation_weight_config = weight
        self.critic_weight_config = 1.0 - weight  # Ensure they sum to 1.0
        logger.debug(f"Set validation weight: {weight}, critic weight: {self.critic_weight_config}")
        return self

    def critic_weight(self, weight: float) -> "Sifaka":
        """Set the weight for critic feedback in improvement decisions.

        Args:
            weight: Weight between 0.0 and 1.0 for critic feedback

        Returns:
            Self for method chaining
        """
        self.critic_weight_config = weight
        self.validation_weight_config = 1.0 - weight  # Ensure they sum to 1.0
        logger.debug(
            f"Set critic weight: {weight}, validation weight: {self.validation_weight_config}"
        )
        return self

    def always_apply_critics(self, enabled: bool = True) -> "Sifaka":
        """Configure whether to always apply critics regardless of validation results.

        Args:
            enabled: Whether to always apply critics

        Returns:
            Self for method chaining
        """
        self.always_apply_critics_config = enabled
        if enabled:
            self.never_apply_critics_config = False
        logger.debug(f"Set always apply critics: {enabled}")
        return self

    def never_apply_critics(self, enabled: bool = True) -> "Sifaka":
        """Configure whether to never apply critics (validation only).

        Args:
            enabled: Whether to never apply critics

        Returns:
            Self for method chaining
        """
        self.never_apply_critics_config = enabled
        if enabled:
            self.always_apply_critics_config = False
        logger.debug(f"Set never apply critics: {enabled}")
        return self

    # Memory management methods
    def auto_optimize_memory(
        self, enabled: bool = True, optimization_interval: int = 5
    ) -> "Sifaka":
        """Configure automatic memory optimization during processing.

        Args:
            enabled: Whether to enable automatic memory optimization
            optimization_interval: Optimize memory every N iterations

        Returns:
            Self for method chaining
        """
        self.auto_optimize_memory_config = enabled
        self.memory_optimization_interval_config = optimization_interval
        logger.debug(f"Set auto memory optimization: {enabled}, interval: {optimization_interval}")
        return self

    def memory_settings(
        self,
        keep_last_n_iterations: int = 3,
        max_messages_per_iteration: int = 10,
        max_tool_result_size_bytes: int = 10240,
    ) -> "Sifaka":
        """Configure memory management settings.

        Args:
            keep_last_n_iterations: Number of iterations to keep in history
            max_messages_per_iteration: Maximum conversation messages per iteration
            max_tool_result_size_bytes: Maximum size for tool results

        Returns:
            Self for method chaining
        """
        self.keep_last_n_iterations_config = keep_last_n_iterations
        self.max_messages_per_iteration_config = max_messages_per_iteration
        self.max_tool_result_size_bytes_config = max_tool_result_size_bytes
        logger.debug(
            f"Set memory settings: keep={keep_last_n_iterations}, "
            f"max_messages={max_messages_per_iteration}, "
            f"max_tool_size={max_tool_result_size_bytes}"
        )
        return self

    # Execution methods
    def build(self) -> "SifakaEngine":
        """Build and return a SifakaEngine with the configured dependencies.

        This method is useful for advanced usage where you want to access
        the engine directly for multiple operations.

        Returns:
            Configured SifakaEngine instance
        """
        # Lazy import to avoid dependency issues
        from sifaka.core.engine import SifakaEngine
        from sifaka.utils.config import SifakaConfig

        # Extract validator configurations for simple config
        min_length = None
        max_length = None
        required_sentiment = None

        # Parse validators to extract simple configurations
        for validator in self.validators:
            if hasattr(validator, "min_length"):
                min_length = validator.min_length
            elif hasattr(validator, "max_length"):
                max_length = validator.max_length
            elif hasattr(validator, "required_sentiments") and validator.required_sentiments:
                required_sentiment = validator.required_sentiments[0]

        # Create config with accumulated configuration
        config = SifakaConfig(
            model=self.generator_config or "openai:gpt-4o-mini",
            max_iterations=self.max_iterations_config,
            min_length=min_length,
            max_length=max_length,
            required_sentiment=required_sentiment,
            critics=list(self.critics.keys()) if self.critics else ["reflexion"],
        )

        # Create engine with config
        engine = SifakaEngine(config=config)

        logger.info(
            "Built SifakaEngine from fluent API using SifakaConfig",
            extra={
                "model": config.model,
                "max_iterations": config.max_iterations,
                "critics": config.critics,
                "min_length": config.min_length,
                "max_length": config.max_length,
                "required_sentiment": config.required_sentiment,
            },
        )

        return engine

    async def improve(self) -> "SifakaThought":
        """Execute the workflow and return the improved thought.

        This is the main execution method that builds the engine and
        processes the prompt through the complete Sifaka workflow.

        Returns:
            SifakaThought with the final improved text and full audit trail
        """
        engine = self.build()

        logger.info(
            f"Starting fluent API workflow for prompt: {self.prompt[:50]}...",
            extra={
                "prompt_length": len(self.prompt),
                "max_iterations": self.max_iterations_config,
            },
        )

        result = await engine.think(self.prompt, max_iterations=self.max_iterations_config)

        logger.info(
            "Completed fluent API workflow",
            extra={
                "final_iteration": result.iteration,
                "validation_passed": result.validation_passed(),
                "final_text_length": len(result.final_text) if result.final_text else 0,
            },
        )

        return result
