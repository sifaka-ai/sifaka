"""Chain configuration management for Sifaka.

This module contains the ChainConfig class which manages all configuration
state for a Chain, including models, retrievers, validators, critics, and options.
"""

import uuid
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Critic, Model, Retriever, Validator
from sifaka.storage.checkpoints import CachedCheckpointStorage
from sifaka.storage.protocol import Storage


class ChainConfig:
    """Configuration state manager for Chain.

    This class encapsulates all the configuration state for a Chain,
    providing validation and management of the chain's components.

    Attributes:
        model: The language model to use for text generation.
        prompt: The prompt to use for text generation.
        model_retrievers: Retrievers for model context (pre-generation).
        critic_retrievers: Retrievers for critic context (post-generation).
        validators: List of validators to check if the generated text meets requirements.
        critics: List of critics to improve the generated text.
        storage: Storage backend for thoughts.
        checkpoint_storage: Storage backend for checkpoints.
        options: Dictionary of chain execution options.
        chain_id: Unique identifier for this chain configuration.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        prompt: Optional[str] = None,
        model_retrievers: Optional[List[Retriever]] = None,
        critic_retrievers: Optional[List[Retriever]] = None,
        storage: Optional[Storage] = None,
        checkpoint_storage: Optional[CachedCheckpointStorage] = None,
        max_improvement_iterations: int = 3,
        apply_improvers_on_validation_failure: bool = False,
        always_apply_critics: bool = False,
    ):
        """Initialize the chain configuration.

        Args:
            model: Optional language model to use for text generation.
            prompt: Optional prompt to use for text generation.
            model_retrievers: Optional list of retrievers for model context.
            critic_retrievers: Optional list of retrievers for critic context.
            storage: Optional storage for saving intermediate thoughts.
            checkpoint_storage: Optional storage for chain execution checkpoints.
            max_improvement_iterations: Maximum number of improvement iterations.
            apply_improvers_on_validation_failure: Whether to apply improvers when validation fails.
            always_apply_critics: Whether to always apply critics regardless of validation status.
        """
        self.model = model
        self.prompt = prompt
        self.model_retrievers = model_retrievers or []
        self.critic_retrievers = critic_retrievers or []

        # Default to memory storage if none provided
        if storage is None:
            from sifaka.storage.memory import MemoryStorage

            self.storage: Storage = MemoryStorage()
        else:
            self.storage = storage

        self.checkpoint_storage = checkpoint_storage

        self.validators: List[Validator] = []
        self.critics: List[Critic] = []

        self.options: Dict[str, Any] = {
            "max_improvement_iterations": max_improvement_iterations,
            "apply_improvers_on_validation_failure": apply_improvers_on_validation_failure,
            "always_apply_critics": always_apply_critics,
        }

        self.chain_id = str(uuid.uuid4())

    def set_model(self, model: Model) -> None:
        """Set the model for the chain.

        Args:
            model: The model to use for text generation.
        """
        self.model = model

    def set_prompt(self, prompt: str) -> None:
        """Set the prompt for the chain.

        Args:
            prompt: The prompt to use for text generation.
        """
        self.prompt = prompt

    def set_model_retrievers(self, retrievers: List[Retriever]) -> None:
        """Set retrievers specifically for model context.

        Args:
            retrievers: List of retrievers to use for model context.
        """
        self.model_retrievers = retrievers

    def set_critic_retrievers(self, retrievers: List[Retriever]) -> None:
        """Set retrievers specifically for critic context.

        Args:
            retrievers: List of retrievers to use for critic context.
        """
        self.critic_retrievers = retrievers

    def add_validator(self, validator: Validator) -> None:
        """Add a validator to the chain.

        Args:
            validator: The validator to check if the generated text meets requirements.
        """
        self.validators.append(validator)

    def add_critic(self, critic: Critic) -> None:
        """Add a critic to the chain.

        Args:
            critic: The critic to improve the generated text.
        """
        self.critics.append(critic)

    def update_options(self, **options: Any) -> None:
        """Update options for the chain.

        Args:
            **options: Options to update for the chain.
        """
        self.options.update(options)

    def validate(self) -> None:
        """Validate that the chain configuration is complete and valid.

        Raises:
            ValueError: If the configuration is invalid or incomplete.
        """
        from sifaka.utils.error_handling import ConfigurationError

        errors = []
        suggestions = []

        # Check required components
        if not self.model:
            errors.append("No model specified for the chain")
            suggestions.extend(
                [
                    "Use create_model() to create a model instance",
                    "Example: model = create_model('openai:gpt-4')",
                    "Or use QuickStart.basic_chain() for simplified setup",
                ]
            )

        if not self.prompt:
            errors.append("No prompt specified for the chain")
            suggestions.extend(
                [
                    "Provide a prompt string for text generation",
                    "Example: prompt = 'Write a story about AI'",
                    "Or use QuickStart.for_development() for testing",
                ]
            )

        # Storage is guaranteed to be set in __init__, so no need to check for None

        # Validate model provider availability
        if self.model:
            try:
                # Try to get model info to validate it's properly configured
                model_name = getattr(self.model, "model_name", "unknown")
                if hasattr(self.model, "validate_configuration"):
                    self.model.validate_configuration()
            except Exception as e:
                errors.append(f"Model configuration error: {str(e)}")
                suggestions.extend(
                    [
                        "Check that required API keys are set in environment variables",
                        "Verify model name is correct and supported",
                        "Test model connectivity before creating the chain",
                    ]
                )

        # Validate validators
        for i, validator in enumerate(self.validators):
            if not hasattr(validator, "validate"):
                errors.append(f"Validator {i} does not implement validate() method")
                suggestions.append("Ensure all validators inherit from the Validator interface")

        # Validate critics
        for i, critic in enumerate(self.critics):
            if not hasattr(critic, "critique"):
                errors.append(f"Critic {i} does not implement critique() method")
                suggestions.append("Ensure all critics inherit from the Critic interface")

        # Validate options
        max_iterations = self.options.get("max_improvement_iterations", 3)
        if not isinstance(max_iterations, int) or max_iterations < 0:
            errors.append("max_improvement_iterations must be a non-negative integer")
            suggestions.append("Set max_improvement_iterations to a value >= 0")

        if max_iterations > 10:
            suggestions.append("Consider using fewer iterations (<=10) for better performance")

        # If there are errors, raise a comprehensive error
        if errors:
            error_message = "Chain configuration validation failed:\n" + "\n".join(
                f"- {error}" for error in errors
            )
            raise ConfigurationError(
                error_message,
                component="ChainConfig",
                operation="validation",
                suggestions=suggestions,
            )

    def get_option(self, key: str, default: Any = None) -> Any:
        """Get a configuration option value.

        Args:
            key: The option key to retrieve.
            default: Default value if key is not found.

        Returns:
            The option value or default.
        """
        return self.options.get(key, default)

    def copy(self) -> "ChainConfig":
        """Create a copy of this configuration.

        Returns:
            A new ChainConfig instance with the same configuration.
        """
        new_config = ChainConfig(
            model=self.model,
            prompt=self.prompt,
            model_retrievers=self.model_retrievers.copy(),
            critic_retrievers=self.critic_retrievers.copy(),
            storage=self.storage,
            checkpoint_storage=self.checkpoint_storage,
            **self.options,
        )

        # Copy validators and critics
        new_config.validators = self.validators.copy()
        new_config.critics = self.critics.copy()

        return new_config
