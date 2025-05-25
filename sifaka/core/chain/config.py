"""Chain configuration management for Sifaka.

This module contains the ChainConfig class which manages all configuration
state for a Chain, including models, retrievers, validators, critics, and options.
"""

import uuid
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Critic, Model, Retriever, Validator
from sifaka.storage.protocol import Storage
from sifaka.storage.checkpoints import CachedCheckpointStorage


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
        if not self.model:
            raise ValueError("No model specified for the chain")
        if not self.prompt:
            raise ValueError("No prompt specified for the chain")

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
