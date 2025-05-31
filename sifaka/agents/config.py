"""Configuration objects for PydanticAI chains.

This module provides configuration classes to replace scattered parameters
and simplify chain initialization.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from sifaka.core.interfaces import Critic, Validator
from sifaka.storage.protocol import Storage


@dataclass
class ChainConfig:
    """Configuration object for PydanticAI chains.

    This replaces scattered parameters with a single configuration object,
    making chain initialization cleaner and more maintainable.
    """

    # Core execution settings
    max_improvement_iterations: int = 2
    always_apply_critics: bool = False

    # Components
    validators: List[Validator] = field(default_factory=list)
    critics: List[Critic] = field(default_factory=list)
    model_retrievers: List = field(default_factory=list)
    critic_retrievers: List = field(default_factory=list)

    # Storage (no aliases - clean API)
    analytics_storage: Optional[Storage] = None

    # Chain identification
    chain_id: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_improvement_iterations < 0:
            raise ValueError("max_improvement_iterations must be non-negative")

        if not isinstance(self.validators, list):
            raise TypeError("validators must be a list")

        if not isinstance(self.critics, list):
            raise TypeError("critics must be a list")

        if not isinstance(self.model_retrievers, list):
            raise TypeError("model_retrievers must be a list")

        if not isinstance(self.critic_retrievers, list):
            raise TypeError("critic_retrievers must be a list")

    @classmethod
    def create(
        cls,
        validators: Optional[List[Validator]] = None,
        critics: Optional[List[Critic]] = None,
        model_retrievers: Optional[List] = None,
        critic_retrievers: Optional[List] = None,
        max_improvement_iterations: int = 2,
        always_apply_critics: bool = False,
        analytics_storage: Optional[Storage] = None,
        chain_id: Optional[str] = None,
    ) -> "ChainConfig":
        """Factory method to create a ChainConfig with clean defaults.

        Args:
            validators: List of validators to apply.
            critics: List of critics to apply.
            model_retrievers: List of retrievers for pre-generation context.
            critic_retrievers: List of retrievers for pre-critic context.
            max_improvement_iterations: Maximum number of improvement iterations.
            always_apply_critics: Whether to always apply critics.
            analytics_storage: Optional storage backend for analytics.
            chain_id: Optional chain identifier.

        Returns:
            A configured ChainConfig instance.
        """
        return cls(
            validators=validators or [],
            critics=critics or [],
            model_retrievers=model_retrievers or [],
            critic_retrievers=critic_retrievers or [],
            max_improvement_iterations=max_improvement_iterations,
            always_apply_critics=always_apply_critics,
            analytics_storage=analytics_storage,
            chain_id=chain_id,
        )
