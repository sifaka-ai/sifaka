"""Composite configuration that combines all config modules."""

from typing import List, Optional

from pydantic import Field, model_validator

from ..types import StorageType
from ..validation import validate_config_params
from .base import BaseConfig
from .critic import CriticConfig
from .engine import EngineConfig
from .llm import LLMConfig
from .storage import StorageConfig
from .validation import ValidationConfig


class Config(BaseConfig):
    """Composite configuration for Sifaka operations.

    Clean, simple configuration using focused sub-configs.
    No backward compatibility cruft.

    Example:
        >>> config = Config(
        ...     llm=LLMConfig(model="gpt-4", temperature=0.8),
        ...     engine=EngineConfig(max_iterations=5),
        ...     critic=CriticConfig(critics=["reflexion", "self_rag"])
        ... )
    """

    # Sub-configurations - that's it, no magic
    llm: LLMConfig = Field(default_factory=LLMConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @model_validator(mode="after")
    def validate_config_consistency(self) -> "Config":
        """Validate configuration consistency using enhanced validation."""
        try:
            validate_config_params(
                model=self.llm.model,
                temperature=self.llm.temperature,
                max_iterations=self.engine.max_iterations,
                timeout_seconds=self.llm.timeout_seconds,
            )
        except ValueError as e:
            raise ValueError(f"Configuration validation failed: {e}") from e

        return self

    @classmethod
    def fast(cls) -> "Config":
        """Fast configuration for quick iterations."""
        return cls(
            llm=LLMConfig(
                model="gpt-3.5-turbo",
                critic_model="gpt-3.5-turbo",
                temperature=0.7,
                timeout_seconds=30,
            ),
            engine=EngineConfig(
                max_iterations=2,
                total_timeout_seconds=60,
                force_improvements=False,
            ),
            critic=CriticConfig(
                timeout_seconds=15,
            ),
        )

    @classmethod
    def quality(cls) -> "Config":
        """High-quality configuration for best results."""
        return cls(
            llm=LLMConfig(
                model="gpt-4o",
                critic_model="gpt-4o-mini",
                temperature=0.6,
                timeout_seconds=60,
            ),
            engine=EngineConfig(
                max_iterations=5,
                total_timeout_seconds=600,
                force_improvements=True,
            ),
            critic=CriticConfig(
                base_confidence=0.8,
                timeout_seconds=60,
            ),
        )

    @classmethod
    def creative(cls) -> "Config":
        """Creative configuration for imaginative content."""
        return cls(
            llm=LLMConfig(
                model="gpt-4o-mini",
                critic_model="gpt-4o-mini",
                temperature=0.9,
                critic_temperature=0.7,
            ),
            engine=EngineConfig(
                max_iterations=3,
                force_improvements=True,
            ),
        )

    @classmethod
    def research(cls) -> "Config":
        """Research configuration with fact-checking tools."""
        return cls(
            llm=LLMConfig(
                model="gpt-4o",
                critic_model="gpt-4o",
                temperature=0.5,
            ),
            engine=EngineConfig(
                max_iterations=4,
                force_improvements=True,
            ),
            critic=CriticConfig(
                enable_tools=True,
                critic_settings={
                    "self_rag": {"enable_tools": True},
                    "constitutional": {"enable_tools": True},
                    "reflexion": {"enable_tools": True},
                },
            ),
        )

    @classmethod
    def minimal(cls) -> "Config":
        """Minimal configuration with bare essentials."""
        return cls(
            llm=LLMConfig(
                model="gpt-3.5-turbo",
                temperature=0.7,
            ),
            engine=EngineConfig(
                max_iterations=1,
                retry_enabled=False,
                enable_middleware=False,
                enable_metrics=False,
            ),
            storage=StorageConfig(
                backend=StorageType.MEMORY,
                store_thoughts=False,
            ),
        )

    @classmethod
    def development(cls) -> "Config":
        """Development configuration with debugging enabled."""
        return cls(
            engine=EngineConfig(
                show_improvement_prompt=True,
                show_critic_prompts=True,
                enable_metrics=True,
            ),
            storage=StorageConfig(
                backend=StorageType.FILE,
                storage_path="./dev_storage",
                store_thoughts=True,
            ),
        )

    @classmethod
    def production(cls) -> "Config":
        """Production configuration with optimizations."""
        return cls(
            llm=LLMConfig(
                model="gpt-4o-mini",
                critic_model="gpt-3.5-turbo",
                temperature=0.7,
                connection_pool_size=20,
            ),
            engine=EngineConfig(
                max_iterations=3,
                parallel_critics=True,
                enable_caching=True,
                retry_max_attempts=5,
            ),
            storage=StorageConfig(
                backend=StorageType.REDIS,
                enable_compression=True,
                ttl_seconds=86400,  # 24 hours
            ),
        )

    @classmethod
    def style_transfer(
        cls,
        style_guide: str,
        examples: Optional[List[str]] = None,
        reference_text: Optional[str] = None,
    ) -> "Config":
        """Configuration for style transformation."""
        return cls(
            llm=LLMConfig(
                model="gpt-4o-mini",
                critic_model="gpt-3.5-turbo",
                temperature=0.7,
            ),
            engine=EngineConfig(
                max_iterations=3,
                force_improvements=True,
            ),
            critic=CriticConfig(
                style_description=style_guide,
                style_examples=examples,
                style_reference_text=reference_text,
            ),
        )
