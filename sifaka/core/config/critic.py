"""Critic-specific configuration."""

from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from ..type_defs import CriticSettings
from ..types import CriticType
from .base import BaseConfig


class CriticConfig(BaseConfig):
    """Configuration for critic behavior and settings.

    Controls which critics are used, how they behave, and critic-specific
    parameters like confidence thresholds and tool usage.

    Example:
        >>> critic_config = CriticConfig(
        ...     critics=["reflexion", "self_rag"],
        ...     base_confidence=0.8,
        ...     enable_tools=True
        ... )
    """

    # Critic selection
    critics: List[CriticType] = Field(
        default_factory=lambda: [CriticType.REFLEXION],
        description="List of critics to use for text improvement",
    )

    @field_validator("critics", mode="before")
    @classmethod
    def validate_critics(cls, v: Any) -> List[CriticType]:
        """Validate critic types - ONLY CriticType enums allowed."""
        if isinstance(v, CriticType):
            v = [v]

        if not isinstance(v, list):
            raise ValueError(
                f"critics must be a list of CriticType enums, got {type(v).__name__}"
            )

        result: List[CriticType] = []
        for critic in v:
            if isinstance(critic, CriticType):
                result.append(critic)
            else:
                available = ", ".join(f"CriticType.{c.name}" for c in CriticType)
                raise ValueError(
                    f"Invalid critic type: {critic} (type: {type(critic).__name__}). "
                    f"Must use CriticType enum values: {available}"
                )

        return result

    # General critic behavior
    base_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Base confidence level for critic assessments",
    )

    context_window: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of previous critiques to include as context",
    )

    # Timeout for critic operations
    timeout_seconds: float = Field(
        default=30.0,
        ge=10.0,
        le=300.0,
        description="Timeout for individual critic operations",
    )

    # Tool usage settings
    enable_tools: bool = Field(
        default=True, description="Enable tool usage for critics that support it"
    )

    tool_timeout: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Maximum time for tool calls in seconds",
    )

    tool_cache_ttl: int = Field(
        default=3600,
        ge=0,
        description="Tool result cache TTL in seconds (0 = no cache)",
    )

    # Per-critic settings
    critic_settings: Dict[str, CriticSettings] = Field(
        default_factory=dict,  # Empty dict for now, will be populated in __init__
        description="Per-critic configuration overrides",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize with default critic settings."""
        # Set default critic settings if not provided
        if "critic_settings" not in data:
            data["critic_settings"] = {
                "self_rag": {"enable_tools": True},
                "constitutional": {"enable_tools": False},
                "meta_rewarding": {"enable_tools": False},
                "reflexion": {"enable_tools": False},
                "self_consistency": {"enable_tools": False, "num_samples": 3},
                "self_refine": {"enable_tools": False},
                "style": {"enable_tools": False},
            }
        super().__init__(**data)

    # Style critic specific settings
    style_reference_text: Optional[str] = Field(
        default=None, description="Reference text for Style critic"
    )

    style_description: Optional[str] = Field(
        default=None, description="Description of target style for Style critic"
    )

    style_examples: Optional[List[str]] = Field(
        default=None, description="Example phrases in target style"
    )

    # Constitutional AI settings
    constitutional_principles: Optional[List[str]] = Field(
        default=None, description="Custom principles for Constitutional AI critic"
    )

    # Self-consistency settings
    self_consistency_num_samples: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Number of samples for Self-Consistency critic",
    )

    def get_critic_settings(self, critic_name: str) -> CriticSettings:
        """Get settings for a specific critic.

        Args:
            critic_name: Name of the critic

        Returns:
            Dictionary of settings for the critic
        """
        base_settings: CriticSettings = {
            "enable_tools": self.enable_tools,
            "tool_timeout": self.tool_timeout,
            "base_confidence": self.base_confidence,
            "context_window": self.context_window,
        }

        # Merge with critic-specific settings
        specific_settings = self.critic_settings.get(critic_name, {})
        # Create a new dict to avoid TypedDict issues
        result: CriticSettings = {}
        result.update(base_settings)
        result.update(specific_settings)
        return result

    def is_tools_enabled_for(self, critic_name: str) -> bool:
        """Check if tools are enabled for a specific critic."""
        settings = self.get_critic_settings(critic_name)
        return bool(settings.get("enable_tools", self.enable_tools))
