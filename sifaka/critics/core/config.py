"""Simplified critic configuration."""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class CriticConfig(BaseModel):
    """Essential configuration for critic behavior."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Core settings
    base_confidence: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="Base confidence level for assessments"
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Model temperature for critic responses"
    )
    
    # Response format (JSON only for consistency)
    response_format: str = Field(
        default="json",
        const="json",
        description="Response format (JSON only)"
    )
    
    # Context settings
    context_window: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of previous critiques to consider"
    )
    
    # Timeout
    timeout_seconds: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Timeout for critic operations"
    )


# Critic-specific configurations (minimal, essential only)
class ConstitutionalConfig(CriticConfig):
    """Configuration for Constitutional AI critic."""
    
    principles: List[str] = Field(
        default=[
            "Clarity: Is the text clear and easy to understand?",
            "Accuracy: Are the claims factually correct?",
            "Completeness: Does the text fully address the topic?",
            "Objectivity: Is the text balanced and unbiased?",
            "Engagement: Is the text interesting and engaging?"
        ],
        description="Evaluation principles"
    )
    
    temperature: float = Field(default=0.3)  # Lower for principled evaluation


class SelfConsistencyConfig(CriticConfig):
    """Configuration for Self-Consistency critic."""
    
    num_samples: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Number of evaluation samples"
    )
    
    temperature: float = Field(default=0.8)  # Higher for diversity


class NCriticsConfig(CriticConfig):
    """Configuration for N-Critics ensemble."""
    
    perspectives: List[str] = Field(
        default=[
            "Technical expert focused on accuracy",
            "General reader focused on clarity",
            "Editor focused on structure and flow"
        ],
        description="Critical perspectives"
    )
    
    temperature: float = Field(default=0.6)  # Medium for balanced perspectives