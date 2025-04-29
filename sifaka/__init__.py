"""
Sifaka: A framework for building reliable and reflective AI systems.
"""

from sifaka.chain import Chain
from sifaka.critics import PromptCritic
from sifaka.models import AnthropicProvider, OpenAIProvider

from sifaka.rules import (
    LengthRule,
    ProhibitedContentRule,
    Rule,
    RuleConfig,
    SentimentRule,
    ToxicityRule,
    create_length_rule,
)

# Version information
__version__ = "0.1.0"

__all__ = [
    # Core components
    "Chain",
    "Rule",
    "RuleConfig",
    # Model providers
    "AnthropicProvider",
    "OpenAIProvider",
    # Rules
    "LengthRule",
    "ProhibitedContentRule",
    "SentimentRule",
    "ToxicityRule",
    "create_length_rule",
    # Critics
    "PromptCritic",
]
