"""
Sifaka: A framework for building reliable and reflective AI systems.
"""

from sifaka.chain import Chain, ChainResult
from sifaka.critics import PromptCritic
from sifaka.generation import Generator
from sifaka.improvement import Improver, ImprovementResult
from sifaka.models import AnthropicProvider, OpenAIProvider
from sifaka.validation import Validator, ValidationResult

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
    "ChainResult",
    "Generator",
    "Improver",
    "ImprovementResult",
    "Validator",
    "ValidationResult",
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
