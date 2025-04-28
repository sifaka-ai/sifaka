"""
Sifaka: A framework for building reliable and reflective AI systems.
"""

import warnings

from sifaka.chain import Chain
from sifaka.critics import PromptCritic
from sifaka.models import AnthropicProvider, OpenAIProvider

# Import deprecated Reflector for backward compatibility
from sifaka.reflector import Reflector
from sifaka.rules import (
    FormatRule,
    LengthRule,
    ProhibitedContentRule,
    RepetitionRule,
    Rule,
    SentimentRule,
    SymmetryRule,
    ToxicityRule,
)

# Version information
__version__ = "1.0.0"

# Show deprecation warning for Reflector
warnings.warn(
    "The Reflector class is deprecated and will be removed in version 2.0.0. "
    "Use SymmetryRule and RepetitionRule from sifaka.rules.pattern_rules instead. "
    "See the migration guide in the documentation for more details.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # Core components
    "Chain",
    "Rule",
    # Model providers
    "AnthropicProvider",
    "OpenAIProvider",
    # Rules
    "LengthRule",
    "ProhibitedContentRule",
    "SentimentRule",
    "ToxicityRule",
    "FormatRule",
    "SymmetryRule",
    "RepetitionRule",
    # Critics
    "PromptCritic",
    # Deprecated (will be removed in 2.0.0)
    "Reflector",
]
