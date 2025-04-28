"""
Sifaka: A framework for building reliable and reflective AI systems.
"""

import warnings

from sifaka.chain import Chain
from sifaka.critics import PromptCritic
from sifaka.models import AnthropicProvider, OpenAIProvider

# Import deprecated Reflector for backward compatibility
try:
    from sifaka.reflector import Reflector
except ImportError:
    # Create a dummy Reflector class for backward compatibility
    class Reflector:
        """
        Deprecated: This class is kept for backward compatibility.
        """

        def __init__(self, *args, **kwargs):
            warnings.warn(
                "The Reflector class is deprecated and will be removed in version 2.0.0. "
                "See the migration guide in the documentation for more details.",
                DeprecationWarning,
                stacklevel=2,
            )


from sifaka.rules import (
    FormatRule,
    LengthRule,
    ProhibitedContentRule,
    Rule,
    SentimentRule,
    ToxicityRule,
)

# Version information
__version__ = "1.0.0"

# Show deprecation warning for Reflector
warnings.warn(
    "The Reflector class is deprecated and will be removed in version 2.0.0. "
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
    # Critics
    "PromptCritic",
    # Deprecated (will be removed in 2.0.0)
    "Reflector",
]
