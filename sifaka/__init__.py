"""
Sifaka: A framework for building reliable and reflective AI systems.
"""

from sifaka.chain import ChainCore, ChainResult
from sifaka.critics import CriticCore, create_prompt_critic, create_reflexion_critic
from sifaka.generation import Generator
from sifaka.improvement import Improver, ImprovementResult
from sifaka.models import AnthropicProvider, OpenAIProvider
from sifaka.validation import Validator, ValidationResult

from sifaka.rules import (
    LengthRule,
    Rule,
    RuleConfig,
    create_length_rule,
    create_prohibited_content_rule,
    create_sentiment_rule,
    create_toxicity_rule,
)

# Version information
__version__ = "0.1.0"

__all__ = [
    # Core components
    "ChainCore",
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
    "create_length_rule",
    "create_prohibited_content_rule",
    "create_sentiment_rule",
    "create_toxicity_rule",
    # Critics
    "CriticCore",
    "create_prompt_critic",
    "create_reflexion_critic",
]
