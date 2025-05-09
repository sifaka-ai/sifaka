"""
Sifaka: A framework for building reliable and reflective AI systems.
"""

# Apply compatibility patches early
try:
    from sifaka.utils.patches import apply_all_patches

    apply_all_patches()
except ImportError:
    # If patches are not available, continue without them
    pass

from sifaka.chain import ChainCore, ChainResult
from sifaka.critics import CriticCore, create_prompt_critic, create_reflexion_critic
from sifaka.generation import Generator
from sifaka.improvement import Improver, ImprovementResult
from sifaka.models import AnthropicProvider, OpenAIProvider
from sifaka.validation import Validator, ValidationResult, ValidatorConfig

from sifaka.rules.base import Rule
from sifaka.rules.formatting.length import create_length_rule

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
    "ValidatorConfig",
    "Rule",
    # Model providers
    "AnthropicProvider",
    "OpenAIProvider",
    # Rules
    "create_length_rule",
    # Critics
    "CriticCore",
    "create_prompt_critic",
    "create_reflexion_critic",
]
