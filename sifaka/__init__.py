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


# Import lazily to avoid circular dependencies
def __getattr__(name):
    """Lazily import components to avoid circular dependencies."""
    if name in ("Chain", "ChainResult"):
        from sifaka.chain import Chain, ChainResult

        globals()[name] = locals()[name]
        return locals()[name]
    elif name in ("CriticCore", "create_prompt_critic", "create_reflexion_critic"):
        from sifaka.critics import CriticCore, create_prompt_critic, create_reflexion_critic

        globals()[name] = locals()[name]
        return locals()[name]
    elif name == "Generator":
        from sifaka.core.generation import Generator

        globals()[name] = Generator
        return Generator
    elif name in ("Improver", "ImprovementResult"):
        from sifaka.core.improvement import Improver, ImprovementResult

        globals()[name] = locals()[name]
        return locals()[name]
    elif name in ("AnthropicProvider", "OpenAIProvider"):
        from sifaka.models import AnthropicProvider, OpenAIProvider

        globals()[name] = locals()[name]
        return locals()[name]
    elif name in ("Validator", "ValidationResult", "ValidatorConfig"):
        from sifaka.core.validation import Validator, ValidationResult, ValidatorConfig

        globals()[name] = locals()[name]
        return locals()[name]
    elif name == "Rule":
        from sifaka.rules.base import Rule

        globals()[name] = Rule
        return Rule
    elif name == "create_length_rule":
        from sifaka.rules.formatting.length import create_length_rule

        globals()[name] = create_length_rule
        return create_length_rule

    raise AttributeError(f"module 'sifaka' has no attribute '{name}'")


# Define __all__ to specify public API
__all__ = [
    # Core components
    "Chain",
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
