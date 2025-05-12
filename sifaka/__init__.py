"""
Sifaka: A framework for building reliable and reflective AI systems.
"""

from typing import List, Any


def __getattr__(name) -> Any:
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


__all__: List[str] = [
    "Chain",
    "ChainResult",
    "Generator",
    "Improver",
    "ImprovementResult",
    "Validator",
    "ValidationResult",
    "ValidatorConfig",
    "Rule",
    "AnthropicProvider",
    "OpenAIProvider",
    "create_length_rule",
    "CriticCore",
    "create_prompt_critic",
    "create_reflexion_critic",
]
__version__: str = "0.1.0"
