#!/usr/bin/env python3
"""
Test script to verify all Sifaka imports work correctly.

This test ensures that all core components can be imported without issues,
helping to catch circular import problems and missing dependencies.
"""

import pytest


def test_basic_python_imports():
    """Test that basic Python imports work."""
    import datetime  # noqa: F401
    import uuid  # noqa: F401
    from typing import Any, Dict, List, Optional  # noqa: F401
    from pydantic import BaseModel, Field  # noqa: F401


def test_core_thought_imports():
    """Test that core thought components can be imported."""
    from sifaka.core.thought import Document  # noqa: F401
    from sifaka.core.thought import ValidationResult  # noqa: F401
    from sifaka.core.thought import CriticFeedback  # noqa: F401
    from sifaka.core.thought import Thought  # noqa: F401


def test_core_interfaces_imports():
    """Test that core interfaces can be imported."""
    from sifaka.core.interfaces import Model, Validator, Critic, Retriever  # noqa: F401


def test_utils_imports():
    """Test that utility modules can be imported."""
    from sifaka.utils.error_handling import ChainError, chain_context  # noqa: F401
    from sifaka.utils.logging import get_logger  # noqa: F401


def test_chain_import():
    """Test that the main Chain class can be imported."""
    from sifaka.chain import Chain  # noqa: F401


def test_models_imports():
    """Test that model implementations can be imported."""
    from sifaka.models.base import create_model  # noqa: F401
    from sifaka.models.base import MockModel  # noqa: F401


def test_validators_imports():
    """Test that validator implementations can be imported."""
    from sifaka.validators.base import LengthValidator, RegexValidator  # noqa: F401


def test_critics_imports():
    """Test that critic implementations can be imported."""
    from sifaka.critics.reflexion import ReflexionCritic  # noqa: F401


def test_retrievers_imports():
    """Test that retriever implementations can be imported."""
    from sifaka.retrievers import MockRetriever  # noqa: F401


def test_all_imports_integration():
    """Integration test that imports all major components together."""
    # Import all major components in one test to catch interaction issues
    from sifaka.chain import Chain
    from sifaka.core.thought import Thought
    from sifaka.models.base import create_model
    from sifaka.validators.base import LengthValidator
    from sifaka.critics.reflexion import ReflexionCritic
    from sifaka.retrievers import MockRetriever

    # Verify we can create instances
    model = create_model("mock:test")
    validator = LengthValidator(min_length=10, max_length=100)
    critic = ReflexionCritic(model_name="mock:critic")
    retriever = MockRetriever()

    # Verify we can create a chain
    chain = Chain(model=model, prompt="Test prompt", retrievers=[retriever])

    # Verify chain configuration
    chain.validate_with(validator)
    chain.improve_with(critic)

    assert chain._model is not None
    assert chain._prompt == "Test prompt"
    assert len(chain._validators) == 1
    assert len(chain._critics) == 1


if __name__ == "__main__":
    # Run tests manually if executed directly
    print("🔍 Running Import Tests")
    print("=" * 40)

    tests = [
        test_basic_python_imports,
        test_core_thought_imports,
        test_core_interfaces_imports,
        test_utils_imports,
        test_chain_import,
        test_models_imports,
        test_validators_imports,
        test_critics_imports,
        test_retrievers_imports,
        test_all_imports_integration,
    ]

    for test_func in tests:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
            break
    else:
        print("\n🎉 All import tests passed!")
