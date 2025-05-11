"""
Test script to check for circular imports in the Sifaka codebase.
"""

import sys
import os
import importlib
import traceback


def test_import(module_name):
    """Test importing a module and report any issues."""
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as e:
        return False, f"Error importing {module_name}: {str(e)}\n{traceback.format_exc()}"


def main():
    """Test importing all major Sifaka components."""
    modules_to_test = [
        # Core modules
        "sifaka.core.base",
        "sifaka.core.dependency",
        "sifaka.core.factories",
        # Interface modules
        "sifaka.interfaces.model",
        "sifaka.interfaces.chain",
        "sifaka.interfaces.retrieval",
        "sifaka.interfaces.classifier",
        "sifaka.interfaces.critic",
        "sifaka.interfaces.adapter",
        "sifaka.interfaces.rule",
        # Model modules
        "sifaka.models.base",
        "sifaka.models.core",
        "sifaka.models.config",
        "sifaka.models.factories",
        "sifaka.models.providers.openai",
        "sifaka.models.providers.anthropic",
        # Chain modules
        "sifaka.chain.chain",
        "sifaka.chain.config",
        "sifaka.chain.factories",
        "sifaka.chain.engine",
        "sifaka.chain.managers.memory",
        # Retrieval modules
        "sifaka.retrieval.core",
        "sifaka.retrieval.config",
        "sifaka.retrieval.factories",
        # Classifier modules
        "sifaka.classifiers.classifier",
        "sifaka.classifiers.config",
        "sifaka.classifiers.factories",
        "sifaka.classifiers.engine",
        # Critic modules
        "sifaka.critics.base",
        "sifaka.critics.core",
        "sifaka.critics.config",
        "sifaka.critics.implementations.prompt",
        "sifaka.critics.strategies.improvement",
        # Adapter modules
        "sifaka.adapters.base",
        "sifaka.adapters.pydantic_ai.adapter",
        "sifaka.adapters.guardrails.adapter",
        # Rule modules
        "sifaka.rules.base",
        "sifaka.rules.factories",
        # Utility modules
        "sifaka.utils.config",
        "sifaka.utils.state",
        "sifaka.utils.errors",
    ]

    failures = []

    print("Testing imports for circular dependencies...")
    for module in modules_to_test:
        success, error = test_import(module)
        if not success:
            failures.append((module, error))
        else:
            print(f"✅ {module}")

    if failures:
        print("\nThe following modules have import issues:")
        for module, error in failures:
            print(f"\n❌ {module}")
            print(error)
        return 1
    else:
        print("\nAll modules imported successfully! No circular dependencies detected.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
