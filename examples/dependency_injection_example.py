#!/usr/bin/env python
"""
Dependency Injection Example

This example demonstrates how to use the dependency injection system in Sifaka.
It shows how to create custom model factories and inject them into the Chain class.
"""

import sys
import os
from typing import Any, Dict, Optional

# Add the project root to the path so we can import sifaka
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sifaka.chain import Chain
from sifaka.interfaces import Model
from sifaka.registry import register_model, get_model_factory


class CustomModel:
    """A custom model implementation for demonstration purposes."""

    def __init__(self, model_name: str, **options: Any):
        """Initialize the custom model."""
        self.model_name = model_name
        self.options = options
        print(f"Initialized CustomModel with model_name={model_name}, options={options}")

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt."""
        merged_options = {**self.options, **options}
        print(f"Generating with prompt: {prompt}, options: {merged_options}")
        return f"Custom response for: {prompt}"

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(text.split())


@register_model("custom")
def create_custom_model(model_name: str, **options: Any) -> Model:
    """Create a custom model instance.

    This factory function is registered with the registry system using the
    @register_model decorator.
    """
    return CustomModel(model_name, **options)


def demonstrate_direct_injection():
    """Demonstrate direct injection of a model instance."""
    print("\n=== Direct Injection of Model Instance ===")

    # Create a custom model
    model = CustomModel("custom-model", temperature=0.7)

    # Create a chain with the custom model
    chain = Chain().with_model(model).with_prompt("Tell me a story")

    # Run the chain
    result = chain.run()

    # Print the result
    print(f"Result: {result.text}")


def demonstrate_factory_injection():
    """Demonstrate injection of a model factory."""
    print("\n=== Injection of Model Factory ===")

    # Create a chain with a custom model factory
    chain = Chain(model_factory=create_custom_model)

    # Configure the chain
    chain.with_model("custom:model").with_prompt("Tell me a joke")

    # Run the chain
    result = chain.run()

    # Print the result
    print(f"Result: {result.text}")


def demonstrate_registry_usage():
    """Demonstrate usage of the registry system."""
    print("\n=== Registry System Usage ===")

    # The model is already registered with the @register_model decorator

    # Get the factory from the registry
    factory = get_model_factory("custom")

    if factory:
        # Create a model using the factory
        model = factory("registry-model", temperature=0.5)

        # Create a chain with the model
        chain = Chain().with_model(model).with_prompt("Tell me a fact")

        # Run the chain
        result = chain.run()

        # Print the result
        print(f"Result: {result.text}")
    else:
        print("Factory not found in registry")


if __name__ == "__main__":
    # Demonstrate different ways to use dependency injection
    demonstrate_direct_injection()
    demonstrate_factory_injection()
    demonstrate_registry_usage()
