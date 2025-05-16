#!/usr/bin/env python
"""
Basic Example

This example demonstrates the basic usage of the Sifaka framework with the new
dependency injection architecture.
"""

import sys
import os
from typing import Any

# Add the project root to the path so we can import sifaka
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sifaka import Chain, create_model
from sifaka.validators import length
from sifaka.improvers import clarity


def run_basic_example():
    """Run a basic example of the Sifaka framework."""
    print("\n=== Basic Example ===\n")
    
    # Create a model
    model = create_model("openai", "gpt-3.5-turbo")
    
    # Create a chain
    chain = Chain()
    
    # Configure the chain
    chain.with_model(model)
    chain.with_prompt("Write a short story about a robot in 3 sentences.")
    chain.validate_with(length(min_length=3, max_length=5, unit="words"))
    chain.improve_with(clarity(model))
    
    # Run the chain
    result = chain.run()
    
    # Print the result
    print(f"Validation passed: {result.passed}")
    print(f"Text: {result.text}")
    
    # Print validation results
    print("\nValidation Results:")
    for i, validation_result in enumerate(result.validation_results):
        print(f"  {i+1}. {validation_result.message}")
    
    # Print improvement results
    print("\nImprovement Results:")
    for i, improvement_result in enumerate(result.improvement_results):
        print(f"  {i+1}. {improvement_result.message}")
        if improvement_result.changes_made:
            print(f"     Original: {improvement_result.original_text[:100]}...")
            print(f"     Improved: {improvement_result.improved_text[:100]}...")


def run_string_example():
    """Run an example using string-based model specification."""
    print("\n=== String-Based Model Example ===\n")
    
    # Create a chain with a string-based model
    chain = Chain()
    
    # Configure the chain
    chain.with_model("openai:gpt-3.5-turbo")
    chain.with_prompt("Explain quantum computing in 2 sentences.")
    chain.validate_with(length(max_length=50, unit="words"))
    
    # Run the chain
    result = chain.run()
    
    # Print the result
    print(f"Validation passed: {result.passed}")
    print(f"Text: {result.text}")


def run_factory_example():
    """Run an example using factory functions."""
    print("\n=== Factory Functions Example ===\n")
    
    from sifaka.factories import create_model_from_string, create_validator, create_improver
    
    # Create components using factory functions
    model = create_model_from_string("openai:gpt-3.5-turbo")
    validator = create_validator("length", min_length=10, max_length=100, unit="words")
    improver = create_improver("clarity", model)
    
    # Create a chain
    chain = Chain()
    
    # Configure the chain
    chain.with_model(model)
    chain.with_prompt("Write a haiku about programming.")
    chain.validate_with(validator)
    chain.improve_with(improver)
    
    # Run the chain
    result = chain.run()
    
    # Print the result
    print(f"Validation passed: {result.passed}")
    print(f"Text: {result.text}")


if __name__ == "__main__":
    # Run the examples
    run_basic_example()
    run_string_example()
    run_factory_example()
