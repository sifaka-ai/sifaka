#!/usr/bin/env python
"""
Registry System Example

This example demonstrates how to use the registry system in Sifaka to register
and retrieve custom components.
"""

import sys
import os
from typing import Dict, Any, List, Optional, Tuple

# Add the project root to the path so we can import sifaka
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sifaka.interfaces import Model, Validator, Improver
from sifaka.registry import (
    register_model,
    register_validator,
    register_improver,
    get_model_factory,
    get_validator_factory,
    get_improver_factory,
)
from sifaka.factories import (
    create_model,
    create_model_from_string,
    create_validator,
    create_improver,
)
from sifaka.results import ValidationResult, ImprovementResult
from sifaka.chain import Chain


class EchoModel:
    """A simple model that echoes the prompt."""

    def __init__(self, model_name: str, **options: Any):
        """Initialize the echo model.
        
        Args:
            model_name: The name of the model.
            **options: Additional options.
        """
        self.model_name = model_name
        self.options = options
        self.prefix = options.get("prefix", "")
        self.suffix = options.get("suffix", "")
    
    def generate(self, prompt: str, **options: Any) -> str:
        """Generate a response by echoing the prompt.
        
        Args:
            prompt: The prompt to echo.
            **options: Additional options.
            
        Returns:
            The echoed prompt with optional prefix and suffix.
        """
        return f"{self.prefix}{prompt}{self.suffix}"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (simplified as word count).
        
        Args:
            text: The text to count tokens in.
            
        Returns:
            The number of words in the text.
        """
        return len(text.split())


class LengthValidator:
    """A validator that checks text length."""
    
    def __init__(self, min_length: Optional[int] = None, max_length: Optional[int] = None, **options: Any):
        """Initialize the length validator.
        
        Args:
            min_length: Minimum allowed length.
            max_length: Maximum allowed length.
            **options: Additional options.
        """
        self.min_length = min_length
        self.max_length = max_length
        self.options = options
    
    def validate(self, text: str) -> ValidationResult:
        """Validate text length.
        
        Args:
            text: The text to validate.
            
        Returns:
            ValidationResult indicating whether the text length is valid.
        """
        length = len(text)
        
        if self.min_length is not None and length < self.min_length:
            return ValidationResult(
                passed=False,
                message=f"Text is too short ({length} chars, minimum {self.min_length})",
                details={"length": length, "min_length": self.min_length}
            )
        
        if self.max_length is not None and length > self.max_length:
            return ValidationResult(
                passed=False,
                message=f"Text is too long ({length} chars, maximum {self.max_length})",
                details={"length": length, "max_length": self.max_length}
            )
        
        return ValidationResult(
            passed=True,
            message=f"Text length is valid ({length} chars)",
            details={"length": length}
        )


class ReverseImprover:
    """An improver that reverses text."""
    
    def __init__(self, model: Model, **options: Any):
        """Initialize the reverse improver.
        
        Args:
            model: The model to use (not actually used).
            **options: Additional options.
        """
        self.model = model
        self.options = options
    
    def improve(self, text: str) -> Tuple[str, ImprovementResult]:
        """Improve text by reversing it.
        
        Args:
            text: The text to improve.
            
        Returns:
            Tuple of (improved_text, ImprovementResult).
        """
        improved_text = text[::-1]
        
        return improved_text, ImprovementResult(
            original_text=text,
            improved_text=improved_text,
            changes_made=True,
            message="Text has been reversed",
            details={"method": "reverse"}
        )


# Register the custom components
@register_model("echo")
def create_echo_model(model_name: str, **options: Any) -> Model:
    """Create an echo model.
    
    Args:
        model_name: The name of the model.
        **options: Additional options.
        
    Returns:
        An EchoModel instance.
    """
    return EchoModel(model_name, **options)


@register_validator("char_length")
def create_char_length_validator(min_length: Optional[int] = None, max_length: Optional[int] = None, **options: Any) -> Validator:
    """Create a character length validator.
    
    Args:
        min_length: Minimum allowed length.
        max_length: Maximum allowed length.
        **options: Additional options.
        
    Returns:
        A LengthValidator instance.
    """
    return LengthValidator(min_length=min_length, max_length=max_length, **options)


@register_improver("reverse")
def create_reverse_improver(model: Model, **options: Any) -> Improver:
    """Create a reverse improver.
    
    Args:
        model: The model to use.
        **options: Additional options.
        
    Returns:
        A ReverseImprover instance.
    """
    return ReverseImprover(model, **options)


def demonstrate_registry():
    """Demonstrate the registry system."""
    print("\n=== Registry System Demonstration ===")
    
    # Get factories from the registry
    echo_factory = get_model_factory("echo")
    length_factory = get_validator_factory("char_length")
    reverse_factory = get_improver_factory("reverse")
    
    # Create components using factories
    echo_model = echo_factory("echo-model", prefix="Echo: ")
    length_validator = length_factory(min_length=10, max_length=100)
    reverse_improver = reverse_factory(echo_model)
    
    # Use the components
    prompt = "Hello, world!"
    response = echo_model.generate(prompt)
    validation_result = length_validator.validate(response)
    improved_text, improvement_result = reverse_improver.improve(response)
    
    print(f"Original prompt: {prompt}")
    print(f"Model response: {response}")
    print(f"Validation result: {validation_result.passed} - {validation_result.message}")
    print(f"Improved text: {improved_text}")
    print(f"Improvement result: {improvement_result.message}")


def demonstrate_factories():
    """Demonstrate the factory functions."""
    print("\n=== Factory Functions Demonstration ===")
    
    # Create components using factory functions
    echo_model = create_model("echo", "echo-model", prefix="Factory: ")
    length_validator = create_validator("char_length", min_length=15, max_length=150)
    reverse_improver = create_improver("reverse", echo_model)
    
    # Use the components
    prompt = "Using factory functions!"
    response = echo_model.generate(prompt)
    validation_result = length_validator.validate(response)
    improved_text, improvement_result = reverse_improver.improve(response)
    
    print(f"Original prompt: {prompt}")
    print(f"Model response: {response}")
    print(f"Validation result: {validation_result.passed} - {validation_result.message}")
    print(f"Improved text: {improved_text}")
    print(f"Improvement result: {improvement_result.message}")


def demonstrate_chain():
    """Demonstrate using the components with Chain."""
    print("\n=== Chain Integration Demonstration ===")
    
    # Create a chain
    chain = Chain()
    
    # Create components
    model = create_model("echo", "echo-model", prefix="Chain: ")
    validator = create_validator("char_length", min_length=20, max_length=200)
    improver = create_improver("reverse", model)
    
    # Configure the chain
    chain.with_model(model)
    chain.with_prompt("Using the Chain API!")
    chain.validate_with(validator)
    chain.improve_with(improver)
    
    # Run the chain
    result = chain.run()
    
    # Print the result
    print(f"Original prompt: {chain._prompt}")
    print(f"Final result: {result.text}")
    print(f"Validation results: {len(result.validation_results)}")
    for i, vr in enumerate(result.validation_results):
        print(f"  {i+1}. {vr.passed} - {vr.message}")
    print(f"Improvement results: {len(result.improvement_results)}")
    for i, ir in enumerate(result.improvement_results):
        print(f"  {i+1}. {ir.message}")


def main():
    """Run the registry example."""
    print("Registry System Example")
    print("======================")
    
    # Demonstrate the registry system
    demonstrate_registry()
    
    # Demonstrate factory functions
    demonstrate_factories()
    
    # Demonstrate chain integration
    demonstrate_chain()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
