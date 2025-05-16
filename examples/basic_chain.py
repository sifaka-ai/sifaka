"""
Basic example of using the Sifaka Chain API.

This example demonstrates how to create a simple chain with a mock model.
"""

import sys
import os

# Add the project root to the path so we can import sifaka
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any
from sifaka.chain import Chain
from sifaka.results import ValidationResult, ImprovementResult


class MockModel:
    """A simple mock model for demonstration purposes."""
    
    def __init__(self, name: str = "mock-model"):
        self.name = name
    
    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt."""
        print(f"Generating text with {self.name} for prompt: {prompt}")
        print(f"Options: {options}")
        return f"This is a generated response for: {prompt}"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(text.split())


class LengthValidator:
    """A simple validator that checks text length."""
    
    def __init__(self, min_words: int = 0, max_words: int = 100):
        self.min_words = min_words
        self.max_words = max_words
    
    def validate(self, text: str) -> ValidationResult:
        """Validate text length."""
        word_count = len(text.split())
        
        if word_count < self.min_words:
            return ValidationResult(
                passed=False,
                message=f"Text is too short ({word_count} words, minimum {self.min_words})"
            )
        
        if word_count > self.max_words:
            return ValidationResult(
                passed=False,
                message=f"Text is too long ({word_count} words, maximum {self.max_words})"
            )
        
        return ValidationResult(
            passed=True,
            message=f"Text length is within limits ({word_count} words)"
        )


class SimpleImprover:
    """A simple improver that adds a conclusion."""
    
    def improve(self, text: str) -> tuple[str, ImprovementResult]:
        """Add a conclusion to the text."""
        improved_text = f"{text}\n\nIn conclusion, this is an improved response."
        
        return improved_text, ImprovementResult(
            original_text=text,
            improved_text=improved_text,
            changes_made=True,
            message="Added a conclusion"
        )


def main():
    """Run the example."""
    # Create a mock model
    model = MockModel("example-model")
    
    # Create a chain with the model
    chain = Chain()
    
    # Configure the chain
    chain.with_model(model)
    chain.with_prompt("Tell me about Sifaka")
    chain.validate_with(LengthValidator(min_words=5, max_words=50))
    chain.improve_with(SimpleImprover())
    chain.with_options(temperature=0.7, max_tokens=100)
    
    # Run the chain
    print("\nRunning chain...")
    result = chain.run()
    
    # Print the result
    print("\nResult:")
    print(f"Text: {result.text}")
    print(f"Passed: {result.passed}")
    
    print("\nValidation Results:")
    for i, validation_result in enumerate(result.validation_results):
        print(f"  {i+1}. Passed: {validation_result.passed}")
        print(f"     Message: {validation_result.message}")
    
    print("\nImprovement Results:")
    for i, improvement_result in enumerate(result.improvement_results):
        print(f"  {i+1}. Changes Made: {improvement_result.changes_made}")
        print(f"     Message: {improvement_result.message}")
        print(f"     Original: {improvement_result.original_text}")
        print(f"     Improved: {improvement_result.improved_text}")


if __name__ == "__main__":
    main()
