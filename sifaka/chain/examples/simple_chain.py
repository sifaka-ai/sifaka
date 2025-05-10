"""
Simple Chain Example

This example demonstrates how to create and use a simple chain with the Sifaka chain system.
It shows how to create a chain with a model and validators, and how to run it on a prompt.

## Usage
```bash
python -m sifaka.chain.examples.simple_chain
```
"""

import os
import sys
from typing import List

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from sifaka.chain import Chain
from sifaka.chain.interfaces import Model, Validator, ValidationResult


class SimpleModel(Model):
    """A simple model that generates text by appending to the prompt."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        return f"{prompt}\n\nThis is a simple generated response to demonstrate the chain system."

    async def generate_async(self, prompt: str) -> str:
        """Generate text asynchronously."""
        return self.generate(prompt)


class LengthValidator(Validator):
    """A validator that checks the length of the output."""

    def __init__(self, min_length: int, max_length: int):
        """Initialize the validator."""
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, output: str) -> ValidationResult:
        """Validate the output length."""
        length = len(output)

        if length < self.min_length:
            return ValidationResult(
                passed=False,
                message=f"Output too short: {length} < {self.min_length}",
                score=0.0,
                issues=[f"Output length ({length}) is less than minimum ({self.min_length})"],
                suggestions=["Make the output longer"],
            )

        if length > self.max_length:
            return ValidationResult(
                passed=False,
                message=f"Output too long: {length} > {self.max_length}",
                score=0.0,
                issues=[f"Output length ({length}) is greater than maximum ({self.max_length})"],
                suggestions=["Make the output shorter"],
            )

        return ValidationResult(passed=True, message="Length validation passed", score=1.0)

    async def validate_async(self, output: str) -> ValidationResult:
        """Validate the output length asynchronously."""
        return self.validate(output)


class KeywordValidator(Validator):
    """A validator that checks for required keywords in the output."""

    def __init__(self, required_keywords: List[str]):
        """Initialize the validator."""
        self.required_keywords = required_keywords

    def validate(self, output: str) -> ValidationResult:
        """Validate the output contains required keywords."""
        output_lower = output.lower()
        missing_keywords = []

        for keyword in self.required_keywords:
            if keyword.lower() not in output_lower:
                missing_keywords.append(keyword)

        if missing_keywords:
            return ValidationResult(
                passed=False,
                message=f"Missing required keywords: {', '.join(missing_keywords)}",
                score=0.0,
                issues=[f"Output is missing required keywords: {', '.join(missing_keywords)}"],
                suggestions=[f"Include the keywords: {', '.join(missing_keywords)}"],
            )

        return ValidationResult(passed=True, message="Keyword validation passed", score=1.0)

    async def validate_async(self, output: str) -> ValidationResult:
        """Validate the output contains required keywords asynchronously."""
        return self.validate(output)


def main():
    """Run the example."""
    # Create components
    model = SimpleModel()
    validators = [
        LengthValidator(min_length=10, max_length=1000),
        KeywordValidator(required_keywords=["simple", "chain"]),
    ]

    # Create chain
    chain = Chain(
        model=model,
        validators=validators,
        max_attempts=3,
        name="simple_chain",
        description="A simple chain for demonstration",
    )

    # Run chain
    prompt = "Write a short story about a robot."
    print(f"Running chain with prompt: '{prompt}'")
    result = chain.run(prompt)

    # Print result
    print("\nResult:")
    print(f"Output: {result.output}")
    print(f"All validations passed: {result.all_passed}")
    print(f"Validation score: {result.validation_score}")
    print(f"Execution time: {result.execution_time:.4f} seconds")
    print(f"Attempt count: {result.attempt_count}")

    # Print validation results
    print("\nValidation Results:")
    for i, validation_result in enumerate(result.validation_results):
        print(f"Validation {i+1}:")
        print(f"  Passed: {validation_result.passed}")
        print(f"  Message: {validation_result.message}")
        print(f"  Score: {validation_result.score}")

        if validation_result.issues:
            print(f"  Issues: {', '.join(validation_result.issues)}")

        if validation_result.suggestions:
            print(f"  Suggestions: {', '.join(validation_result.suggestions)}")

    # Print chain statistics
    print("\nChain Statistics:")
    stats = chain.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
