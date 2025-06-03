#!/usr/bin/env python3
"""Input validation example for Sifaka.

This example demonstrates:
1. Prompt validation with different constraints
2. Parameter validation (max_iterations, timeout, etc.)
3. Model name validation
4. Error handling and suggestions
5. Custom validation parameters

Run this example to see validation utilities in action:
    python examples/validation_example.py
"""

from pathlib import Path

# Add the project root to the path so we can import sifaka
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sifaka.utils import (
    validate_prompt,
    validate_max_iterations,
    validate_model_name,
    validate_timeout,
    ValidationError,
    configure_for_development,
    get_logger,
)

# Setup logging
configure_for_development()
logger = get_logger(__name__)


def demonstrate_prompt_validation():
    """Demonstrate prompt validation with various inputs."""
    print("\n1. Prompt Validation")
    print("-" * 30)
    
    # Valid prompts
    valid_prompts = [
        "Write about renewable energy",
        "Explain quantum computing in simple terms",
        "  Describe the benefits of exercise  ",  # Should be trimmed
        "A" * 100,  # Long but valid prompt
    ]
    
    print("Valid prompts:")
    for prompt in valid_prompts:
        try:
            validated = validate_prompt(prompt)
            print(f"  ✓ '{prompt[:30]}...' → '{validated[:30]}...'")
        except ValidationError as e:
            print(f"  ✗ '{prompt[:30]}...': {e.message}")
    
    # Invalid prompts
    invalid_prompts = [
        "",  # Empty
        "   ",  # Only whitespace
        "A",  # Too short (with custom min_length)
        "A" * 15000,  # Too long
        123,  # Wrong type
    ]
    
    print("\nInvalid prompts:")
    for prompt in invalid_prompts:
        try:
            if isinstance(prompt, str) and len(prompt) == 1:
                # Test with custom min_length
                validate_prompt(prompt, min_length=5)
            else:
                validate_prompt(prompt)
            print(f"  ✗ {repr(prompt)}: Should have failed")
        except ValidationError as e:
            print(f"  ✓ {repr(prompt)}: {e.message}")
            if e.suggestions:
                print(f"    Suggestions: {', '.join(e.suggestions[:2])}")


def demonstrate_iterations_validation():
    """Demonstrate max_iterations validation."""
    print("\n2. Max Iterations Validation")
    print("-" * 30)
    
    # Valid iterations
    valid_iterations = [1, 3, 5, 10, 20]
    
    print("Valid iterations:")
    for iterations in valid_iterations:
        try:
            validated = validate_max_iterations(iterations)
            print(f"  ✓ {iterations} → {validated}")
        except ValidationError as e:
            print(f"  ✗ {iterations}: {e.message}")
    
    # Invalid iterations
    invalid_iterations = [
        0,  # Too low
        -1,  # Negative
        25,  # Too high
        3.5,  # Wrong type
        "5",  # String
    ]
    
    print("\nInvalid iterations:")
    for iterations in invalid_iterations:
        try:
            validate_max_iterations(iterations)
            print(f"  ✗ {iterations}: Should have failed")
        except ValidationError as e:
            print(f"  ✓ {iterations}: {e.message}")


def demonstrate_model_validation():
    """Demonstrate model name validation."""
    print("\n3. Model Name Validation")
    print("-" * 30)
    
    # Valid model names
    valid_models = [
        "openai:gpt-4",
        "anthropic:claude-3-sonnet",
        "gemini-1.5-flash",
        "groq:llama-3.1-8b-instant",
        "ollama:mistral",
        "provider:custom-model-name-123",
    ]
    
    print("Valid model names:")
    for model in valid_models:
        try:
            validated = validate_model_name(model)
            print(f"  ✓ '{model}' → '{validated}'")
        except ValidationError as e:
            print(f"  ✗ '{model}': {e.message}")
    
    # Invalid model names
    invalid_models = [
        "",  # Empty
        "   ",  # Only whitespace
        "gpt-4",  # Missing provider
        ":gpt-4",  # Empty provider
        "openai:",  # Empty model
        "openai::",  # Double colon
        123,  # Wrong type
    ]
    
    print("\nInvalid model names:")
    for model in invalid_models:
        try:
            validate_model_name(model)
            print(f"  ✗ {repr(model)}: Should have failed")
        except ValidationError as e:
            print(f"  ✓ {repr(model)}: {e.message}")


def demonstrate_timeout_validation():
    """Demonstrate timeout validation."""
    print("\n4. Timeout Validation")
    print("-" * 30)
    
    # Valid timeouts
    valid_timeouts = [0.1, 1.0, 30.0, 60.0, 120.0, 300.0]
    
    print("Valid timeouts:")
    for timeout in valid_timeouts:
        try:
            validated = validate_timeout(timeout)
            print(f"  ✓ {timeout} → {validated}")
        except ValidationError as e:
            print(f"  ✗ {timeout}: {e.message}")
    
    # Invalid timeouts
    invalid_timeouts = [
        0.0,  # Too low
        -1.0,  # Negative
        500.0,  # Too high
        "30",  # String
        None,  # None
    ]
    
    print("\nInvalid timeouts:")
    for timeout in invalid_timeouts:
        try:
            validate_timeout(timeout)
            print(f"  ✗ {timeout}: Should have failed")
        except ValidationError as e:
            print(f"  ✓ {timeout}: {e.message}")


def demonstrate_custom_validation():
    """Demonstrate validation with custom parameters."""
    print("\n5. Custom Validation Parameters")
    print("-" * 30)
    
    # Custom prompt validation
    print("Custom prompt validation (min=10, max=100):")
    test_prompts = [
        "Short",  # Too short
        "This is a medium length prompt that should work fine",  # Good
        "A" * 150,  # Too long
    ]
    
    for prompt in test_prompts:
        try:
            validated = validate_prompt(prompt, min_length=10, max_length=100)
            print(f"  ✓ '{prompt[:30]}...' → Valid")
        except ValidationError as e:
            print(f"  ✗ '{prompt[:30]}...': {e.message}")
    
    # Custom iterations validation
    print("\nCustom iterations validation (min=2, max=5):")
    test_iterations = [1, 3, 6]
    
    for iterations in test_iterations:
        try:
            validated = validate_max_iterations(iterations, min_value=2, max_value=5)
            print(f"  ✓ {iterations} → Valid")
        except ValidationError as e:
            print(f"  ✗ {iterations}: {e.message}")
    
    # Custom timeout validation
    print("\nCustom timeout validation (min=5.0, max=60.0):")
    test_timeouts = [1.0, 30.0, 120.0]
    
    for timeout in test_timeouts:
        try:
            validated = validate_timeout(timeout, min_value=5.0, max_value=60.0)
            print(f"  ✓ {timeout} → Valid")
        except ValidationError as e:
            print(f"  ✗ {timeout}: {e.message}")


def demonstrate_error_details():
    """Demonstrate detailed error information."""
    print("\n6. Detailed Error Information")
    print("-" * 30)
    
    try:
        validate_prompt("", min_length=1, max_length=1000)
    except ValidationError as e:
        print("ValidationError details:")
        print(f"  Message: {e.message}")
        print(f"  Error code: {e.error_code}")
        print(f"  Context: {e.context}")
        print(f"  Suggestions: {e.suggestions}")
        print(f"\nFull error string:")
        print(f"  {str(e)}")


def main():
    """Run all validation examples."""
    logger.info("Starting validation examples")
    
    print("Sifaka Input Validation Examples")
    print("=" * 40)
    
    try:
        demonstrate_prompt_validation()
        demonstrate_iterations_validation()
        demonstrate_model_validation()
        demonstrate_timeout_validation()
        demonstrate_custom_validation()
        demonstrate_error_details()
        
        print("\n" + "=" * 40)
        print("All validation examples completed successfully!")
        
        logger.info("Validation examples completed successfully")
        
    except Exception as e:
        logger.error(
            "Validation examples failed",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True
        )
        print(f"\nExamples failed: {type(e).__name__} - {str(e)}")
        raise


if __name__ == "__main__":
    main()
