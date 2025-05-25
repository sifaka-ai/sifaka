"""Basic Ollama example for Sifaka.

This example demonstrates how to use the Sifaka framework with Ollama models
for local LLM inference. It shows basic text generation and integration with
the Chain orchestrator.

Prerequisites:
- Ollama must be installed and running (ollama serve)
- A model must be available (e.g., ollama pull llama2)

Example usage:
    python examples/ollama/basic_ollama.py
"""

import logging
import os
from typing import Optional

from sifaka.core.chain import Chain
from sifaka.core.thought import Thought
from sifaka.models.base import create_model
from sifaka.utils.logging import configure_logging
from sifaka.validators.base import LengthValidator


def test_ollama_model_direct():
    """Test Ollama model directly without Chain."""
    print("=== Testing Ollama Model Direct ===")

    try:
        # Create an Ollama model
        model = create_model("ollama:llama3.2:1b")

        # Test basic generation
        prompt = "Write a very short story about a robot in exactly 2 sentences."
        print(f"Prompt: {prompt}")

        response = model.generate(prompt, temperature=0.7, max_tokens=300)
        print(f"Response: {response}")

        # Test token counting
        token_count = model.count_tokens(response)
        print(f"Token count: {token_count}")

        return True

    except Exception as e:
        print(f"Error testing Ollama model: {e}")
        return False


def test_ollama_with_thought():
    """Test Ollama model with Thought container."""
    print("\n=== Testing Ollama with Thought ===")

    try:
        # Create an Ollama model (using the model we just pulled)
        model = create_model("ollama:llama3.2:1b")

        # Create a thought
        thought = Thought(
            prompt="Write a very short story about a robot in exactly 2 sentences.",
            system_prompt="You are a creative writer who writes concise stories.",
        )

        # Generate with thought
        response, actual_prompt = model.generate_with_thought(
            thought, temperature=0.7, max_tokens=300
        )
        print(f"Generated text: {response}")
        print(f"Actual prompt used: {actual_prompt[:100]}...")

        return True

    except Exception as e:
        print(f"Error testing Ollama with Thought: {e}")
        return False


def test_ollama_with_chain():
    """Test Ollama model with Chain orchestrator."""
    print("\n=== Testing Ollama with Chain ===")

    try:
        # Create an Ollama model (using the model we just pulled)
        model = create_model("ollama:llama3.2:1b")

        # Create a chain with validation
        chain = (
            Chain()
            .with_model(model)
            .with_prompt("Write a very short story about a robot in exactly 2 sentences.")
            .validate_with(LengthValidator(min_length=10, max_length=500))
        )

        # Run the chain
        result = chain.run()

        print(f"Final result: {result.text}")
        print(f"Validation results: {result.validation_results}")
        print(f"Chain iterations: {result.iteration}")

        return True

    except Exception as e:
        print(f"Error testing Ollama with Chain: {e}")
        return False


def check_ollama_availability() -> bool:
    """Check if Ollama is available and has models."""
    try:
        from sifaka.models.ollama import OllamaConnection

        connection = OllamaConnection()
        if not connection.health_check():
            print("❌ Ollama server is not accessible")
            print("   Please ensure Ollama is running with 'ollama serve'")
            return False

        models = connection.list_models()
        if not models:
            print("❌ No Ollama models found")
            print("   Please pull a model with 'ollama pull llama2'")
            return False

        print(f"✅ Ollama is available with models: {', '.join(models)}")
        return True

    except Exception as e:
        print(f"❌ Error checking Ollama availability: {e}")
        return False


def main():
    """Main function to run all Ollama tests."""
    # Configure logging
    configure_logging(level=logging.INFO)

    print("Sifaka Ollama Integration Test")
    print("=" * 40)

    # Check if Ollama is available
    if not check_ollama_availability():
        print("\nPlease ensure Ollama is installed and running:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Start Ollama: ollama serve")
        print("3. Pull a model: ollama pull llama2")
        return

    # Run tests
    tests = [
        test_ollama_model_direct,
        test_ollama_with_thought,
        test_ollama_with_chain,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Ollama integration is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
