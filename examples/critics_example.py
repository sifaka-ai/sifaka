"""
Example of using LAC (Language Agent Correction) critics in Sifaka.

This example demonstrates how to use the LAC critic approach from the paper:
"Language Feedback Improves Language Model-based Decision Making"
(Fan et al., 2023, https://arxiv.org/abs/2403.03692)

The LAC approach combines:
1. A feedback critic that provides natural language feedback
2. A value critic that provides numerical scores

Together, these components guide the improvement of text through a structured
process that leverages both qualitative feedback and quantitative evaluation.
"""

import sys
import os
import argparse
import time

# Add the project root to the path so we can import sifaka
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sifaka.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators import length
from sifaka.critics.lac import create_lac_critic


def demonstrate_lac_direct_usage(model_name, text, temperature=0.7):
    """Demonstrate direct usage of the LAC critic without a chain.

    This shows how to use the LAC critic directly for text evaluation and improvement.

    Args:
        model_name: The model to use (e.g., "openai:gpt-4")
        text: The text to evaluate and improve
        temperature: The temperature to use for generation

    Returns:
        A tuple of (original_text, improved_text, critique)
    """
    print("\n=== Direct LAC Critic Usage ===")

    # Create a model
    provider, model_id = model_name.split(":", 1)
    model = create_model(provider, model_id)

    # Create a LAC critic
    lac_critic = create_lac_critic(
        model=model,
        temperature=temperature,
        feedback_weight=0.7,
        max_improvement_iterations=2,
        system_prompt=(
            "You are an expert language model that provides both detailed feedback "
            "and numerical evaluation to improve text quality following the "
            "Language Agent Correction (LAC) approach."
        ),
    )

    # Evaluate the text
    print("Evaluating text...")
    critique = lac_critic._critique(text)

    # Print critique results
    print(f"\nQuality Score: {critique['score']:.2f}/1.0")
    print(f"Needs Improvement: {critique['needs_improvement']}")
    print("\nFeedback:")
    print("-" * 40)
    print(critique["feedback"])
    print("-" * 40)

    # Improve the text
    print("\nImproving text...")
    improved_text = lac_critic._improve(text, critique)

    # Evaluate the improved text
    improved_critique = lac_critic._critique(improved_text)

    # Print improvement results
    print(f"\nImproved Quality Score: {improved_critique['score']:.2f}/1.0")
    print(f"Improvement: {improved_critique['score'] - critique['score']:.2f}")

    return text, improved_text, critique


def demonstrate_lac_in_chain(model_name, prompt, temperature=0.7, max_tokens=500):
    """Demonstrate using the LAC critic in a Sifaka chain.

    This shows how to integrate the LAC critic into a Sifaka chain for text generation,
    validation, and improvement.

    Args:
        model_name: The model to use (e.g., "openai:gpt-4")
        prompt: The prompt to use for generation
        temperature: The temperature to use for generation
        max_tokens: The maximum number of tokens to generate

    Returns:
        The chain result
    """
    print("\n=== LAC Critic in Chain ===")

    # Create a model
    provider, model_id = model_name.split(":", 1)
    model = create_model(provider, model_id)

    # Create a LAC critic
    lac_critic = create_lac_critic(
        model=model, temperature=temperature, feedback_weight=0.7, max_improvement_iterations=2
    )

    # Create a chain
    chain = Chain()

    # Configure the chain
    chain.with_model(model_name)
    chain.with_prompt(prompt)

    # Add validators
    chain.validate_with(length(min_words=50, max_words=500))

    # Add the LAC critic as an improver
    chain.improve_with(lac_critic)

    # Set options
    chain.with_options(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Run the chain
    print("\nRunning chain...")
    start_time = time.time()
    result = chain.run()
    elapsed_time = time.time() - start_time

    # Print the result
    print(f"\nChain completed in {elapsed_time:.2f} seconds")
    print(f"Validation Passed: {result.passed}")

    print("\nGenerated Text:")
    print("=" * 40)
    print(result.text)
    print("=" * 40)

    print("\nValidation Results:")
    for i, validation_result in enumerate(result.validation_results):
        print(f"  {i+1}. Passed: {validation_result.passed}")
        print(f"     Message: {validation_result.message}")

    print("\nImprovement Results:")
    for i, improvement_result in enumerate(result.improvement_results):
        print(f"  {i+1}. Changes Made: {improvement_result.changes_made}")
        print(f"     Message: {improvement_result.message}")

    return result


def main():
    """Run the LAC critic example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demonstrate LAC critics in Sifaka")
    parser.add_argument("--model", default="openai:gpt-4", help="Model to use for critics")
    parser.add_argument(
        "--prompt", default="Write a short explanation of quantum computing.", help="Prompt to use"
    )
    parser.add_argument("--text", help="Text to evaluate directly (bypasses generation)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens to generate")
    parser.add_argument(
        "--mode",
        choices=["direct", "chain", "both"],
        default="both",
        help="Mode to demonstrate (direct LAC usage, chain integration, or both)",
    )
    args = parser.parse_args()

    try:
        # If text is provided, use it directly
        if args.text and (args.mode == "direct" or args.mode == "both"):
            demonstrate_lac_direct_usage(args.model, args.text, args.temperature)

        # If no text is provided or chain mode is selected, run the chain example
        if not args.text or args.mode == "chain" or args.mode == "both":
            demonstrate_lac_in_chain(args.model, args.prompt, args.temperature, args.max_tokens)

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This example requires the OpenAI package to be installed.")
        print("Install it with: pip install openai tiktoken")
        return 1


if __name__ == "__main__":
    sys.exit(main())
