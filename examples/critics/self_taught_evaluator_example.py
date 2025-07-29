"""Example demonstrating the Self-Taught Evaluator critic.

This critic evaluates text by generating contrasting outputs and comparing them,
providing transparent reasoning traces for its evaluations.
"""

import asyncio

from sifaka import improve
from sifaka.critics import SelfTaughtEvaluatorCritic


async def main():
    # Example text that could benefit from improvement
    text = """
    The company's new product is really good and has many features that customers
    will probably like. It does things better than the old version and costs less
    money. People should buy it because it's worth the price.
    """

    print("Original text:")
    print(text)
    print("\n" + "=" * 50 + "\n")

    # Method 1: Using the critic directly
    print("Method 1: Direct critic usage")
    print("-" * 30)

    critic = SelfTaughtEvaluatorCritic(
        model="gpt-4o-mini",  # Can use any supported model
        temperature=0.7,
    )

    # Create a minimal result object for standalone usage
    from sifaka.core.models import SifakaResult

    result = SifakaResult(original_text=text, final_text=text)

    critique = await critic.critique(text, result)

    print(f"Feedback: {critique.feedback}\n")
    print(f"Needs improvement: {critique.needs_improvement}")
    print(f"Confidence: {critique.confidence:.2f}\n")

    if critique.suggestions:
        print("Suggestions:")
        for i, suggestion in enumerate(critique.suggestions, 1):
            print(f"{i}. {suggestion}")

    # Show contrasting outputs if available
    if "contrasting_outputs" in critique.metadata:
        print("\nContrasting outputs generated:")
        for i, output in enumerate(critique.metadata["contrasting_outputs"], 1):
            print(f"\n--- Version {i} ---")
            print(f"Text: {output.get('version', 'No text provided')}")
            print(f"Reasoning: {output.get('reasoning', '')}")
            print(f"Quality: {output.get('quality_assessment', '')}")

    # Debug: Show what's actually in metadata
    print(f"\nMetadata keys: {list(critique.metadata.keys())}")
    if "reasoning_trace" in critique.metadata:
        print(f"\nReasoning trace: {critique.metadata['reasoning_trace'][:200]}...")

    print("\n" + "=" * 50 + "\n")

    # Method 2: Using the improve function with the critic
    print("Method 2: Using improve() with self_taught_evaluator")
    print("-" * 30)

    result = await improve(text, critics=["self_taught_evaluator"], max_iterations=2)

    print("\nFinal improved text:")
    print(result.final_text)

    print(f"\nIterations: {result.iteration}")
    print(f"Total critiques: {len(result.critiques)}")

    # Show the reasoning traces from each iteration
    print("\nReasoning traces:")
    for i, critique in enumerate(result.critiques):
        if (
            critique.critic == "self_taught_evaluator"
            and "reasoning_trace" in critique.metadata
        ):
            print(
                f"\nIteration {i + 1}: {critique.metadata['reasoning_trace'][:200]}..."
            )


if __name__ == "__main__":
    asyncio.run(main())
