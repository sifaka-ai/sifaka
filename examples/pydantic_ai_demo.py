"""Demonstration of the new PydanticAI-based Sifaka system."""

import asyncio
import sys

sys.path.insert(0, "..")

from sifaka import (
    improve,
    improve_sync,
    improve_stream,
    configure,
    quick_improve,
)
from sifaka.validators.basic import LengthValidator, PatternValidator


async def basic_improvement_demo():
    """Basic text improvement using PydanticAI agents."""
    print("=== Basic Improvement Demo ===\n")

    text = """
    Artificial intelligence is a technology that enables machines to simulate human
    intelligence. It includes various subfields like machine learning, natural language
    processing, and computer vision. AI has many applications in different industries.
    """

    # Configure with Logfire monitoring (optional)
    configure(
        model="gpt-4o-mini",
        temperature=0.7,
        max_iterations=2,
        # logfire_token="your-token-here",  # Enable monitoring
    )

    # Improve with single critic
    print("Improving with Reflexion critic...")
    result = await improve(
        text,
        critics=["reflexion"],
    )

    print(f"Original: {text[:100]}...")
    print(f"\nImproved: {result.final_text[:200]}...")
    print(f"Iterations: {result.iteration}")
    print(f"Final confidence: {result.get_final_confidence():.2f}")
    print()


async def multi_critic_demo():
    """Demonstrate multiple critics working together."""
    print("=== Multi-Critic Demo ===\n")

    text = "AI is good for business. Companies should use AI to make more money."

    # Use multiple critics
    result = await improve(
        text,
        critics=["constitutional", "self_refine", "meta_rewarding"],
        max_iterations=3,
        context={"style": "business"},
    )

    print(f"Original: {text}")
    print(f"\nImproved: {result.final_text}")

    # Show critique progression
    print("\nCritique progression:")
    for i, critique in enumerate(result.critiques):
        print(f"  {i+1}. {critique.critic}: {critique.feedback[:100]}...")
    print()


async def streaming_demo():
    """Demonstrate streaming improvements."""
    print("=== Streaming Demo ===\n")

    text = "Machine learning is when computers learn from data."

    print("Streaming improvements...")
    async for partial in improve_stream(
        text,
        critics=["self_refine"],
        max_iterations=1,
    ):
        # In a real app, you might update UI here
        print(f"\rProgress: {partial.tokens_generated} tokens...", end="")

    print(f"\n\nFinal: {partial.text_so_far}")
    print()


async def validation_demo():
    """Demonstrate validators with PydanticAI."""
    print("=== Validation Demo ===\n")

    text = "Buy now! Limited offer! Act fast! Best deal ever!"

    # Create validators
    validators = [
        LengthValidator(min_length=50, max_length=500),
        PatternValidator(
            pattern=r"(!)\1+",  # No multiple exclamation marks
            should_match=False,
            error_message="Avoid excessive exclamation marks",
        ),
    ]

    # Improve with validation
    result = await improve(
        text,
        critics=["constitutional", "self_refine"],
        validators=validators,
        context={"style": "professional"},
    )

    print(f"Original: {text}")
    print(f"\nImproved: {result.final_text}")

    # Show validation results
    print("\nValidation results:")
    for validation in result.validations:
        print(f"  {validation.validator}: {validation.passed}")
    print()


async def custom_prompt_critic_demo():
    """Demonstrate custom prompt critic."""
    print("=== Custom Prompt Critic Demo ===\n")

    text = """
    The quantum computer uses qubits instead of regular bits. This allows it to
    perform certain calculations much faster than classical computers.
    """

    # Use prompt critic with custom template
    result = await improve(
        text,
        critics=["prompt"],
        context={
            "critique_template": "technical",
        },
    )

    print(f"Improved technical documentation: {result.final_text[:200]}...")
    print()


async def n_critics_demo():
    """Demonstrate N-Critics multi-perspective critique."""
    print("=== N-Critics Demo ===\n")

    text = """
    Our new software solution increases productivity by 50%. It uses advanced
    algorithms to optimize workflows and reduce manual tasks.
    """

    # Use N-Critics for multi-perspective feedback
    result = await improve(
        text,
        critics=["n_critics"],
        max_iterations=2,
    )

    print(f"Original: {text}")
    print(f"\nImproved with multiple perspectives: {result.final_text}")

    # Show perspectives used
    if result.critiques:
        metadata = result.critiques[-1].metadata
        if "perspectives_used" in metadata:
            print(f"\nPerspectives used: {', '.join(metadata['perspectives_used'])}")
    print()


async def quick_improve_demo():
    """Demonstrate quick improvement function."""
    print("=== Quick Improve Demo ===\n")

    text = "Python is a programming language used for many things like web development and data science."

    improved = await quick_improve(text, style="technical")

    print(f"Original: {text}")
    print(f"\nQuick improved: {improved}")
    print()


def sync_demo():
    """Demonstrate synchronous API."""
    print("=== Sync API Demo ===\n")

    text = "Sifaka helps improve text using AI critics."

    # Synchronous improvement
    result = improve_sync(
        text,
        critics=["meta_rewarding"],
        max_iterations=1,
    )

    print(f"Original: {text}")
    print(f"Improved (sync): {result.final_text}")
    print()


async def main():
    """Run all demos."""
    print("🚀 Sifaka PydanticAI Demo\n")

    # Run demos
    await basic_improvement_demo()
    await multi_critic_demo()
    await streaming_demo()
    await validation_demo()
    await custom_prompt_critic_demo()
    await n_critics_demo()
    await quick_improve_demo()

    # Sync demo
    sync_demo()

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
