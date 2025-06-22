"""Simple example showing how to use the new Sifaka."""

import asyncio
import os
from sifaka import improve
from sifaka.validators import LengthValidator, ContentValidator


async def basic_example():
    """Basic text improvement example."""
    print("🚀 Basic Example: Simple text improvement")

    result = await improve(
        "Write about renewable energy benefits", max_iterations=2, critics=["reflexion"]
    )

    print(f"Original: {result.original_text}")
    print(f"Final: {result.final_text}")
    print(f"Iterations: {result.iteration}")
    # print(f"Total cost: ${result.total_cost:.4f}")  # Not implemented yet
    print(f"Processing time: {result.processing_time:.2f}s")
    print()


async def validators_example():
    """Example with custom validators."""
    print("🎯 Validators Example: Text improvement with quality checks")

    result = await improve(
        "Write about climate change",
        max_iterations=3,
        critics=["reflexion", "constitutional"],
        validators=[
            LengthValidator(min_length=100, max_length=500),
            ContentValidator(required_terms=["climate", "environment"]),
        ],
    )

    print(f"Final text: {result.final_text}")
    print(f"Validations run: {len(result.validations)}")

    # Show validation results
    for validation in result.validations:
        status = "✅ PASSED" if validation.passed else "❌ FAILED"
        print(f"  {validation.validator}: {status} - {validation.details}")

    print(f"Critiques received: {len(result.critiques)}")
    for critique in result.critiques:
        print(f"  {critique.critic}: {critique.feedback[:100]}...")
    print()


async def multiple_critics_example():
    """Example with multiple critics."""
    print("🔬 Research Critics Example: Using multiple research techniques")

    result = await improve(
        "Explain quantum computing basics",
        max_iterations=4,
        critics=["reflexion", "constitutional", "n_critics", "self_rag"],
        validators=[LengthValidator(min_length=200)],
    )

    print(f"Final text length: {len(result.final_text)} characters")
    print(f"Improvement iterations: {result.iteration}")

    # Show the audit trail
    print("\n📊 Audit Trail:")
    for i, generation in enumerate(result.generations):
        print(
            f"Generation {i+1}: {len(generation.text)} chars (${generation.cost:.4f})"
        )

    print("\n🔍 Critics Applied:")
    critics_used = set(c.critic for c in result.critiques)
    for critic in critics_used:
        critic_critiques = [c for c in result.critiques if c.critic == critic]
        print(f"  {critic}: {len(critic_critiques)} critiques")
    print()


async def main():
    """Run all examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable to run examples")
        print("   export OPENAI_API_KEY=your_key_here")
        return

    print("🎉 Sifaka Examples - Simple AI Text Improvement\n")

    try:
        await basic_example()
        await validators_example()
        await multiple_critics_example()

        print("✅ All examples completed successfully!")
        print("\n💡 Key Benefits:")
        print("   • Simple API - just call improve()")
        print("   • Complete observability - see every step")
        print("   • Memory-bounded - no OOM crashes")
        print("   • Research-backed - proven techniques")
        print("   • Cost-controlled - set spending limits")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure you have:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Installed sifaka: pip install -e .")
        print("3. Internet connection for OpenAI API")


if __name__ == "__main__":
    asyncio.run(main())
