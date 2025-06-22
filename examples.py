"""Comprehensive examples demonstrating Sifaka capabilities."""

import asyncio
import os
from sifaka import improve
from sifaka.validators import LengthValidator
from sifaka.storage import FileStorage, MemoryStorage


async def basic_example():
    """Basic text improvement with default settings."""
    print("=== Basic Example ===")

    result = await improve(
        "Write about the benefits of renewable energy for the environment and economy"
    )

    print(f"Original: {result.original_text}")
    print(f"Final: {result.final_text}")
    print(f"Iterations: {result.iteration}")
    print()


async def multi_critic_example():
    """Using multiple critics for comprehensive evaluation."""
    print("=== Multi-Critic Example ===")

    result = await improve(
        "Explain quantum computing and its applications in cryptography",
        critics=["reflexion", "constitutional", "self_refine"],
        max_iterations=3,
    )

    print(f"Critics used: {[c.critic for c in result.critiques]}")
    print(f"Final quality score: {result.confidence:.2f}")

    for i, critique in enumerate(result.critiques):
        print(f"\nCritic {i+1} ({critique.critic}):")
        print(f"  Feedback: {critique.feedback[:100]}...")
        print(f"  Suggestions: {len(critique.suggestions)} items")
        print(f"  Confidence: {critique.confidence:.2f}")
    print()


async def advanced_critics_example():
    """Advanced critics: N-Critics, Self-RAG, Meta-Rewarding."""
    print("=== Advanced Critics Example ===")

    result = await improve(
        "Write a research paper abstract about machine learning in healthcare",
        critics=["n_critics", "self_rag", "meta_rewarding"],
        max_iterations=2,
    )

    print(f"Advanced critique results:")
    for critique in result.critiques:
        print(f"  {critique.critic}: {critique.needs_improvement}")
    print()


async def validation_example():
    """Using validators to enforce quality constraints."""
    print("=== Validation Example ===")

    try:
        result = await improve(
            "Brief note on AI",
            validators=[LengthValidator(min_length=100, max_length=500)],
            max_iterations=3,
        )

        print(f"Validation passed: {result.all_passed}")
        print(f"Final length: {len(result.final_text)}")

    except Exception as e:
        print(f"Validation failed: {e}")
    print()


async def iteration_control_example():
    """Demonstrating iteration controls."""
    print("=== Iteration Control Example ===")

    result = await improve(
        "Write about sustainable agriculture practices",
        max_iterations=2,
        model="gpt-4o-mini",  # Fast model
    )

    print(f"Iterations completed: {result.iteration}")
    print()


async def storage_example():
    """Using different storage backends."""
    print("=== Storage Example ===")

    # File storage
    file_storage = FileStorage(storage_dir="sifaka_results")

    result = await improve(
        "Explain photosynthesis process", storage=file_storage, critics=["reflexion"]
    )

    print(f"Result saved with ID: {result.id}")

    # Load it back
    loaded = await file_storage.load(result.id)
    print(f"Loaded result: {loaded.original_text[:50]}...")
    print()


async def self_consistency_example():
    """Self-Consistency for reliable assessment."""
    print("=== Self-Consistency Example ===")

    result = await improve(
        "Analyze the pros and cons of remote work",
        critics=["self_consistency"],
        max_iterations=2,
    )

    # Self-consistency provides consensus-based feedback
    for critique in result.critiques:
        if critique.critic == "self_consistency":
            print(f"Consensus feedback: {critique.feedback}")
            print(f"Consistency confidence: {critique.confidence:.2f}")
    print()


async def constitutional_ai_example():
    """Constitutional AI with principle-based evaluation."""
    print("=== Constitutional AI Example ===")

    result = await improve(
        "Write guidelines for ethical AI development",
        critics=["constitutional"],
        max_iterations=2,
    )

    for critique in result.critiques:
        if critique.critic == "constitutional":
            print(f"Constitutional feedback: {critique.feedback}")
            print(f"Ethical compliance: {critique.confidence:.2f}")
    print()


async def iterative_improvement_example():
    """Showing the iterative improvement process."""
    print("=== Iterative Improvement Example ===")

    result = await improve(
        "Explain machine learning",
        critics=["reflexion", "self_refine"],
        max_iterations=4,
    )

    print("Improvement trajectory:")
    print(f"Generation 0: {result.original_text}")

    for i, generation in enumerate(result.generations):
        print(f"Generation {i+1}: {generation.text[:100]}...")
        print(f"  Model: {generation.model}")
    print()


async def comprehensive_analysis_example():
    """Comprehensive analysis with all available critics."""
    print("=== Comprehensive Analysis Example ===")

    result = await improve(
        "Write about the future of artificial intelligence",
        critics=[
            "reflexion",  # Self-reflection and learning
            "constitutional",  # Ethical and principle-based
            "self_refine",  # Iterative self-improvement
            "n_critics",  # Multi-perspective ensemble
            "self_rag",  # Retrieval-augmented
            "meta_rewarding",  # Meta-evaluation
            "self_consistency",  # Consensus-based
        ],
        max_iterations=3,
    )

    print("Comprehensive Analysis Results:")
    print(f"Total critiques: {len(result.critiques)}")
    print(f"Final confidence: {result.confidence:.2f}")

    critic_summary = {}
    for critique in result.critiques:
        if critique.critic not in critic_summary:
            critic_summary[critique.critic] = []
        critic_summary[critique.critic].append(critique.confidence)

    print("\nCritic Performance Summary:")
    for critic, confidences in critic_summary.items():
        avg_confidence = sum(confidences) / len(confidences)
        print(
            f"  {critic}: {avg_confidence:.2f} avg confidence ({len(confidences)} runs)"
        )
    print()


async def error_handling_example():
    """Demonstrating error handling and robustness."""
    print("=== Error Handling Example ===")

    try:
        # This should work fine
        result = await improve("Short text", max_iterations=1, critics=["reflexion"])
        print("✓ Normal operation successful")

    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    try:
        # This should test error handling
        result = await improve(
            "Test text for error handling", max_iterations=1, critics=["reflexion"]
        )
        print("✓ Error handling test completed")

    except Exception as e:
        print(f"✗ Unexpected error in test: {type(e).__name__}")

    print()


async def main():
    """Run all examples."""
    print("Sifaka Comprehensive Examples")
    print("=" * 50)

    # Set up environment (in real usage, set OPENAI_API_KEY)
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No OPENAI_API_KEY found. Using mock responses for demo.")
        print("   Set OPENAI_API_KEY environment variable for real usage.\n")

    examples = [
        basic_example,
        multi_critic_example,
        advanced_critics_example,
        validation_example,
        iteration_control_example,
        storage_example,
        self_consistency_example,
        constitutional_ai_example,
        iterative_improvement_example,
        comprehensive_analysis_example,
        error_handling_example,
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Example {example.__name__} failed: {e}\n")

    print("=" * 50)
    print("Examples complete! See the output above for detailed results.")


if __name__ == "__main__":
    asyncio.run(main())
