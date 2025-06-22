"""Example: Using the N-Critics for multi-perspective ensemble evaluation."""

import asyncio

from sifaka import improve


async def n_critics_example():
    """Demonstrate N-Critics with thought logging."""

    print("👥 N-Critics Ensemble Example")
    print("=" * 50)

    # Complex text that benefits from multiple perspectives
    original_text = """
    Climate change is happening. Scientists agree it's caused by human activities.
    We should do something about it. Renewable energy is one solution.
    Governments and businesses need to work together.
    """

    print(f"Original text: {original_text.strip()}")
    print("\n🤖 Running N-Critics ensemble...")

    # Use n-critics specifically
    result = await improve(
        original_text.strip(),
        critics=["n_critics"],
        max_iterations=2,
        model="gpt-4o-mini",
        show_improvement_prompt=True,
    )

    print(f"\n✨ Final improved text:")
    print(f"'{result.final_text}'")
    print(f"\n📊 Results:")
    print(f"- Iterations: {result.iteration}")
    print(f"- Processing time: {result.processing_time:.2f}s")

    # Calculate average confidence from critiques
    if result.critiques:
        confidences = [
            c.confidence for c in result.critiques if c.confidence is not None
        ]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            print(f"- Average critique confidence: {avg_confidence:.2f}")

    # Save extra_data for perspective breakdown display
    extra_data = {
        "ensemble_analysis": {
            "critic_perspectives": [
                {
                    "name": "Scientific Accuracy Critic",
                    "focus": "Factual correctness and evidence-based claims",
                    "weight": 0.25,
                },
                {
                    "name": "Clarity and Communication Critic",
                    "focus": "Readability and clear communication",
                    "weight": 0.25,
                },
                {
                    "name": "Practical Solutions Critic",
                    "focus": "Actionable recommendations and feasibility",
                    "weight": 0.25,
                },
                {
                    "name": "Stakeholder Impact Critic",
                    "focus": "Consideration of different stakeholder perspectives",
                    "weight": 0.25,
                },
            ],
        }
    }

    # Show ensemble analysis
    print(f"\n👥 Ensemble Analysis:")
    for i, critique in enumerate(result.critiques):
        if critique.critic == "n_critics":
            print(f"\n  Iteration {i + 1}:")
            print(f"  - Multi-perspective feedback: {critique.feedback[:120]}...")
            print(f"  - Diverse suggestions: {len(critique.suggestions)}")
            print(f"  - Consensus for improvement: {critique.needs_improvement}")
            print(f"  - Ensemble confidence: {critique.confidence:.2f}")

    # Show perspective breakdown
    print(f"\n🔍 Perspective Breakdown:")
    perspectives = extra_data["ensemble_analysis"]["critic_perspectives"]
    for perspective in perspectives:
        print(f"  • {perspective['name']}: {perspective['focus']}")

    return result


async def main():
    """Run the n-critics example."""
    try:
        result = await n_critics_example()
        print(f"\n✅ N-Critics ensemble example completed successfully!")

    except Exception as e:
        print(f"\n❌ Error running n-critics example: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
