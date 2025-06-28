"""Example of using Self-Consistency critic for consensus-based evaluation.

Self-Consistency evaluates multiple outputs to ensure consistent quality.
"""

import asyncio
from sifaka import improve


async def main() -> None:
    """Run Self-Consistency improvement example."""

    # Complex reasoning that benefits from consistency checking
    text = """
    To solve climate change, we need to stop using all fossil fuels immediately.
    This is the only solution that will work. Nothing else matters - not renewable
    energy, not carbon capture, just complete cessation of fossil fuel use today.
    """

    print("🔄 Self-Consistency Example - Consensus evaluation")
    print("=" * 50)
    print(f"Original text ({len(text.split())} words):")
    print(text.strip())
    print()

    # Run improvement with Self-Consistency critic
    result = await improve(text, critics=["self_consistency"], max_iterations=2)

    print(f"✅ Improved text ({len(result.final_text.split())} words):")
    print(result.final_text.strip())
    print(f"\n📊 Iterations: {result.iteration}")
    print(f"⏱️  Time: {result.processing_time:.2f}s")


if __name__ == "__main__":
    # Note: Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
    asyncio.run(main())
