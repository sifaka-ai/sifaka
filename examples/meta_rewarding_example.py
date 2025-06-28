"""Example of using Meta-Rewarding critic for self-evaluation.

Meta-Rewarding uses a two-stage process to evaluate and improve text quality.
"""

import asyncio
from sifaka import improve


async def main():
    """Run Meta-Rewarding improvement example."""

    # Technical explanation that needs quality evaluation
    text = """
    Machine learning is when computers learn from data. They use algorithms
    to find patterns. Deep learning is a type of machine learning that uses
    neural networks. It's very powerful and can do many things.
    """

    print("🏆 Meta-Rewarding Example - Self-evaluation")
    print("=" * 50)
    print(f"Original text ({len(text.split())} words):")
    print(text.strip())
    print()

    # Run improvement with Meta-Rewarding critic
    result = await improve(text, critics=["meta_rewarding"], max_iterations=2)

    print(f"✅ Improved text ({len(result.final_text.split())} words):")
    print(result.final_text.strip())
    print(f"\n📊 Iterations: {result.iteration}")
    print(f"⏱️  Time: {result.processing_time:.2f}s")


if __name__ == "__main__":
    # Note: Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
    asyncio.run(main())
