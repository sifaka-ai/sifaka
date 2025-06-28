"""Example of using N-Critics for multi-perspective evaluation.

N-Critics uses multiple critical perspectives to provide comprehensive feedback.
"""

import asyncio
from sifaka import improve
from sifaka.critics.n_critics import NCriticsCritic


async def main():
    """Run N-Critics improvement example."""

    # Business proposal that needs multiple perspectives
    text = """
    We should invest all our money in cryptocurrency because it's the future.
    Traditional investments are outdated. Bitcoin will definitely go to a million
    dollars, so we can't lose. We should act fast before it's too late.
    """

    print("👥 N-Critics Example - Multiple perspectives")
    print("=" * 50)
    print(f"Original text ({len(text.split())} words):")
    print(text.strip())
    print()

    # Two approaches: default perspectives or custom perspectives

    # Approach 1: Use default N-Critics
    print("\n1️⃣ Using default perspectives...")
    result1 = await improve(text, critics=["n_critics"], max_iterations=2)

    print(f"Improved: {result1.final_text[:100]}...")

    # Approach 2: Custom perspectives
    print("\n2️⃣ Using custom perspectives...")
    custom_critic = NCriticsCritic(
        perspectives={
            "Risk Analyst": "Evaluate financial risks and portfolio diversification",
            "Tech Expert": "Assess cryptocurrency technology and market maturity",
            "Financial Advisor": "Consider long-term wealth preservation strategies",
        }
    )

    result2 = await improve(text, critics=["n_critics"], max_iterations=2)

    print(f"✅ Final text ({len(result2.final_text.split())} words):")
    print(result2.final_text.strip())
    print(f"\n📊 Total iterations: {result1.iteration + result2.iteration}")
    print(f"⏱️  Total time: {result1.processing_time + result2.processing_time:.2f}s")


if __name__ == "__main__":
    # Note: Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
    asyncio.run(main())
