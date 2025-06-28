"""Example of using Meta-Rewarding critic for self-evaluation.

Meta-Rewarding uses a two-stage process to evaluate and improve text quality.
This example uses Anthropic Claude for nuanced meta-evaluation.
"""

import asyncio
import os
from sifaka import improve, Config
from sifaka.storage.file import FileStorage


async def main() -> None:
    """Run Meta-Rewarding improvement example with Anthropic."""

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

    # Run improvement with Meta-Rewarding critic using Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        print("🤖 Using Anthropic Claude for meta-evaluation...")
        result = await improve(
            text,
            critics=["meta_rewarding"],
            max_iterations=2,
            provider="anthropic",
            model="claude-3-haiku-20240307",
            config=Config(
                critic_model="claude-3-haiku-20240307",
                temperature=0.6,  # Lower for consistent evaluation
            ),
            storage=FileStorage(),
        )
    elif os.getenv("GOOGLE_API_KEY"):
        print("🌐 Using Google Gemini as fallback...")
        result = await improve(
            text,
            critics=["meta_rewarding"],
            max_iterations=2,
            provider="google",
            model="gemini-1.5-pro",  # Pro model for complex meta-evaluation
            config=Config(critic_model="gemini-1.5-pro", temperature=0.6),
            storage=FileStorage(),
        )
    else:
        print("❌ No API keys found. Please set ANTHROPIC_API_KEY or GOOGLE_API_KEY")
        return

    print(f"✅ Improved text ({len(result.final_text.split())} words):")
    print(result.final_text.strip())
    print(f"\n📊 Iterations: {result.iteration}")
    print(f"⏱️  Time: {result.processing_time:.2f}s")
    print("\n💡 Meta-Rewarding evaluates its own critique quality!")


if __name__ == "__main__":
    # Note: Prefers ANTHROPIC_API_KEY for meta-evaluation, falls back to GOOGLE_API_KEY
    asyncio.run(main())
